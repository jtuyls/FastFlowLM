# FastFlowLM on HRX (AMD XDNA2 NPU)

Run **unmodified** FastFlowLM on an AMD XDNA2 NPU through **HRX** (the IREE-based
runtime with an `amdxdna` HAL) instead of XRT — via a transparent interposer.
No changes to FLM's (closed-source) NPU engines are required.

```
$ ./scripts/run.sh run qwen3:0.6b
>>> What is the capital of France? One word.
**Paris**.
```

This directory lives on the FastFlowLM branch
**https://github.com/jtuyls/FastFlowLM/tree/perf-cull-final** and pairs with the
HRX `amdxdna` HAL in **https://github.com/ROCm/hrx-system/pull/37**.

---

## 1. Why this exists

FastFlowLM's per-model NPU engines (`lib/libqwen3_npu.so`, `libmha.so`, …) are
**prebuilt, closed-source, and hard-linked against XRT** — each imports ~33
`xrt::` symbols from `libxrt_coreutil.so.2` (verify with
`nm -DC src/lib/libqwen3_npu.so | grep ' U .*xrt::'`). They cannot be recompiled,
so a source port to HRX is impossible.

Instead, this layer **interposes `libxrt_coreutil.so.2`**: it provides those
`xrt::` symbols with an ABI-compatible layout but, on every kernel dispatch,
forwards the work to HRX's `amdxdna` backend. The closed engine is unaware — it
thinks it is talking to XRT.

## 2. How it works

The shim (`src/xrt_coreutil_shim.cpp`) implements the `xrt::` classes FLM uses
(`device`, `bo`, `ext::bo`, `hw_context`, `elf`, `module`, `ext::kernel`, `run`,
`runlist`, `xclbin`). At dispatch (`xrt::run::start` / `xrt::runlist::execute`),
with `FLM_FORWARD=1`, it:

1. Takes the control ELF FLM generated (the `aiebu` instruction-transaction blob
   it passes to `xrt::elf`) and extracts `.ctrltext` → TXN control code, and
   `.rela.dyn` → a `(offset, arg_idx, arg_plus)` patch table.
2. Wraps the kernel's `.xclbin` + control code + patch table into an HRX **XADX**
   executable (`src/hrx_xadx_builder.hpp`), cached by `.xclbin` + control-code
   hash.
3. Allocates an HRX device buffer per `xrt::bo`, syncs host↔device explicitly
   (`copy_h2d`/`copy_d2h`), and dispatches via `hrx_stream_dispatch`. HRX
   host-patches buffer addresses into the control code (npu4 `COMMAND_CHAIN`
   path) and runs it on the NPU.

Static weights upload to the device **once** (dirty tracking driven by
`xrt::bo::sync(TO_DEVICE)`), not per dispatch.

---

## 3. Prerequisites

### 3a. Hardware + NPU driver/firmware

- An **AMD XDNA2 NPU** (Strix Point / Kraken Point / Strix Halo), exposed at
  `/dev/accel/accel0`.
- An `amdxdna` kernel driver loaded (kernel ≥ 6.19 in-tree, or the out-of-tree
  DKMS driver), and **NPU firmware ≥ 1.1.x**.
  - Follow FastFlowLM's Linux setup: https://fastflowlm.com/docs/install_lin/
  - The firmware must report `1.1.x`. With the out-of-tree driver the loaded file
    is `/lib/firmware/amdnpu/17f0_10/npu.dev.sbin` (device `1022:17f0` rev10 =
    npu4 / Strix Point). A 1.1 build (e.g. `1.7_npu.sbin.1.1.2.64` from
    `gitlab.com/kernel-firmware/drm-firmware`, branch `amd-ipu-staging`) reports
    `1.1.2.64`. Reload the driver after swapping (`rmmod amdxdna; modprobe amdxdna`).
- **XRT userspace** installed at `/opt/xilinx/xrt` (provides `libxrt_coreutil.so.2`,
  which this shim **shadows** at runtime — the real XRT *NPU plugin* is **not**
  required; that is the whole point).
- `ulimit -l` should be `unlimited` (else add `* - memlock unlimited` to
  `/etc/security/limits.conf` and reboot).

Sanity: with the shim **not** active (plain XRT), `flm validate` should print
`NPU FW Version: 1.1.x`. (Through this interposer, FW ≥1.1 is what HRX expects.)

### 3b. HRX (with the amdxdna HAL) — ROCm/hrx-system PR #37 + perf patches

PR #37 provides the `amdxdna` HAL backend. On top of it, this directory ships two
small performance commits (the persistent host-mapped buffer API + a bounded
spin-then-block completion wait) as patches under
[`patches/`](patches/) — together 3 files / ~95 lines. Apply them after checking
out PR #37, then build:

```bash
git clone https://github.com/ROCm/hrx-system.git hrx
cd hrx
gh pr checkout 37          # or: git fetch origin pull/37/head:amdxdna && git checkout amdxdna

# Apply the two perf patches (each patch's header explains what/why).
# They were cut against PR #37's tip; `git am` keeps them as real commits.
git am /path/to/FastFlowLM/hrx-integration/patches/*.patch

python dev.py cmake setup
python dev.py cmake configure -DIREE_HAL_DRIVER_AMDXDNA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
python dev.py cmake build
```

> **Applying the patches.** Each `.patch` is a self-describing git mailbox file
> (commit message explains *what* and *why*; diff touches only
> `libhrx/include/hrx_runtime.h`, `libhrx/src/libhrx/buffer.c`, and
> `runtime/src/iree/hal/drivers/amdxdna/shim/linux/kmq/hwq.cpp`). If `git am`
> fails because PR #37 has moved since the patches were cut, **read the patch
> headers and reconcile by hand** — the changes are tiny and localized:
> - `0001` wraps `hw_q::wait_command` with a 50 µs busy-spin before the blocking
>   wait (in `hwq.cpp`).
> - `0002` adds a persistent host-mapped buffer API (`hrx_buffer_map_persistent`,
>   `hrx_buffer_flush_range`, `hrx_buffer_invalidate_range`) to `hrx_runtime.h` +
>   `buffer.c`.
>
> Fallbacks if `git am` won't apply cleanly: `git apply --3way patches/*.patch`
> (3-way merge against the current tree), or apply each hunk manually. The shim
> needs the `0002` persistent-map API symbols at build time, so don't skip it.

The build directory must contain `libhrx/src/libhrx/libhrx.so`,
`libflatcc_runtime.a`, `_deps/flatcc-src/include`, and
`runtime/src/iree/schemas/amdxdna_xclbin_executable_def_builder.h`. Note its path
— call it `HRX_BUILD` (and the checkout `HRX_DIR`). `build.sh` auto-detects
`$HRX_DIR/build-amdxdna-native` or `$HRX_DIR/build`; set `HRX_BUILD` explicitly
if dev.py used a different directory.

### 3c. FastFlowLM (this fork/branch)

```bash
git clone https://github.com/jtuyls/FastFlowLM.git
cd FastFlowLM
git checkout perf-cull-final
git submodule update --init --recursive        # tokenizers-cpp + nested

# build deps (Ubuntu 24.04): build-essential cmake ninja-build pkg-config
#   libboost-program-options-dev libcurl4-openssl-dev libreadline-dev
#   libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
#   libswresample-dev libdrm-dev uuid-dev libxrt-dev
# Rust >= 1.79 is required by tokenizers-cpp; the distro rustc may be too old:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

cd src
cmake --preset linux-default -DCARGO_EXECUTABLE="$HOME/.cargo/bin/cargo"
cmake --build --preset linux-default -j"$(nproc)"
# -> src/build/flm
```

Download the model (no NPU needed):

```bash
FLM_CONFIG_PATH="$PWD/model_list.json" build/flm pull qwen3:0.6b
```

---

## 4. Build the interposer shim

From `FastFlowLM/hrx-integration`:

```bash
export HRX_DIR=/path/to/hrx
export HRX_BUILD=/path/to/hrx/<build-dir>   # the dir from step 3b
./scripts/build.sh
# -> build/libxrt_coreutil.so.2  (should export the xrt:: symbols:
#    nm -D --defined-only build/libxrt_coreutil.so.2 | grep -c 3xrt   # ~47)
```

## 5. Run

```bash
# Ensure the NPU is idle (single shared device); then:
export HRX_BUILD=/path/to/hrx/<build-dir>    # so run.sh finds libhrx.so
./scripts/run.sh run   qwen3:0.6b            # interactive chat
./scripts/run.sh bench qwen3:0.6b            # benchmark (0.6B: full 1k–32k sweep)
# For 8B models, cap context to avoid 32k OOM (see §6):
./scripts/run.sh bench llama3.1:8b -i bench_configs/bench_16k.json
```

`run.sh` auto-detects the `flm` binary (`../src/build/flm`), `libhrx.so`, the
model list, and the xclbins; it symlinks `<flm-dir>/xclbins -> ../xclbins`, sets
`LD_LIBRARY_PATH` (shim first, then libhrx, then `/opt/xilinx/xrt/lib`), and
`FLM_FORWARD=1`. Override any with `FLM_BIN`, `HRX_BUILD`, `FLM_CONFIG_PATH`,
`FLM_FORWARD=0`.

**Expected:** coherent output (e.g. `2 + 2 = 4.`). The shim prints periodic
`[fwd] stats: dispatched=… h2d_copies=… h2d_skips=…` to stderr.

---

## 6. Benchmarking (XRT vs HRX)

This section is **self-contained**: once §1–§5 are done (HRX built, shim built,
models pulled, a `run` smoke-tests OK), follow the steps below end-to-end to
benchmark **Qwen3-0.6B, Llama-3.1-8B, and DeepSeek-R1-8B** on both runtimes and
emit a report. No further setup is needed.

We compare two paths on the **same machine**:

- **XRT (native baseline):** the real XRT userspace, **no shim**
  (`LD_LIBRARY_PATH=/opt/xilinx/xrt/lib`). Requires the real XRT XDNA plugin to
  be installed and the NPU reachable; if it is not, the XRT column is `N/A` and
  you report HRX-only.
- **HRX (this interposer):** `scripts/run.sh bench …` (shim → libhrx,
  `FLM_FORWARD=1`, command chaining on).

`flm bench` sweeps context in powers of two from 1k up to `max_length`, running
`iterations` passes per context (default 8), 32 decode tokens each. It writes
`bench_<model>_<YYYYMMDD>.csv` to the **current working directory** with
`prefill_avg_toks_per_s` and `decoding_avg_toks_per_s` columns (among others).

**Sweep ranges (per the request):**

| Model | Sweep | Config |
|---|---|---|
| `qwen3:0.6b` | 1k → **32k** | built-in default (`max_length` 32768) — no `-i` |
| `llama3.1:8b` | 1k → 16k | `-i bench_configs/bench_16k.json` |
| `deepseek-r1:8b` | 1k → 16k | `-i bench_configs/bench_16k.json` |

The 8B models are capped at 16k because a 32k KV cache exceeds the ~7.7 GB
memlock budget on this box (a 32k 8B run gets OOM-killed). `bench_16k.json` ships
in this repo (`max_length` 16384, `iterations` 8); regenerate it from the repo's
default config with:

```bash
cd FastFlowLM/hrx-integration
mkdir -p bench_configs
python3 -c "import json;d=json.load(open('../bench_config.json'));d['max_length']=16384;d['iterations']=8;json.dump(d,open('bench_configs/bench_16k.json','w'))"
```

### 6a. Pull the models

```bash
cd FastFlowLM/src
for m in qwen3:0.6b llama3.1:8b deepseek-r1:8b; do
  FLM_CONFIG_PATH="$PWD/model_list.json" build/flm pull "$m"
done
```

### 6b. Run the sweeps

The NPU is a **single shared device** — never run two sessions at once, and let
each `bench` finish. Each `bench` overwrites `bench_<model>_<date>.csv` in the
cwd, so run the XRT and HRX passes from **separate output directories**.

```bash
cd FastFlowLM/hrx-integration
mkdir -p bench_runs/xrt bench_runs/hrx
FLM="$PWD/../src/build/flm"
CFG="$PWD/bench_configs/bench_16k.json"

# --- XRT native baseline (no shim) ---
( cd bench_runs/xrt
  LD_LIBRARY_PATH=/opt/xilinx/xrt/lib FLM_CONFIG_PATH="$PWD/../../../src/model_list.json" \
    "$FLM" bench qwen3:0.6b                       # 1k–32k
  LD_LIBRARY_PATH=/opt/xilinx/xrt/lib FLM_CONFIG_PATH="$PWD/../../../src/model_list.json" \
    "$FLM" bench llama3.1:8b    -i "$CFG"         # 1k–16k
  LD_LIBRARY_PATH=/opt/xilinx/xrt/lib FLM_CONFIG_PATH="$PWD/../../../src/model_list.json" \
    "$FLM" bench deepseek-r1:8b -i "$CFG" )       # 1k–16k

# --- HRX via the shim (run.sh sets LD_LIBRARY_PATH/FLM_FORWARD/xclbins) ---
( cd bench_runs/hrx
  ../../scripts/run.sh bench qwen3:0.6b                    # 1k–32k
  ../../scripts/run.sh bench llama3.1:8b    -i "$CFG"      # 1k–16k
  ../../scripts/run.sh bench deepseek-r1:8b -i "$CFG" )    # 1k–16k
```

Before trusting a model's numbers, sanity-check correctness once (a `run` that
answers `Paris`, with `exe_fail=0` in the `[fwd] stats` line); DeepSeek-R1 is a
reasoning model and will *reason* its way to "Paris".

### 6c. Generate the report

Build per-model **Prefill** and **Decode** tables with columns
`Ctx | XRT | HRX | HRX%` (`HRX%` = HRX ÷ XRT, same machine). This script reads
the two CSV sets and writes `benchmark_log.md`:

```bash
cd FastFlowLM/hrx-integration
python3 - <<'PY'
import csv, glob, os
models = ["qwen3_0.6b", "llama3.1_8b", "deepseek-r1_8b"]
titles = {"qwen3_0.6b":"qwen3:0.6b","llama3.1_8b":"llama3.1:8b","deepseek-r1_8b":"deepseek-r1:8b"}
def load(d, m):
    hits = glob.glob(f"bench_runs/{d}/bench_{m}_*.csv")
    if not hits: return {}
    with open(sorted(hits)[-1]) as f:
        return {int(r["context_length_k"]): r for r in csv.DictReader(f)}
def table(metric, fmt, xrt, hrx):
    out = ["| Ctx | XRT | HRX | HRX% |", "|---|---|---|---|"]
    for k in sorted(hrx):
        h = float(hrx[k][metric])
        if k in xrt:
            x = float(xrt[k][metric]); out.append(f"| {k}k | {x:{fmt}} | {h:{fmt}} | {h/x*100:.0f}% |")
        else:
            out.append(f"| {k}k | N/A | {h:{fmt}} | N/A |")
    return "\n".join(out)
doc = ["# FastFlowLM on XDNA2 NPU — XRT vs HRX Benchmark", "",
       "Same machine, same session. `HRX%` = HRX ÷ native XRT.", ""]
for m in models:
    xrt, hrx = load("xrt", m), load("hrx", m)
    if not hrx: continue
    doc += [f"## {titles[m]}", "", "### Prefill (tok/s)", "",
            table("prefill_avg_toks_per_s", ".1f", xrt, hrx), "",
            "### Decode (tok/s)", "",
            table("decoding_avg_toks_per_s", ".2f", xrt, hrx), ""]
open("benchmark_log.md", "w").write("\n".join(doc) + "\n")
print("wrote benchmark_log.md")
PY
```

The result is a `benchmark_log.md` like the reference run below. (You may add an
**Environment** block — date, CPU/APU, NPU FW, FLM/HRX commit, XRT version, power
mode — at the top; query FW/XRT with `flm validate` on the native path.)

### 6d. Reference results

AMD Ryzen AI 9 HX 370 (Strix Point), NPU driver-default power mode (dynamic DPM)
for both paths, 8 iterations. HRX sits at XRT parity — tying/beating native on
small-context prefill and holding 95–99% elsewhere; the only sub-parity area is
0.6B decode at small context, where fixed per-step overhead dominates.

#### qwen3:0.6b (1k–32k)

Prefill (tok/s):

| Ctx | XRT | HRX | HRX% |
|---|---|---|---|
| 1k  | 1352.7 | 1292.1 | 96%  |
| 2k  | 1846.7 | 1779.9 | 96%  |
| 4k  | 1929.4 | 1954.1 | 101% |
| 8k  | 1872.0 | 1779.2 | 95%  |
| 16k | 1431.4 | 1396.2 | 98%  |
| 32k | 891.5  | 864.5  | 97%  |

Decode (tok/s):

| Ctx | XRT | HRX | HRX% |
|---|---|---|---|
| 1k  | 74.09 | 64.29 | 87% |
| 2k  | 61.43 | 52.76 | 86% |
| 4k  | 46.91 | 41.96 | 89% |
| 8k  | 31.90 | 28.44 | 89% |
| 16k | 19.69 | 18.21 | 92% |
| 32k | 11.27 | 10.91 | 97% |

#### llama3.1:8b (1k–16k)

Prefill (tok/s):

| Ctx | XRT | HRX | HRX% |
|---|---|---|---|
| 1k  | 330.8 | 350.9 | 106% |
| 2k  | 410.0 | 407.7 | 99%  |
| 4k  | 449.9 | 438.8 | 98%  |
| 8k  | 447.3 | 435.0 | 97%  |
| 16k | 382.9 | 374.9 | 98%  |

Decode (tok/s):

| Ctx | XRT | HRX | HRX% |
|---|---|---|---|
| 1k  | 10.95 | 10.51 | 96% |
| 2k  | 10.65 | 10.21 | 96% |
| 4k  | 10.09 | 9.72  | 96% |
| 8k  | 9.17  | 8.83  | 96% |
| 16k | 7.77  | 7.53  | 97% |

#### deepseek-r1:8b (1k–16k)

Prefill (tok/s):

| Ctx | XRT | HRX | HRX% |
|---|---|---|---|
| 1k  | 322.5 | 331.0 | 103% |
| 2k  | 404.8 | 399.3 | 99%  |
| 4k  | 447.2 | 435.4 | 97%  |
| 8k  | 445.6 | 435.8 | 98%  |
| 16k | 382.0 | 373.9 | 98%  |

Decode (tok/s):

| Ctx | XRT | HRX | HRX% |
|---|---|---|---|
| 1k  | 10.94 | 10.47 | 96% |
| 2k  | 10.65 | 10.23 | 96% |
| 4k  | 10.12 | 9.73  | 96% |
| 8k  | 9.18  | 8.90  | 97% |
| 16k | 7.77  | 7.54  | 97% |

### Command chaining (on by default)

Each FLM runlist is batched into one HRX `ERT_CMD_CHAIN` (`forward_runlist`)
instead of one synchronous dispatch per kernel, amortizing per-dispatch
submit/completion overhead (the numbers above). Set `FLM_CHAIN=0` to fall back
to one synchronous dispatch per kernel.

Requires the HRX `amdxdna` command-chain support in [ROCm/hrx-system#37].
`bench/` is a standalone, HRX-only microbenchmark of the chain path (chain vs
separate dispatch over one captured runlist).

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `flm: error while loading shared libraries: libhrx.so.0` | `run.sh` couldn't find libhrx — set `HRX_BUILD` to your HRX build dir. |
| `[fwd] … exe_fail=…` and garbage output | The shim read **empty xclbins**. FLM resolves them at `<flm-dir>/xclbins`; `run.sh` symlinks that to `../xclbins`. Confirm `src/xclbins/<model>/*.xclbin` exist. |
| All-zero / garbage tokens | Usually firmware <1.1, or the `xclbins` symlink missing, or a stale shim build. Rebuild the shim; re-check FW. |
| `INVALID_ARGUMENT … xclbin 0 is empty` (with `FLM_FORWARD_DEBUG=1`) | Same as above — xclbin bytes not found. |
| Killed during `bench` | The default sweep goes to 32k context (huge KV cache) → OOM on 8B models. Cap with `bench … -i bench_configs/bench_16k.json` (§6) or a smaller `max_length`. |
| Hang / NPU wedged | The NPU is a single shared device — never run two sessions at once, and avoid SIGKILL mid-dispatch. |

Debug env: `FLM_FORWARD_DEBUG=1` logs executable build/load failures and xclbin
byte counts; `FLM_FORWARD=0` disables forwarding entirely.

### Graph dump

Set `FLM_GRAPH_DUMP=/path/to/flm_graph.jsonl` to write a default-off JSONL
trace of the dynamic dispatch graph observed by the interposer. Each record is a
single JSON object. The dump includes runlist nodes, dispatch nodes, xclbin and
control-code hashes, bound BO ids/sizes, and host boundary events such as
`sync_to`, `sync_from`, `map`, `h2d_upload`, and `readback`.

Use this for offline analysis of repeated dispatch shapes and host/device
boundaries. It is diagnostic infrastructure, not a benchmark mode; graph records
are flushed as they are written so the file survives interrupted runs.

## 8. Limitations

- Runlists are batched into HRX command chains (§6, on by default); single
  `xrt::run::start` dispatches still take the per-dispatch synchronous path.
- Verified end-to-end (correctness + benchmarks, §6) with `qwen3:0.6b`,
  `llama3.1:8b`, and `deepseek-r1:8b`; other models share the kernel ABI and
  should work but are unverified here.
- The shim assumes FLM's `arg3` is the kernel output (true for the Qwen3
  kernels). The `xrt::uuid` ABI is opaque, so xclbin↔context association uses
  construction order (the engine builds each xclbin immediately before its
  `hw_context`).

[ROCm/hrx-system#37]: https://github.com/ROCm/hrx-system/pull/37
