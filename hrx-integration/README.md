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
**https://github.com/jtuyls/FastFlowLM/tree/hrx-integration** and pairs with the
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
   executable (`src/hrx_xadx_builder.hpp`), cached by control-code hash.
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

### 3b. HRX (with the amdxdna HAL) — ROCm/hrx-system PR #37

```bash
git clone https://github.com/ROCm/hrx-system.git hrx
cd hrx
gh pr checkout 37          # or: git fetch origin pull/37/head:amdxdna && git checkout amdxdna
python dev.py cmake setup
python dev.py cmake configure -DIREE_HAL_DRIVER_AMDXDNA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
python dev.py cmake build
```

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
git checkout hrx-integration
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
./scripts/run.sh bench qwen3:0.6b -c 2048    # benchmark (cap context to avoid 32k OOM)
```

`run.sh` auto-detects the `flm` binary (`../src/build/flm`), `libhrx.so`, the
model list, and the xclbins; it symlinks `<flm-dir>/xclbins -> ../xclbins`, sets
`LD_LIBRARY_PATH` (shim first, then libhrx, then `/opt/xilinx/xrt/lib`), and
`FLM_FORWARD=1`. Override any with `FLM_BIN`, `HRX_BUILD`, `FLM_CONFIG_PATH`,
`FLM_FORWARD=0`.

**Expected:** coherent output (e.g. `2 + 2 = 4.`). The shim prints periodic
`[fwd] stats: dispatched=… h2d_copies=… h2d_skips=…` to stderr.

---

## 6. Measured performance

AMD Ryzen AI 9 365 (Strix Point), Qwen3-0.6B, via this interposer → HRX (`flm
bench`, command chaining on), vs FastFlowLM's published **native XRT** figures
(measured on Ryzen AI 7 350 / Kraken Point — different hardware, so indicative
only):

| Context | Decode (HRX) | Decode (native adv.) | Prefill (HRX) | Prefill (native adv.) |
|--------:|-------------:|---------------------:|--------------:|----------------------:|
| 1k      | 45.1 tok/s   | 66.5                 | 752 tok/s     | 1494                  |
| 4k      | 33.0         | 44.5                 | 1406          | 2165                  |
| 16k     | 16.8         | 19.6                 | 1170          | 1485                  |
| 32k     | 10.0         | 14.1                 | 768           | 907                   |

The native columns were measured on **different hardware** (Kraken Point), and
**native XRT cannot run on this machine at all** (no XRT XDNA driver plugin is
installed — only emulation shims), which is the whole reason this HRX interposer
exists. So the absolute gap is *not* directly attributable from here.

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
| Killed during `bench` | The full bench sweeps to 32k context (huge KV cache) over many iterations → OOM/timeout. Use `bench … -c 2048` or a shorter run. |
| Hang / NPU wedged | The NPU is a single shared device — never run two sessions at once, and avoid SIGKILL mid-dispatch. |

Debug env: `FLM_FORWARD_DEBUG=1` logs executable build/load failures and xclbin
byte counts; `FLM_FORWARD=0` disables forwarding entirely.

## 8. Limitations

- Runlists are batched into HRX command chains (§6, on by default); single
  `xrt::run::start` dispatches still take the per-dispatch synchronous path.
- Verified with `qwen3:0.6b`; other models share the kernel ABI and should work
  but are unverified here.
- The shim assumes FLM's `arg3` is the kernel output (true for the Qwen3
  kernels). The `xrt::uuid` ABI is opaque, so xclbin↔context association uses
  construction order (the engine builds each xclbin immediately before its
  `hw_context`).

[ROCm/hrx-system#37]: https://github.com/ROCm/hrx-system/pull/37
