# HRX-only runlist microbenchmark

`flm_hrx_runlist_bench.cpp` replays **one real FastFlowLM runlist** through pure
HRX (links `libhrx.so` only — no shim, no FLM runtime) two ways and reports the
ratio:

- **SEPARATE**: per run, `hrx_stream_dispatch` + `hrx_stream_synchronize` (N
  individual `START_NPU` submits)
- **CHAIN**: N × `hrx_stream_dispatch` (no sync) + one `hrx_stream_synchronize`
  (one `ERT_CMD_CHAIN`)

It isolates HRX's command-chain path from FLM entirely and runs in ~30s with no
model load — a fast, safe check for the chain path. Self-contained within this
branch plus an HRX checkout/build (ROCm/hrx-system#37).

## 1. Capture a runlist (binaries — not committed)

The bench needs a real runlist: an `xclbin.bin`, one control ELF per run
(`run_NN.elf`), and a `manifest.txt` of binding sizes/ids. These are large
binary artifacts and are **not** committed; regenerate them locally with the
shim's env-gated capture:

```bash
# from the hrx-integration root:
FLM_DUMP_RUNLIST=20 ./scripts/run.sh run qwen3:0.6b   # then type a prompt + /bye
# -> /tmp/flm_runlist/{xclbin.bin, run_00.elf ... run_NN.elf, manifest.txt}
```
`FLM_DUMP_RUNLIST=<n>` dumps the first runlist with ≥ n runs (20 captures a full
~28-kernel transformer-layer runlist). Override the output dir with
`FLM_DUMP_RUNLIST_DIR=/path`.

## 2. Build

```bash
# auto-detects HRX_DIR / HRX_BUILD; override either if needed.
./build.sh                       # -> build/flm_hrx_runlist_bench
```

## 3. Run

```bash
# Ensure the NPU is idle (single shared device).
build/flm_hrx_runlist_bench /tmp/flm_runlist
```

Expected on npu4 for a ~28-kernel runlist:

```
SEPARATE: ~390 us/dispatch
CHAIN   : ~315 us/dispatch
ratio chain/separate = ~0.8x  (chain FASTER)
```

The buffers are allocated to the captured sizes but not filled, so the bench
measures dispatch/submit/chain timing, not numerical correctness (a ~28-layer
runlist is ~4 GB of buffers; ≥ ~8 GB free RAM recommended).
