#!/usr/bin/env bash
# Build the standalone HRX-only runlist microbenchmark (flm_hrx_runlist_bench).
#
# Pure HRX: links libhrx.so only (no shim, no FastFlowLM runtime). Replays one
# captured FLM runlist as an ERT_CMD_CHAIN vs separate dispatches. Self-contained
# within this branch + an HRX checkout/build.
#
# Env (auto-detected if unset):
#   HRX_DIR     hrx checkout (for headers)
#   HRX_BUILD   hrx build dir (for libhrx.so + generated headers)
#   OUT         output binary path (default ./build/flm_hrx_runlist_bench)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../src"   # hrx_xadx_builder.hpp lives next to the shim

HRX_DIR="${HRX_DIR:-}"
if [[ -z "$HRX_DIR" ]]; then
  for c in "$HOME/hrx" "$HOME/workspace/hrx" "$HERE/../../../hrx"; do
    [[ -d "$c/libhrx/include" ]] && HRX_DIR="$(cd "$c" && pwd)" && break
  done
fi
[[ -n "$HRX_DIR" && -d "$HRX_DIR/libhrx/include" ]] || { echo "ERROR: set HRX_DIR to your hrx checkout"; exit 1; }

HRX_BUILD="${HRX_BUILD:-}"
if [[ -z "$HRX_BUILD" ]]; then
  for c in "$HRX_DIR/build-amdxdna-native" "$HRX_DIR/build"; do
    [[ -f "$c/libhrx/src/libhrx/libhrx.so" ]] && HRX_BUILD="$c" && break
  done
fi
[[ -n "$HRX_BUILD" && -f "$HRX_BUILD/libhrx/src/libhrx/libhrx.so" ]] || {
  echo "ERROR: set HRX_BUILD to your hrx build dir (with libhrx/src/libhrx/libhrx.so)"; exit 1; }

OUT="${OUT:-$HERE/build/flm_hrx_runlist_bench}"
mkdir -p "$(dirname "$OUT")"

set -x
g++ -std=c++20 -O2 -o "$OUT" \
  "$HERE/flm_hrx_runlist_bench.cpp" \
  -I "$HRX_DIR/libhrx/include" \
  -I "$SRC" \
  -I "$HRX_DIR/runtime/src" \
  -I "$HRX_BUILD/runtime/src" \
  -I "$HRX_BUILD/_deps/flatcc-src/include" \
  "$HRX_BUILD/libhrx/src/libhrx/libhrx.so" \
  "$HRX_BUILD/libflatcc_runtime.a" \
  -Wl,-rpath,"$HRX_BUILD/libhrx/src/libhrx"
set +x
echo "built: $OUT"
