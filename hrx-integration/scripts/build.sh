#!/usr/bin/env bash
# Build the XRT->HRX interposer shim (libxrt_coreutil.so.2) for FastFlowLM.
#
# The shim provides the xrt:: symbols FLM's closed NPU engines import, but
# forwards every dispatch to HRX's amdxdna backend instead of XRT. See README.
#
# Required: an HRX checkout built with the amdxdna HAL enabled.
#   HRX_DIR        path to the hrx source checkout
#   HRX_BUILD      path to the hrx build dir (e.g. <hrx>/build-amdxdna-native)
# Both can be passed as env vars or auto-detected from common locations.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../src"
OUT="$HERE/../build"
mkdir -p "$OUT"

# --- locate HRX ---
HRX_DIR="${HRX_DIR:-}"
HRX_BUILD="${HRX_BUILD:-}"
if [[ -z "$HRX_DIR" ]]; then
  for c in "$HOME/hrx" "../hrx" "../../hrx" "../../../hrx"; do
    [[ -f "$c/libhrx/include/hrx_runtime.h" ]] && HRX_DIR="$(cd "$c" && pwd)" && break
  done
fi
[[ -n "$HRX_DIR" && -f "$HRX_DIR/libhrx/include/hrx_runtime.h" ]] || {
  echo "ERROR: set HRX_DIR to your hrx checkout (with libhrx/include/hrx_runtime.h)"; exit 1; }
if [[ -z "$HRX_BUILD" ]]; then
  for c in "$HRX_DIR/build-amdxdna-native" "$HRX_DIR/build"; do
    [[ -f "$c/libhrx/src/libhrx/libhrx.so" ]] && HRX_BUILD="$c" && break
  done
fi
[[ -n "$HRX_BUILD" && -f "$HRX_BUILD/libhrx/src/libhrx/libhrx.so" ]] || {
  echo "ERROR: set HRX_BUILD to your hrx build dir (with libhrx/src/libhrx/libhrx.so)"; exit 1; }

LIBHRX_DIR="$HRX_BUILD/libhrx/src/libhrx"
FLATCC_INC="$HRX_BUILD/_deps/flatcc-src/include"
FLATCC_LIB="$HRX_BUILD/libflatcc_runtime.a"

echo "HRX_DIR   = $HRX_DIR"
echo "HRX_BUILD = $HRX_BUILD"

g++ -std=c++20 -O2 -fPIC -shared -o "$OUT/libxrt_coreutil.so.2" \
  "$SRC/xrt_coreutil_shim.cpp" \
  -I "$SRC" \
  -I "$HRX_DIR/libhrx/include" \
  -I "$HRX_DIR/runtime/src" \
  -I "$HRX_BUILD/runtime/src" \
  -I "$FLATCC_INC" \
  "$LIBHRX_DIR/libhrx.so" \
  "$FLATCC_LIB" \
  -Wl,-rpath,"$LIBHRX_DIR"

echo "built: $OUT/libxrt_coreutil.so.2"
echo "(libhrx at $LIBHRX_DIR)"
