#!/usr/bin/env bash
# Run FastFlowLM on the NPU through HRX (via the interposer shim).
#
# Usage:   ./scripts/run.sh <flm args...>
# Example: ./scripts/run.sh run qwen3:0.6b
#          ./scripts/run.sh bench qwen3:0.6b
#
# Env (auto-detected if unset):
#   FLM_BIN     path to the built `flm` binary
#   HRX_BUILD   hrx build dir (for libhrx.so)
#   FLM_FORWARD set to 0 to disable forwarding (capture/no-op mode)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SHIM="$HERE/../build/libxrt_coreutil.so.2"
[[ -f "$SHIM" ]] || { echo "ERROR: shim not built. Run scripts/build.sh first."; exit 1; }

# --- locate flm binary ---
FLM_BIN="${FLM_BIN:-}"
if [[ -z "$FLM_BIN" ]]; then
  for c in "$HERE/../../src/build/flm" "$HERE/../../build/flm" "$(command -v flm || true)"; do
    [[ -n "$c" && -x "$c" ]] && FLM_BIN="$c" && break
  done
fi
[[ -n "$FLM_BIN" && -x "$FLM_BIN" ]] || { echo "ERROR: set FLM_BIN to your built flm binary"; exit 1; }
FLM_BUILD_DIR="$(cd "$(dirname "$FLM_BIN")" && pwd)"

# --- locate libhrx.so ---
HRX_BUILD="${HRX_BUILD:-}"
if [[ -z "$HRX_BUILD" ]]; then
  for c in "$HOME/workspace/iree-ai/hrx/build-amdxdna-native" "$HOME/hrx/build-amdxdna-native"; do
    [[ -f "$c/libhrx/src/libhrx/libhrx.so" ]] && HRX_BUILD="$c" && break
  done
fi
[[ -n "$HRX_BUILD" && -f "$HRX_BUILD/libhrx/src/libhrx/libhrx.so" ]] || {
  echo "ERROR: set HRX_BUILD to your hrx build dir (with libhrx/src/libhrx/libhrx.so)"; exit 1; }
LIBHRX_DIR="$HRX_BUILD/libhrx/src/libhrx"

# --- xclbins: FLM resolves them exe-relative to <flm dir>/xclbins; the real
#     files live one level up under src/xclbins. Symlink if missing. ---
if [[ ! -e "$FLM_BUILD_DIR/xclbins" && -d "$FLM_BUILD_DIR/../xclbins" ]]; then
  ln -sfn ../xclbins "$FLM_BUILD_DIR/xclbins"
  echo "[run] linked $FLM_BUILD_DIR/xclbins -> ../xclbins"
fi

# --- FLM model list (FLM_CONFIG_PATH): needed when running from a build tree ---
if [[ -z "${FLM_CONFIG_PATH:-}" ]]; then
  for c in "$FLM_BUILD_DIR/model_list.json" "$HERE/../../src/model_list.json"; do
    [[ -f "$c" ]] && export FLM_CONFIG_PATH="$c" && break
  done
fi

SHIM_DIR="$(cd "$(dirname "$SHIM")" && pwd)"
export LD_LIBRARY_PATH="$SHIM_DIR:$LIBHRX_DIR:/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}"
export FLM_FORWARD="${FLM_FORWARD:-1}"

echo "[run] shim=$SHIM_DIR  libhrx=$LIBHRX_DIR  FLM_FORWARD=$FLM_FORWARD"
echo "[run] $FLM_BIN $*"
exec "$FLM_BIN" "$@"
