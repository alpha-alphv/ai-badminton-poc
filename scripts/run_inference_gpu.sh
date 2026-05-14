#!/usr/bin/env bash
# Launch the FastAPI inference service on the GPU host.
# Run this on Grafilab after scripts/grafilab_setup.sh has completed.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="$REPO_DIR"
export DATA_DIR="${DATA_DIR:-$REPO_DIR/data}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$REPO_DIR/data/yolo_cache}"
export INFERENCE_DEVICE="${INFERENCE_DEVICE:-cuda}"
export INFERENCE_MODEL="${INFERENCE_MODEL:-yolov8s-pose.pt}"
export INFERENCE_MAX_FRAMES="${INFERENCE_MAX_FRAMES:-1000}"
export INFERENCE_CONF_THRESHOLD="${INFERENCE_CONF_THRESHOLD:-0.4}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

mkdir -p "$DATA_DIR/uploads" "$DATA_DIR/artifacts" "$DATA_DIR/db" "$DATA_DIR/yolo_cache"

echo "==> Launch config"
echo "    repo=$REPO_DIR"
echo "    device=$INFERENCE_DEVICE  model=$INFERENCE_MODEL  max_frames=$INFERENCE_MAX_FRAMES"
echo "    bind=$HOST:$PORT"

echo "==> GPU sanity check"
python - <<'PY'
import os, sys
try:
    import torch
    avail = torch.cuda.is_available()
    print(f"    torch={torch.__version__}  cuda_available={avail}  devices={torch.cuda.device_count()}")
    if avail:
        for i in range(torch.cuda.device_count()):
            print(f"    cuda:{i} -> {torch.cuda.get_device_name(i)}")
    req = os.environ.get("INFERENCE_DEVICE", "")
    if req.startswith("cuda") and not avail:
        print("    !! INFERENCE_DEVICE=cuda but no CUDA device detected — falling back will fail.", file=sys.stderr)
        sys.exit(2)
except Exception as e:
    print(f"    torch import failed: {e}", file=sys.stderr)
    sys.exit(2)
PY

echo "==> Starting uvicorn"
exec uvicorn services.inference_api.app.main:app --host "$HOST" --port "$PORT"
