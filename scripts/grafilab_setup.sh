#!/usr/bin/env bash
# One-time setup on the Grafilab GPU host.
# Auto-detects the repo root from the script's own location, so it just
# works whether the repo lives in $HOME, /var/www, or anywhere else.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DEFAULT_REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="${REPO_DIR:-$DEFAULT_REPO_DIR}"

CUDA_CHANNEL="${CUDA_CHANNEL:-cu121}"   # cu118 / cu121 / cu124 — match the host
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.0}"

if [ ! -d "$REPO_DIR/services/inference_api" ]; then
  echo "REPO_DIR=$REPO_DIR doesn't look like the ai-badminton-poc repo." >&2
  echo "Either run this script from within the cloned repo, or pass REPO_DIR=<path>." >&2
  exit 1
fi

cd "$REPO_DIR"
echo "==> Repo: $REPO_DIR"

# sudo is unavailable in many GPU containers; skip it when we're already root.
if [ "$(id -u)" = "0" ]; then
  SUDO=""
else
  SUDO="sudo"
fi

echo "==> System packages"
if command -v apt-get >/dev/null 2>&1; then
  $SUDO apt-get update
  $SUDO apt-get install -y --no-install-recommends \
    python3-venv python3-pip libgl1 libglib2.0-0 ffmpeg
else
  echo "    apt-get not found — assuming base image already provides python/ffmpeg/libgl1."
fi

echo "==> Python venv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

echo "==> CUDA torch (${CUDA_CHANNEL})"
pip install --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}" \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

echo "==> Inference deps"
pip install -r services/inference_api/requirements.txt

echo "==> Data dirs"
mkdir -p data/uploads data/artifacts data/db data/yolo_cache

echo "==> Pre-download YOLOv8-pose weights"
YOLO_CONFIG_DIR="$PWD/data/yolo_cache" python -c \
  "from ultralytics import YOLO; YOLO('yolov8s-pose.pt')"

echo "==> CUDA check"
python - <<'PY'
import torch
print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}  "
      f"devices={torch.cuda.device_count()}  "
      f"name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
PY

cat <<EOF

Setup complete.

To start the inference service on this host:

  cd $REPO_DIR
  source .venv/bin/activate
  export PYTHONPATH=\$PWD
  export DATA_DIR=\$PWD/data
  export YOLO_CONFIG_DIR=\$PWD/data/yolo_cache
  export INFERENCE_DEVICE=cuda
  export INFERENCE_MODEL=yolov8s-pose.pt
  uvicorn services.inference_api.app.main:app --host 127.0.0.1 --port 8000

On your local machine, open the SSH tunnel:

  # PowerShell
  .\\scripts\\tunnel.ps1 -User root -RemoteHost 118.107.222.200 -Port 39225

Then start the local stack:

  docker compose up --build

EOF
