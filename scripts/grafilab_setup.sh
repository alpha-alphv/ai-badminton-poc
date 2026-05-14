#!/usr/bin/env bash
# One-time setup on the Grafilab GPU host.
# Run AFTER cloning the repo to ~/badminton-ai-poc (or wherever; set REPO_DIR).
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/ai-badminton-poc}"
CUDA_CHANNEL="${CUDA_CHANNEL:-cu121}"   # cu118 / cu121 / cu124 — match the host
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.0}"

if [ ! -d "$REPO_DIR" ]; then
  echo "Repo not found at $REPO_DIR. Clone it first or set REPO_DIR." >&2
  exit 1
fi

cd "$REPO_DIR"

echo "==> System packages (sudo)"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  python3-venv python3-pip libgl1 libglib2.0-0 ffmpeg

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
