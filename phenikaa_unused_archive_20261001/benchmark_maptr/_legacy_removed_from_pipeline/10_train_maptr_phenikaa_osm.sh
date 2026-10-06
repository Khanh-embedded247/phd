#!/usr/bin/env bash
set -euo pipefail

# Train MapTR 12 camera tren GT OSM Phenikaa.
#
# Yeu cau:
#   - chay trong conda env maptr
#   - NVIDIA driver/CUDA phai hoat dong: python -c "import torch; print(torch.cuda.is_available())" -> True
#
# Chay:
#   cd /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/vendor/MapTR
#   bash /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/10_train_maptr_phenikaa_osm.sh

PHENIKAA_ROOT="/home/khanh247/Documents/Survey/phd/phenikaa"
MAPTR_ROOT="${PHENIKAA_ROOT}/benchmark_maptr/vendor/MapTR"
CONFIG="/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/config/maptr_tiny_r50_phenikaa_12cam_train_osm.py"
PYTHON="/home/khanh247/miniconda3/envs/maptr/bin/python"

export PYTHONPATH="${MAPTR_ROOT}/mmdetection3d:${MAPTR_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/phenikaa_maptr_matplotlib"
export XDG_CACHE_HOME="/tmp/phenikaa_maptr_matplotlib"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"

# Runtime CUDA:
# - PyTorch 1.9.1+cu111 da kem CUDA runtime 11.1 trong wheel/conda.
# - Driver 570/CUDA 12.8 cua he thong van chay duoc binary cu111.
# - Khong nen symlink CUDA 12.8 de thay CUDA 11.1 trong env, de tranh lech ABI
#   voi mmcv/mmdet3d ops da build cho torch cu111.
# - Chi them duong dan libcuda cua driver he thong de torch thay GPU.
export CUDA_HOME="${CUDA_HOME:-/home/khanh247/miniconda3}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/home/khanh247/miniconda3/lib:/home/khanh247/miniconda3/envs/maptr/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${MPLCONFIGDIR}"

cd "${MAPTR_ROOT}"
"${PYTHON}" - <<'PY'
import torch
print("torch:", torch.__version__, "compiled cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(), "device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch trong env maptr chua thay GPU. Kiem tra driver/libcuda trong terminal nay.")
print("device 0:", torch.cuda.get_device_name(0))
PY
"${PYTHON}" tools/train.py "${CONFIG}" --no-validate --gpus 1
