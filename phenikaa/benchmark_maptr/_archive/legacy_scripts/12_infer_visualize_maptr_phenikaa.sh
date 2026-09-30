#!/usr/bin/env bash
set -euo pipefail

# Chạy inference MapTR Phenikaa 12 camera bằng checkpoint đã train.
# Đầu ra gồm ảnh BEV GT/dự đoán/overlay và file prediction json/pkl để kiểm tra bằng mắt.

SURVEY_ROOT="/home/khanh247/Documents/Survey"
MAPTR_ROOT="${SURVEY_ROOT}/MapTR"
PHENIKAA_ROOT="${SURVEY_ROOT}/phd/phenikaa"
CONDA_PY="/home/khanh247/miniconda3/envs/maptr/bin/python"

CONFIG="${PHENIKAA_ROOT}/benchmark_maptr/config/maptr_tiny_r50_phenikaa_12cam_train_osm.py"
CHECKPOINT="${PHENIKAA_ROOT}/outputs/benchmark_maptr/Normal/work_dirs/maptr_12cam_osm/latest.pth"
OUT_DIR="${PHENIKAA_ROOT}/outputs/benchmark_maptr/Normal/inference_vis"

# Cac tham so hay can doi khi kiem tra benchmark.
NUM_SAMPLES="${NUM_SAMPLES:-20}"
START_INDEX="${START_INDEX:-0}"
SCORE_THRESH="${SCORE_THRESH:-0.30}"
IMAGE_SCALE="${IMAGE_SCALE:-0.25}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
export PYTHONPATH="${MAPTR_ROOT}/mmdetection3d:${MAPTR_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12.8/lib64:/home/khanh247/miniconda3/envs/maptr/lib:${LD_LIBRARY_PATH:-}"

cd "${MAPTR_ROOT}"

"${CONDA_PY}" "${PHENIKAA_ROOT}/benchmark_maptr/11_infer_visualize_maptr_phenikaa.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --out-dir "${OUT_DIR}" \
  --num-samples "${NUM_SAMPLES}" \
  --start-index "${START_INDEX}" \
  --score-thresh "${SCORE_THRESH}" \
  --image-scale "${IMAGE_SCALE}"
