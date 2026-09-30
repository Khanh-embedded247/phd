#!/usr/bin/env bash
set -euo pipefail

# Chay MapTR tren doan Normal khong dung lam train chinh / khong can GT.
# Muc dich: xem inference thuc te va tao prediction vector map bang mat.

SURVEY_ROOT="/home/khanh247/Documents/Survey"
MAPTR_ROOT="${SURVEY_ROOT}/MapTR"
PHENIKAA_ROOT="${SURVEY_ROOT}/phd/phenikaa"
CONDA_PY="/home/khanh247/miniconda3/envs/maptr/bin/python"

CONFIG="${PHENIKAA_ROOT}/benchmark_maptr/config/maptr_tiny_r50_phenikaa_12cam_train_osm.py"
CHECKPOINT="${PHENIKAA_ROOT}/outputs/benchmark_maptr/Normal/work_dirs/maptr_12cam_osm/latest.pth"
INFOS="${PHENIKAA_ROOT}/outputs/benchmark_maptr/Normal/phenikaa_maptr_infos_val.pkl"
OUT_DIR="${PHENIKAA_ROOT}/outputs/benchmark_maptr/Normal/inference_unlabeled_val"

# Chinh truc tiep cac gia tri nay de chay doan khac cua infos val.
NUM_SAMPLES="${NUM_SAMPLES:-80}"
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
  --infos "${INFOS}" \
  --out-dir "${OUT_DIR}" \
  --num-samples "${NUM_SAMPLES}" \
  --start-index "${START_INDEX}" \
  --score-thresh "${SCORE_THRESH}" \
  --image-scale "${IMAGE_SCALE}" \
  --no-gt

"${CONDA_PY}" "${PHENIKAA_ROOT}/benchmark_maptr/15_export_prediction_osm.py" \
  --pred-dir "${OUT_DIR}" \
  --infos "${INFOS}" \
  --out-osm "${PHENIKAA_ROOT}/outputs/benchmark_maptr/Normal/predicted_vector_map_unlabeled_val.osm" \
  --score-thresh "${SCORE_THRESH}"

"${CONDA_PY}" "${PHENIKAA_ROOT}/benchmark_maptr/16_visualize_maptr_paper_style.py" \
  --pred-dir "${OUT_DIR}" \
  --infos "${INFOS}" \
  --out-dir "${PHENIKAA_ROOT}/outputs/benchmark_maptr/Normal/paper_style_unlabeled_val" \
  --max-samples 8 \
  --score-thresh "${SCORE_THRESH}"

