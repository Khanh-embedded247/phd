#!/usr/bin/env bash
set -euo pipefail

# Chay inference tren nhieu doan sample khac nhau roi tinh metric tung doan.
# Muc dich: khong chi xem 20 frame dau, ma kiem tra nhieu vi tri trong dataset/OSM GT.

SURVEY_ROOT="/home/khanh247/Documents/Survey"
MAPTR_ROOT="${SURVEY_ROOT}/MapTR"
PHENIKAA_ROOT="${SURVEY_ROOT}/phd/phenikaa"
CONDA_PY="/home/khanh247/miniconda3/envs/maptr/bin/python"

CONFIG="${PHENIKAA_ROOT}/benchmark_maptr/config/maptr_tiny_r50_phenikaa_12cam_train_osm.py"
CHECKPOINT="${PHENIKAA_ROOT}/outputs/benchmark_maptr/Normal/work_dirs/maptr_12cam_osm/latest.pth"
BASE_OUT_DIR="${PHENIKAA_ROOT}/outputs/benchmark_maptr/Normal/inference_scan"

# Sua truc tiep cac gia tri nay neu muon quet nhieu/it hon.
START_INDICES=(0 50 100 150)
NUM_SAMPLES="${NUM_SAMPLES:-20}"
SCORE_THRESH="${SCORE_THRESH:-0.30}"
IMAGE_SCALE="${IMAGE_SCALE:-0.25}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
export PYTHONPATH="${MAPTR_ROOT}/mmdetection3d:${MAPTR_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12.8/lib64:/home/khanh247/miniconda3/envs/maptr/lib:${LD_LIBRARY_PATH:-}"

cd "${MAPTR_ROOT}"

for START_INDEX in "${START_INDICES[@]}"; do
  OUT_DIR="${BASE_OUT_DIR}/start_${START_INDEX}_n_${NUM_SAMPLES}"
  echo "============================================================"
  echo "[RUN] start_index=${START_INDEX} num_samples=${NUM_SAMPLES}"
  echo "[OUT] ${OUT_DIR}"

  "${CONDA_PY}" "${PHENIKAA_ROOT}/benchmark_maptr/11_infer_visualize_maptr_phenikaa.py" \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --out-dir "${OUT_DIR}" \
    --num-samples "${NUM_SAMPLES}" \
    --start-index "${START_INDEX}" \
    --score-thresh "${SCORE_THRESH}" \
    --image-scale "${IMAGE_SCALE}"

  "${CONDA_PY}" "${PHENIKAA_ROOT}/benchmark_maptr/13_evaluate_maptr_predictions.py" \
    --in-dir "${OUT_DIR}"
done

echo "============================================================"
echo "[DONE] Tat ca ket qua nam trong: ${BASE_OUT_DIR}"

