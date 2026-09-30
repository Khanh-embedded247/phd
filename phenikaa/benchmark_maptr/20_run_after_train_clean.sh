#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SCRIPT CHINH SAU BUOC 10 TRAIN
# ============================================================
# Chi can doi MODE:
#
# MODE=no_metric
#   - Chay thuc te tren data khong can GT.
#   - Output: predictions.json, predicted_vector_map.osm, anh visualize.
#
# MODE=metric
#   - Chay tren data co GT OSM de tinh metric.
#   - Output them: evaluation_metrics.csv, evaluation_metrics.summary.json.
#
# Luu y:
#   File .pkl o day KHONG phai GT. No la manifest: danh sach frame,
#   duong dan 12 camera, calib, pose. Train hay inference deu can manifest.
# ============================================================

SURVEY_ROOT="/home/khanh247/Documents/Survey"
PHENIKAA_ROOT="${SURVEY_ROOT}/phd/phenikaa"
MAPTR_ROOT="${PHENIKAA_ROOT}/benchmark_maptr/vendor/MapTR"
CONDA_PY="/home/khanh247/miniconda3/envs/maptr/bin/python"

SCENARIO="${SCENARIO:-Normal}"
CHECKPOINT_SCENARIO="${CHECKPOINT_SCENARIO:-Normal}"
MODE="${MODE:-no_metric}"  # no_metric hoac metric

CONFIG="${PHENIKAA_ROOT}/benchmark_maptr/config/maptr_tiny_r50_phenikaa_12cam_train_osm.py"
CHECKPOINT="${CHECKPOINT:-${PHENIKAA_ROOT}/outputs/benchmark_maptr/${CHECKPOINT_SCENARIO}/work_dirs/maptr_12cam_osm/latest.pth}"

# MODE=no_metric mac dinh chay full Normal.
FULL_INFOS="${PHENIKAA_ROOT}/outputs/benchmark_maptr/${SCENARIO}/phenikaa_maptr_infos_full_normal.pkl"

# MODE=metric mac dinh chay tren infos train co GT. Neu co test GT rieng,
# hay doi METRIC_INFOS sang file test do.
METRIC_INFOS="${PHENIKAA_ROOT}/outputs/benchmark_maptr/${SCENARIO}/phenikaa_maptr_infos_train_osm.pkl"

START_INDEX="${START_INDEX:-0}"
# Mac dinh chay 200 frame de test nhanh. Dat NUM_SAMPLES=0 neu muon chay FULL,
# vi full Normal hien co khoang 8704 frame va co the mat vai gio tren RTX 4060.
NUM_SAMPLES="${NUM_SAMPLES:-200}"      # 0 = chay het infos
SCORE_THRESH="${SCORE_THRESH:-0.30}"
IMAGE_SCALE="${IMAGE_SCALE:-0.25}"
VIS_MAX_SAMPLES="${VIS_MAX_SAMPLES:-20}"
# Dat REBUILD_INFOS=1 sau khi sua undistort/calib/sync de tao lai .pkl va
# ghi de anh trong images_undistorted. Mac dinh khong rebuild de chay nhanh.
REBUILD_INFOS="${REBUILD_INFOS:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
export PYTHONPATH="${MAPTR_ROOT}/mmdetection3d:${MAPTR_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12.8/lib64:/home/khanh247/miniconda3/envs/maptr/lib:${LD_LIBRARY_PATH:-}"

if [[ "${MODE}" != "no_metric" && "${MODE}" != "metric" ]]; then
  echo "MODE phai la no_metric hoac metric, hien tai: ${MODE}" >&2
  exit 1
fi

if [[ "${MODE}" == "no_metric" ]]; then
  INFOS="${FULL_INFOS}"
  if [[ "${NUM_SAMPLES}" == "0" ]]; then
    FINAL_DIR="${PHENIKAA_ROOT}/outputs/benchmark_maptr/${SCENARIO}/final_run/no_metric_full_normal"
  else
    FINAL_DIR="${PHENIKAA_ROOT}/outputs/benchmark_maptr/${SCENARIO}/final_run/no_metric_start_${START_INDEX}_n_${NUM_SAMPLES}"
  fi

  if [[ "${REBUILD_INFOS}" == "1" || ! -f "${INFOS}" ]]; then
    echo "[BUILD] Tao full infos cho ${SCENARIO}: ${INFOS}"
    "${CONDA_PY}" "${PHENIKAA_ROOT}/benchmark_maptr/03_build_phenikaa_infos.py" \
      --scenario "${SCENARIO}" \
      --split full \
      --max-samples 0 \
      --out-pkl "${INFOS}"
  fi
  NO_GT_ARG="--no-gt"
else
  INFOS="${METRIC_INFOS}"
  FINAL_DIR="${PHENIKAA_ROOT}/outputs/benchmark_maptr/${SCENARIO}/final_run/metric_with_gt"
  NO_GT_ARG=""
  if [[ "${REBUILD_INFOS}" == "1" || ! -f "${INFOS}" ]]; then
    echo "[BUILD] Tao metric/train infos co GT cho ${SCENARIO}: ${INFOS}"
    "${CONDA_PY}" "${PHENIKAA_ROOT}/benchmark_maptr/03_build_phenikaa_infos.py" \
      --scenario "${SCENARIO}" \
      --split train_osm \
      --max-samples 0 \
      --out-pkl "${INFOS}"
  fi
fi

INTERMEDIATE_DIR="${FINAL_DIR}/_intermediate"
VIS_MAP_DIR="${FINAL_DIR}/bev_map"
VIS_CAMERA_DIR="${FINAL_DIR}/camera_context"
OSM_THIN_OUT="${FINAL_DIR}/predicted_vector_map_thin.osm"
OSM_GRAPH_OUT="${FINAL_DIR}/predicted_vector_map_graph.osm"

mkdir -p "${FINAL_DIR}" "${INTERMEDIATE_DIR}" "${VIS_MAP_DIR}" "${VIS_CAMERA_DIR}"

echo "============================================================"
echo "MODE       : ${MODE}"
echo "INFOS      : ${INFOS}"
echo "CHECKPOINT : ${CHECKPOINT}"
echo "CHECKPOINT_SCENARIO: ${CHECKPOINT_SCENARIO}"
echo "FINAL_DIR  : ${FINAL_DIR}"
echo "REBUILD_INFOS: ${REBUILD_INFOS}"
if [[ "${NUM_SAMPLES}" == "0" ]]; then
  echo "NUM_SAMPLES: FULL DATA, co the rat lau neu infos co hang nghin frame"
else
  echo "NUM_SAMPLES: ${NUM_SAMPLES}"
fi
echo "============================================================"

cd "${MAPTR_ROOT}"

"${CONDA_PY}" "${PHENIKAA_ROOT}/benchmark_maptr/11_infer_visualize_maptr_phenikaa.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --infos "${INFOS}" \
  --out-dir "${INTERMEDIATE_DIR}" \
  --num-samples "${NUM_SAMPLES}" \
  --start-index "${START_INDEX}" \
  --score-thresh "${SCORE_THRESH}" \
  --image-scale "${IMAGE_SCALE}" \
  ${NO_GT_ARG}

if [[ -f "${INTERMEDIATE_DIR}/predictions.json" ]]; then
  cp "${INTERMEDIATE_DIR}/predictions.json" "${FINAL_DIR}/predictions.json"
else
  echo "[WARN] Khong thay predictions.json tong. Gom lai tu tung */prediction.json..."
  "${CONDA_PY}" -c "import json, pathlib; base=pathlib.Path('${INTERMEDIATE_DIR}'); data=[]; [data.append(json.load(p.open(encoding='utf-8'))) for p in sorted(base.glob('*/prediction.json'))]; out=pathlib.Path('${FINAL_DIR}')/'predictions.json'; out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8'); print('[OK] collected', len(data), 'samples ->', out)"
fi
if [[ -f "${INTERMEDIATE_DIR}/predictions.pkl" ]]; then
  cp "${INTERMEDIATE_DIR}/predictions.pkl" "${FINAL_DIR}/predictions.pkl"
fi

"${CONDA_PY}" "${PHENIKAA_ROOT}/benchmark_maptr/21_postprocess_predictions_to_thin_osm.py" \
  --predictions "${FINAL_DIR}/predictions.json" \
  --infos "${INFOS}" \
  --out-osm "${OSM_THIN_OUT}" \
  --score-thresh "${SCORE_THRESH}"

"${CONDA_PY}" "${PHENIKAA_ROOT}/benchmark_maptr/22_export_prediction_osm_graph.py" \
  --predictions "${FINAL_DIR}/predictions.json" \
  --infos "${INFOS}" \
  --out-osm "${OSM_GRAPH_OUT}" \
  --score-thresh "${SCORE_THRESH}"

"${CONDA_PY}" "${PHENIKAA_ROOT}/benchmark_maptr/16_visualize_maptr_paper_style.py" \
  --pred-dir "${INTERMEDIATE_DIR}" \
  --infos "${INFOS}" \
  --out-dir "${VIS_MAP_DIR}" \
  --style map \
  --max-samples "${VIS_MAX_SAMPLES}" \
  --score-thresh "${SCORE_THRESH}"

"${CONDA_PY}" "${PHENIKAA_ROOT}/benchmark_maptr/16_visualize_maptr_paper_style.py" \
  --pred-dir "${INTERMEDIATE_DIR}" \
  --infos "${INFOS}" \
  --out-dir "${VIS_CAMERA_DIR}" \
  --style paper \
  --max-samples "${VIS_MAX_SAMPLES}" \
  --score-thresh "${SCORE_THRESH}"

if [[ "${MODE}" == "metric" ]]; then
  "${CONDA_PY}" "${PHENIKAA_ROOT}/benchmark_maptr/13_evaluate_maptr_predictions.py" \
    --in-dir "${INTERMEDIATE_DIR}" \
    --out-csv "${FINAL_DIR}/evaluation_metrics.csv"
fi

cat > "${FINAL_DIR}/README_RESULT.txt" <<EOF
MODE=${MODE}
INFOS=${INFOS}
CHECKPOINT=${CHECKPOINT}
SCORE_THRESH=${SCORE_THRESH}
START_INDEX=${START_INDEX}
NUM_SAMPLES=${NUM_SAMPLES}

Ket qua chinh:
- predictions.json: ${FINAL_DIR}/predictions.json
- predicted_vector_map_thin.osm: ${OSM_THIN_OUT}
  - ban da gom/downsample, nen mo file nay de xem moi vach/lane chi con mot duong.
- predicted_vector_map_graph.osm: ${OSM_GRAPH_OUT}
  - ban nen mo de xem cac doan cong, nhanh re, vach ke, ped_crossing, speed_bump.
  - File nay giu tung vector sau khi bo trung lap vua phai, khong ep thanh vai duong dai.
- bev_map: ${VIS_MAP_DIR}
  - no_metric: moi sample co PRED_MAP.png
  - metric: moi sample co PRED_MAP.png, GT_MAP.png, OVERLAY_MAP.png
- camera_context: ${VIS_CAMERA_DIR}
  - anh 12 camera quanh xe + panel Prediction/GT de xem pred co hop voi moi truong khong

Ghi chu ve OSM:
- predicted_vector_map_graph.osm dung de xem hinh hoc prediction ro nhat.
- predicted_vector_map_thin.osm dung de xem lanelet cuc gon.

Thu muc _intermediate chi la file phu de cac script doc lai.
EOF

echo "============================================================"
echo "[DONE]"
echo "Ket qua chinh:"
echo "  ${FINAL_DIR}/predictions.json"
echo "  ${OSM_THIN_OUT}"
echo "  ${OSM_GRAPH_OUT}"
echo "  ${VIS_MAP_DIR}"
echo "  ${VIS_CAMERA_DIR}"
if [[ "${MODE}" == "metric" ]]; then
  echo "  ${FINAL_DIR}/evaluation_metrics.csv"
  echo "  ${FINAL_DIR}/evaluation_metrics.summary.json"
fi
echo "============================================================"
