#!/usr/bin/env python3
"""
B1 — Chấm điểm mAP (chỉ camera) trên nuScenes mini.

Trước khi chạy (mỗi terminal mới):
    conda activate maptr

Chạy:
    cd /home/khanh247/Documents/Survey/phenikaa
    python scripts/run_b1_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _maptr_env import parse_metrics, run_eval  # noqa: E402
from ghi_bao_cao import ghi_ket_qua_b1  # noqa: E402

# =============================================================================
# CONFIG — CHỈ SỬA PHẦN NÀY
# =============================================================================

# Thư mục gốc dự án (đổi nếu bạn copy project sang máy/chỗ khác)
PROJECT_ROOT = Path("/home/khanh247/Documents/Survey/phenikaa")

# Đường dẫn TƯƠNG ĐỐI từ third_party/MapTR/
MAPTR_CONFIG = "projects/configs/maptrv2/maptrv2_nusc_r50_24ep_eval_8g.py"
CHECKPOINT = "ckpts/maptrv2_nusc_r50_24ep.pth"

# Thư mục lưu kết quả (tuyệt đối hoặc tương đối PROJECT_ROOT)
OUTPUT_DIR = PROJECT_ROOT / "outputs/nusc_eval/e1_camera"
EVAL_LOG_NAME = "eval_log.txt"

# 1 GPU = RTX 4060 8GB
GPU_COUNT = 1

# True = tự ghi số vào docs/BAO_CAO_QUA_TRINH.docx sau khi chạy xong
GHI_BAO_CAO = True

# =============================================================================


def main() -> int:
    log_path = run_eval(
        project_root=PROJECT_ROOT,
        config_rel=MAPTR_CONFIG,
        checkpoint_rel=CHECKPOINT,
        output_dir=OUTPUT_DIR,
        log_name=EVAL_LOG_NAME,
        gpu_count=GPU_COUNT,
    )

    metrics = parse_metrics(log_path)
    print("\n--- KẾT QUẢ B1 ---")
    print(f"divider      = {metrics['divider']:.3f}")
    print(f"ped_crossing = {metrics['ped_crossing']:.3f}")
    print(f"boundary     = {metrics['boundary']:.3f}")
    print(f"mAP          = {metrics['map']:.3f}")

    if GHI_BAO_CAO:
        ghi_ket_qua_b1(PROJECT_ROOT, metrics, log_path)
        print("Đã ghi vào docs/BAO_CAO_QUA_TRINH.docx")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
