#!/usr/bin/env python3
"""
B2 — Chấm điểm mAP (camera + LiDAR fusion) trên nuScenes mini.

Trước khi chạy:
    conda activate maptr

Chạy:
    cd /home/khanh247/Documents/Survey/phenikaa
    python scripts/run_b2_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _maptr_env import parse_metrics, run_eval  # noqa: E402
from ghi_bao_cao import ghi_ket_qua_b2  # noqa: E402

# =============================================================================
# CONFIG — CHỈ SỬA PHẦN NÀY
# =============================================================================

PROJECT_ROOT = Path("/home/khanh247/Documents/Survey/phenikaa")

MAPTR_CONFIG = "projects/configs/maptr/maptr_tiny_fusion_eval_8g.py"
CHECKPOINT = "ckpts/maptr_tiny_fusion_24e.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs/nusc_eval/e2_fusion"
EVAL_LOG_NAME = "eval_log.txt"

GPU_COUNT = 1

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
    print("\n--- KẾT QUẢ B2 ---")
    print(f"divider      = {metrics['divider']:.3f}")
    print(f"ped_crossing = {metrics['ped_crossing']:.3f}")
    print(f"boundary     = {metrics['boundary']:.3f}")
    print(f"mAP          = {metrics['map']:.3f}")

    if GHI_BAO_CAO:
        ghi_ket_qua_b2(PROJECT_ROOT, metrics, log_path)
        print("Đã ghi vào docs/BAO_CAO_QUA_TRINH.docx")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
