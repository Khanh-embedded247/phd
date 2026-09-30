#!/usr/bin/env python3
"""
B1 — Vẽ hình dự đoán làn (tuỳ chọn, chạy SAU run_b1_eval.py).

Trước khi chạy:
    conda activate maptr

Chạy:
    cd /home/khanh247/Documents/Survey/phenikaa
    python scripts/run_b1_vis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _maptr_env import run_vis  # noqa: E402

# =============================================================================
# CONFIG — CHỈ SỬA PHẦN NÀY
# =============================================================================

PROJECT_ROOT = Path("/home/khanh247/Documents/Survey/phenikaa")

MAPTR_CONFIG = "projects/configs/maptrv2/maptrv2_nusc_r50_24ep_eval_8g.py"
CHECKPOINT = "ckpts/maptrv2_nusc_r50_24ep.pth"

# Thư mục lưu ảnh BEV + surround view
SHOW_DIR = PROJECT_ROOT / "outputs/nusc_eval/e1_camera/vis_pred"

# Ngưỡng score (0.2–0.5). Thấp hơn = vẽ nhiều đường hơn
SCORE_THRESH = 0.3

# Số frame vẽ (None = vẽ hết — lâu). Khuyên 8
MAX_SAMPLES = 8

# =============================================================================


def main() -> int:
    run_vis(
        project_root=PROJECT_ROOT,
        vis_script_rel="tools/maptrv2/nusc_vis_pred.py",
        config_rel=MAPTR_CONFIG,
        checkpoint_rel=CHECKPOINT,
        show_dir=SHOW_DIR,
        score_thresh=SCORE_THRESH,
        max_samples=MAX_SAMPLES,
    )
    print("\nMở thư mục vis_pred, so từng frame:")
    print("  PRED_MAP_plot.png  = model đoán")
    print("  GT_fixednum_pts_MAP.png = đáp án")
    print("  surroud_view.jpg   = 6 camera")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
