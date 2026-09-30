#!/usr/bin/env python3
"""
B2 — Vẽ hình fusion (tuỳ chọn, chạy SAU run_b2_eval.py).

Trước khi chạy:
    conda activate maptr

Chạy:
    cd /home/khanh247/Documents/Survey/phenikaa
    python scripts/run_b2_vis.py
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

MAPTR_CONFIG = "projects/configs/maptr/maptr_tiny_fusion_eval_8g.py"
CHECKPOINT = "ckpts/maptr_tiny_fusion_24e.pth"

SHOW_DIR = PROJECT_ROOT / "outputs/nusc_eval/e2_fusion/vis_pred"

SCORE_THRESH = 0.3

# MapTR v1 vis không có --max-samples; để None = script upstream tự chọn
MAX_SAMPLES = None

# =============================================================================


def main() -> int:
    run_vis(
        project_root=PROJECT_ROOT,
        vis_script_rel="tools/maptr/vis_pred.py",
        config_rel=MAPTR_CONFIG,
        checkpoint_rel=CHECKPOINT,
        show_dir=SHOW_DIR,
        score_thresh=SCORE_THRESH,
        max_samples=MAX_SAMPLES,
        extra_args=["--show-cam"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
