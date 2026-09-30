#!/usr/bin/env python3
"""
In 4 số mAP từ log — không chạy lại model.

Chạy:
    python scripts/xem_ket_qua.py b1
    python scripts/xem_ket_qua.py b2
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _maptr_env import parse_metrics  # noqa: E402

# =============================================================================
# CONFIG — CHỈ SỬA PHẦN NÀY
# =============================================================================

PROJECT_ROOT = Path("/home/khanh247/Documents/Survey/phenikaa")

LOG_B1 = PROJECT_ROOT / "outputs/nusc_eval/e1_camera/eval_log.txt"
LOG_B2 = PROJECT_ROOT / "outputs/nusc_eval/e2_fusion/eval_log.txt"

# =============================================================================


def print_metrics(label: str, log_path: Path) -> None:
    if not log_path.is_file():
        print(f"[{label}] Chưa có log: {log_path}")
        print(f"Chạy trước: python scripts/run_{label}_eval.py")
        return

    m = parse_metrics(log_path)
    print(f"=== {label.upper()} ({log_path}) ===")
    print(f"divider      = {m['divider']:.3f}")
    print(f"ped_crossing = {m['ped_crossing']:.3f}")
    print(f"boundary     = {m['boundary']:.3f}")
    print(f"mAP          = {m['map']:.3f}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Cách dùng: python scripts/xem_ket_qua.py b1|b2")
        return 1

    step = sys.argv[1].lower()
    if step == "b1":
        print_metrics("b1", LOG_B1)
    elif step == "b2":
        print_metrics("b2", LOG_B2)
    else:
        print("Chỉ hỗ trợ: b1 hoặc b2")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
