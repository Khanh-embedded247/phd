"""
Rasterize a cropped static map prior around ego into BEV channels.

TODO (Tuần 6):
  - Load residential.pcd
  - Transform to ego / BEV grid using Pose
  - Export density / height / intensity tensors for fusion
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-pcd", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bev-h", type=int, default=200)
    parser.add_argument("--bev-w", type=int, default=100)
    parser.add_argument("--x-range", type=float, nargs=2, default=[-15.0, 15.0])
    parser.add_argument("--y-range", type=float, nargs=2, default=[-30.0, 30.0])
    args = parser.parse_args()

    raise NotImplementedError(
        "Implement after verify_map_pose.py confirms map ↔ Pose alignment. "
        f"Would rasterize {args.map_pcd} with pose {args.pose} → {args.out}"
    )


if __name__ == "__main__":
    main()
