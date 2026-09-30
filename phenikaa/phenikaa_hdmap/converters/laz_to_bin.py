"""
Convert Phenikaa .laz frame(s) to MapTR fusion .bin points.

Useful when adapting MapTR LiDAR pipelines that expect nuScenes-style bins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def laz_to_xyzir(laz_path: Path) -> np.ndarray:
    import laspy

    las = laspy.read(str(laz_path))
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    if "intensity" in las.point_format.dimension_names:
        intensity = np.asarray(las.intensity, dtype=np.float32).reshape(-1, 1)
    else:
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    ring = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    return np.hstack([xyz, intensity, ring])


def save_bin(points: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    points.astype(np.float32).tofile(str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A .laz file or a directory containing .laz files.")
    parser.add_argument("-o", "--out", type=Path, required=True, help="Output .bin file or output directory.")
    args = parser.parse_args()

    if args.input.is_dir():
        args.out.mkdir(parents=True, exist_ok=True)
        laz_files = sorted(args.input.glob("*.laz"))
        if not laz_files:
            raise FileNotFoundError(f"No .laz files found in {args.input}")
        for laz_path in laz_files:
            out_path = args.out / f"{laz_path.stem}.bin"
            pts = laz_to_xyzir(laz_path)
            save_bin(pts, out_path)
            print(f"{laz_path.name}: {pts.shape[0]} points -> {out_path}")
        return

    pts = laz_to_xyzir(args.input)
    save_bin(pts, args.out)
    print(f"{args.input.name}: {pts.shape[0]} points -> {args.out}")


if __name__ == "__main__":
    main()
