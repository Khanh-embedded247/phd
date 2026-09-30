#!/usr/bin/env python3
"""
Check whether candidate map.pcd files overlap Phenikaa Pose translations.

Heuristic only (AABB vs ego XY). Pass ⇒ safe to promote as residential.pcd.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phenikaa_hdmap.paths import (  # noqa: E402
    MAP_CANDIDATES,
    MAPS_ROOT,
    SEQUENCE_FOLDER,
    collect_timestamps,
)
from phenikaa_hdmap.utils.pose import load_pose_3x4, translation  # noqa: E402


def read_pcd_xyz_sample(pcd_path: Path, max_points: int = 200_000) -> np.ndarray:
    """Read XYZ from PCD (ascii or binary). Returns (N,3) float64 sample."""
    with open(pcd_path, "rb") as f:
        fields: list[str] = []
        sizes: list[int] = []
        types: list[str] = []
        counts: list[int] = []
        n_points = 0
        data_type = "ascii"
        header_ended = False
        while not header_ended:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PCD header: {pcd_path}")
            text = line.decode("ascii", errors="ignore").strip()
            if text.startswith("FIELDS"):
                fields = text.split()[1:]
            elif text.startswith("SIZE"):
                sizes = list(map(int, text.split()[1:]))
            elif text.startswith("TYPE"):
                types = text.split()[1:]
            elif text.startswith("COUNT"):
                counts = list(map(int, text.split()[1:]))
            elif text.startswith("POINTS"):
                n_points = int(text.split()[1])
            elif text.startswith("DATA"):
                data_type = text.split()[1].lower()
                header_ended = True

        if not fields or "x" not in fields or "y" not in fields or "z" not in fields:
            raise ValueError(f"PCD missing xyz fields: {fields}")

        ix, iy, iz = fields.index("x"), fields.index("y"), fields.index("z")
        take = min(n_points, max_points)

        if data_type == "ascii":
            pts = []
            for _ in range(take):
                row = f.readline().decode("ascii", errors="ignore").split()
                if len(row) < len(fields):
                    break
                pts.append((float(row[ix]), float(row[iy]), float(row[iz])))
            return np.asarray(pts, dtype=np.float64)

        if data_type != "binary":
            raise ValueError(f"Unsupported PCD DATA type '{data_type}' in {pcd_path}")

        if not sizes:
            sizes = [4] * len(fields)
        if not types:
            types = ["F"] * len(fields)
        if not counts:
            counts = [1] * len(fields)

        type_map = {
            ("F", 4): "f",
            ("F", 8): "d",
            ("U", 1): "B",
            ("U", 2): "H",
            ("U", 4): "I",
            ("I", 1): "b",
            ("I", 2): "h",
            ("I", 4): "i",
        }
        fmt = ""
        for t, s, c in zip(types, sizes, counts):
            key = (t, s)
            if key not in type_map:
                raise ValueError(f"Unsupported PCD field type {t}/{s}")
            fmt += type_map[key] * c
        point_size = struct.calcsize(fmt)
        xs, ys, zs = [], [], []
        for i in range(take):
            blob = f.read(point_size)
            if len(blob) < point_size:
                break
            vals = struct.unpack(fmt, blob)
            xs.append(vals[ix])
            ys.append(vals[iy])
            zs.append(vals[iz])
        return np.column_stack([xs, ys, zs]).astype(np.float64)


def ego_xy_from_poses(sequence_folder: Path) -> np.ndarray:
    stamps = collect_timestamps(sequence_folder)
    xy = []
    for ts in stamps:
        t = translation(load_pose_3x4(sequence_folder / "Pose" / f"{ts}.txt"))
        xy.append(t[:2])
    return np.asarray(xy, dtype=np.float64)


def overlap_report(ego_xy: np.ndarray, map_xyz: np.ndarray, margin: float = 50.0) -> dict:
    ego_min, ego_max = ego_xy.min(0), ego_xy.max(0)
    map_min, map_max = map_xyz[:, :2].min(0), map_xyz[:, :2].max(0)
    # expand ego bbox by margin
    e0 = ego_min - margin
    e1 = ego_max + margin
    overlap = not (e1[0] < map_min[0] or e0[0] > map_max[0] or e1[1] < map_min[1] or e0[1] > map_max[1])
    # fraction of ego positions inside map AABB
    inside = (
        (ego_xy[:, 0] >= map_min[0])
        & (ego_xy[:, 0] <= map_max[0])
        & (ego_xy[:, 1] >= map_min[1])
        & (ego_xy[:, 1] <= map_max[1])
    )
    return {
        "overlap_aabb": bool(overlap),
        "ego_inside_frac": float(inside.mean()) if len(inside) else 0.0,
        "ego_xy_min": ego_min.tolist(),
        "ego_xy_max": ego_max.tolist(),
        "map_xy_min": map_min.tolist(),
        "map_xy_max": map_max.tolist(),
        "n_map_points_sampled": int(map_xyz.shape[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pcd",
        type=Path,
        nargs="*",
        default=None,
        help="PCD files to test (default: map candidates)",
    )
    parser.add_argument("--margin", type=float, default=50.0)
    args = parser.parse_args()

    candidates = args.pcd if args.pcd else [p for p in MAP_CANDIDATES if p.is_file()]
    if not candidates:
        print("No PCD candidates found under data/phenikaa/maps/")
        return 1

    print(f"Loading ego XY from {SEQUENCE_FOLDER} ...")
    ego_xy = ego_xy_from_poses(SEQUENCE_FOLDER)
    print(f"  frames={len(ego_xy)}  ego_xy range={ego_xy.min(0)} → {ego_xy.max(0)}\n")

    best = None
    for pcd in candidates:
        print(f"=== {pcd} ===")
        try:
            xyz = read_pcd_xyz_sample(pcd)
            rep = overlap_report(ego_xy, xyz, margin=args.margin)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}\n")
            continue
        for k, v in rep.items():
            print(f"  {k}: {v}")
        score = rep["ego_inside_frac"]
        if best is None or score > best[0]:
            best = (score, pcd, rep)
        verdict = "PASS" if score > 0.5 and rep["overlap_aabb"] else "FAIL"
        print(f"  verdict: {verdict}\n")

    if best is None:
        print("No readable PCD.")
        return 1

    score, pcd, rep = best
    print(f"Best candidate: {pcd} (ego_inside_frac={score:.3f})")
    if score > 0.5 and rep["overlap_aabb"]:
        target = MAPS_ROOT / "residential.pcd"
        print(
            f"\nNext:\n  ln -sfn '{pcd.resolve()}' '{target}'\n"
            "then re-run scripts/check_setup.py"
        )
        return 0

    print(
        "\nNo candidate overlaps Pose well. Ask Phenikaa team for the "
        "RESIDENTIAL static map.pcd in the same frame as Pose/."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
