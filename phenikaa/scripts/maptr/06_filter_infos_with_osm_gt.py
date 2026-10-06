#!/usr/bin/env python3
"""Filter Phenikaa infos to frames that contain trainable OSM GT vectors.

This step is meant to make MapTR training practical on a local workstation.
It scans ego poses against the OSM vector map without loading camera images,
then writes a smaller infos.pkl that only keeps frames with visible GT inside
the BEV patch.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

from pathlib import Path

PHENIKAA_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PHENIKAA_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from typing import Any

import numpy as np


MAPTR_ROOT = PHENIKAA_ROOT / "third_party" / "MapTR_phenikaa"
DEFAULT_MAP_CLASSES = [
    "lane_divider",
    "road_edge_marking",
    "stop_line",
    "ped_crossing",
    "boundary",
    "speed_bump",
]

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from phenikaa_maptr.pipeline_config import expand_config_value, load_pipeline_config  # noqa: E402


def add_maptr_paths() -> None:
    for path in (MAPTR_ROOT / "mmdetection3d", MAPTR_ROOT):
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)


def patch_numpy_legacy_aliases() -> None:
    for name, value in {
        "long": int,
        "int": int,
        "float": float,
        "bool": bool,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def make_matplotlib_cache() -> None:
    cache_dir = Path("/tmp") / "phenikaa_maptr_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def parse_args() -> argparse.Namespace:
    cfg = load_pipeline_config()
    vars_cfg = cfg["_vars"]
    filter_cfg = cfg.get("filter_infos", {})
    train_cfg = cfg.get("train", {})
    parser = argparse.ArgumentParser(description="Filter infos.pkl by trainable OSM GT presence.")
    parser.add_argument("--infos", type=Path, default=Path(vars_cfg["INFOS"]))
    parser.add_argument("--osm", type=Path, default=Path(vars_cfg["GT_OSM"]))
    parser.add_argument(
        "--out-pkl",
        type=Path,
        default=Path(expand_config_value(filter_cfg.get("out_infos", ""), cfg)) if filter_cfg.get("out_infos") else None,
    )
    parser.add_argument("--min-gt-vectors", type=int, default=int(filter_cfg.get("min_gt_vectors", 1)))
    parser.add_argument("--progress-every", type=int, default=int(filter_cfg.get("progress_every", 500)))
    parser.add_argument("--fixed-ptsnum-per-line", type=int, default=int(train_cfg.get("fixed_ptsnum_per_line", 20)))
    parser.add_argument("--pc-range", nargs=6, type=float, default=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0])
    parser.add_argument("--map-classes", nargs="+", default=DEFAULT_MAP_CLASSES)
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def dump_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def info_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("infos"), list):
        return data["infos"]
    if isinstance(data, list):
        return data
    raise ValueError("infos.pkl phai la dict co key 'infos' hoac list infos.")


def default_out_path(infos_path: Path) -> Path:
    return infos_path.with_name(infos_path.stem + "_gt_positive.pkl")


def count_gt(vector_map, info: dict[str, Any]) -> int:
    from pyquaternion import Quaternion

    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    lidar2ego[:3, 3] = info["lidar2ego_translation"]

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
    ego2global[:3, 3] = info["ego2global_translation"]

    lidar2global = ego2global @ lidar2ego
    anns = vector_map.gen_vectorized_samples(
        info.get("map_location", "phenikaa"),
        list(lidar2global[:3, 3]),
        list(Quaternion(matrix=lidar2global).q),
    )
    return len(anns.get("gt_vecs_label", []))


def main() -> None:
    args = parse_args()
    infos_path = args.infos.expanduser().resolve()
    osm_path = args.osm.expanduser().resolve()
    out_path = (args.out_pkl or default_out_path(infos_path)).expanduser().resolve()

    if not infos_path.exists():
        raise FileNotFoundError(infos_path)
    if not osm_path.exists():
        raise FileNotFoundError(osm_path)

    make_matplotlib_cache()
    patch_numpy_legacy_aliases()
    add_maptr_paths()

    from projects.mmdet3d_plugin.datasets.nuscenes_map_dataset import OSMVectorizedLocalMap

    data = load_pickle(infos_path)
    infos = info_list(data)
    patch_h = float(args.pc_range[4] - args.pc_range[1])
    patch_w = float(args.pc_range[3] - args.pc_range[0])
    vector_map = OSMVectorizedLocalMap(
        str(osm_path),
        patch_size=(patch_h, patch_w),
        map_classes=args.map_classes,
        fixed_ptsnum_per_line=args.fixed_ptsnum_per_line,
    )

    kept_infos = []
    gt_counts = []
    for idx, info in enumerate(infos):
        gt_num = count_gt(vector_map, info)
        if gt_num >= args.min_gt_vectors:
            kept_infos.append(info)
            gt_counts.append(gt_num)
        if args.progress_every > 0 and ((idx + 1) % args.progress_every == 0 or idx + 1 == len(infos)):
            print(f"[FILTER] scanned={idx + 1}/{len(infos)} kept={len(kept_infos)} last_gt={gt_num}")

    if isinstance(data, dict):
        out_data = dict(data)
        out_data["infos"] = kept_infos
        metadata = dict(out_data.get("metadata", {}))
        metadata["source_infos"] = str(infos_path)
        metadata["gt_osm"] = str(osm_path)
        metadata["filter_min_gt_vectors"] = args.min_gt_vectors
        metadata["filter_map_classes"] = list(args.map_classes)
        out_data["metadata"] = metadata
    else:
        out_data = kept_infos

    dump_pickle(out_data, out_path)

    print("[DONE]")
    print(f"Source infos : {infos_path}")
    print(f"OSM GT       : {osm_path}")
    print(f"Output infos : {out_path}")
    print(f"Kept frames  : {len(kept_infos)} / {len(infos)}")
    if gt_counts:
        print(f"GT min/mean/max: {min(gt_counts)} / {sum(gt_counts) / len(gt_counts):.2f} / {max(gt_counts)}")
    else:
        print("[WARN] Khong giu duoc frame nao. Kiem tra OSM, pose, pc_range, semantic_class.")


if __name__ == "__main__":
    main()
