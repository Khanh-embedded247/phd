#!/usr/bin/env python3
"""Kiem tra dataset train MapTR co doc duoc GT tu OSM hay khong.

Buoc nay CHUA train. No build data.train tu config train, lay mot so sample,
va in so vector GT sinh ra tu normal_main.osm.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


PHENIKAA_ROOT = Path(__file__).resolve().parents[1]
SURVEY_ROOT = PHENIKAA_ROOT.parents[1]
MAPTR_ROOT = PHENIKAA_ROOT / "benchmark_maptr" / "vendor" / "MapTR"
DEFAULT_CONFIG = PHENIKAA_ROOT / "benchmark_maptr" / "config" / "maptr_tiny_r50_phenikaa_12cam_train_osm.py"

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from pipeline_config import load_pipeline_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    pipe_cfg = load_pipeline_config()
    vars_cfg = pipe_cfg["_vars"]
    train_cfg = pipe_cfg.get("train", {})
    parser = argparse.ArgumentParser(description="Kiem tra train dataset OSM.")
    parser.add_argument("--config", type=Path, default=Path(vars_cfg.get("CONFIG", DEFAULT_CONFIG)))
    parser.add_argument("--num-samples", type=int, default=int(train_cfg.get("check_num_samples", 30)))
    parser.add_argument(
        "--max-scan-for-valid",
        type=int,
        default=int(train_cfg.get("check_max_scan_for_valid", 1000)),
        help="Neu cac sample dau rong, scan them toi N index de tim frame co GT.",
    )
    return parser.parse_args()


def make_matplotlib_cache() -> None:
    cache_dir = Path("/tmp") / "phenikaa_maptr_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def patch_numpy_legacy_aliases() -> None:
    for name, value in {
        "long": int,
        "int": int,
        "float": float,
        "bool": bool,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def add_maptr_paths() -> None:
    for path in (MAPTR_ROOT / "mmdetection3d", MAPTR_ROOT):
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)


def count_gt(item) -> int:
    labels = item["gt_labels_3d"]
    labels_data = getattr(labels, "_data", labels)
    return int(labels_data.numel()) if hasattr(labels_data, "numel") else len(labels_data)


def fast_count_gt_from_pose(dataset, index: int) -> int:
    """Dem GT vector khong load anh.

    `prepare_train_data()` doc 12 anh nen scan sau rat cham. Ham nay chi lay
    pose/calib tu infos, goi OSM vector_map cat patch quanh xe, va dem label.
    """
    from pyquaternion import Quaternion

    input_dict = dataset.get_data_info(index)
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(input_dict["lidar2ego_rotation"]).rotation_matrix
    lidar2ego[:3, 3] = input_dict["lidar2ego_translation"]
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(input_dict["ego2global_rotation"]).rotation_matrix
    ego2global[:3, 3] = input_dict["ego2global_translation"]
    lidar2global = ego2global @ lidar2ego
    anns = dataset.vector_map.gen_vectorized_samples(
        input_dict["map_location"],
        list(lidar2global[:3, 3]),
        list(Quaternion(matrix=lidar2global).q),
    )
    return len(anns.get("gt_vecs_label", []))


def leaf_datasets(dataset):
    """Lay cac dataset that ben trong ConcatDataset/RepeatDataset."""
    if hasattr(dataset, "dataset"):
        yield from leaf_datasets(dataset.dataset)
    elif hasattr(dataset, "datasets"):
        for child in dataset.datasets:
            yield from leaf_datasets(child)
    else:
        yield dataset


def check_one_dataset(dataset, num_samples: int, max_scan_for_valid: int, prefix: str) -> tuple[int, int, list[int]]:
    checked = 0
    valid = 0
    gt_counts = []
    valid_indices = []
    for idx in range(min(num_samples, len(dataset))):
        item = dataset.prepare_train_data(idx)
        checked += 1
        if item is None:
            print(f"{prefix}{idx:04d}: EMPTY/filtered")
            continue
        keys = list(item.keys())
        gt_num = count_gt(item)
        valid += 1
        valid_indices.append(idx)
        gt_counts.append(gt_num)
        print(f"{prefix}{idx:04d}: keys={keys} gt_vectors={gt_num}")

    # Nhieu scenario co GT chi nam trong mot doan duong da annotate, nen cac
    # frame dau co the rong. Neu doan dau rong, scan them de xac nhan dataset
    # van co frame train hop le thay vi ket luan nham la OSM/config hong.
    if valid == 0 and max_scan_for_valid > num_samples:
        scan_limit = min(max_scan_for_valid, len(dataset))
        first_valid = []
        for idx in range(num_samples, scan_limit):
            checked += 1
            gt_num = fast_count_gt_from_pose(dataset, idx)
            if gt_num <= 0:
                continue
            valid += 1
            valid_indices.append(idx)
            gt_counts.append(gt_num)
            first_valid.append((idx, gt_num))
            if len(first_valid) >= min(num_samples, 10):
                break
        if first_valid:
            print(f"{prefix} first valid after scan: " + ", ".join(f"idx={idx} gt={gt}" for idx, gt in first_valid))
        else:
            print(f"{prefix} no valid GT found in first {scan_limit} samples")
    return checked, valid, gt_counts


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    make_matplotlib_cache()
    patch_numpy_legacy_aliases()
    add_maptr_paths()

    from mmcv import Config
    from mmdet.datasets import build_dataset

    import projects.mmdet3d_plugin  # noqa: F401

    cfg = Config.fromfile(str(config_path))
    dataset_cfg = cfg.data.train.copy()
    dataset_cfg.pop("samples_per_gpu", None)
    dataset = build_dataset(dataset_cfg)

    checked = 0
    valid = 0
    gt_counts = []
    print("[TRAIN DATASET OSM CHECK]")
    print(f"Config       : {config_path}")
    print(f"Dataset type : {type(dataset).__name__}")
    print(f"Dataset len  : {len(dataset)}")

    leaves = list(leaf_datasets(dataset))
    print(f"Leaf datasets: {len(leaves)}")
    for ds_idx, leaf in enumerate(leaves):
        print("")
        print(f"[LEAF {ds_idx}] type={type(leaf).__name__} len={len(leaf)}")
        print(f"Map ann file : {getattr(leaf, 'map_ann_file', '')}")
        print(f"Ann file     : {getattr(leaf, 'ann_file', '')}")
        print(f"Map classes  : {getattr(leaf, 'MAPCLASSES', '')}")
        c, v, counts = check_one_dataset(
            leaf,
            args.num_samples,
            args.max_scan_for_valid,
            prefix=f"d{ds_idx}:",
        )
        checked += c
        valid += v
        gt_counts.extend(counts)

    print("[SUMMARY]")
    print(f"Checked samples : {checked}")
    print(f"Valid samples   : {valid}")
    if gt_counts:
        print(f"GT min/mean/max : {min(gt_counts)} / {sum(gt_counts)/len(gt_counts):.2f} / {max(gt_counts)}")
    print("Dataset OSM check done.")


if __name__ == "__main__":
    main()
