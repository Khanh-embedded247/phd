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
    print(f"Map ann file : {dataset.map_ann_file}")
    print(f"Map classes  : {dataset.MAPCLASSES}")

    for idx in range(min(args.num_samples, len(dataset))):
        # Goi truc tiep prepare_train_data de thay sample nao bi filter.
        # __getitem__ cua MMDetection se random sang sample khac khi item=None,
        # nen khong phu hop cho viec debug vung GT OSM.
        item = dataset.prepare_train_data(idx)
        checked += 1
        if item is None:
            print(f"{idx:04d}: EMPTY/filtered")
            continue
        keys = list(item.keys())
        gt_num = count_gt(item)
        valid += 1
        gt_counts.append(gt_num)
        print(f"{idx:04d}: keys={keys} gt_vectors={gt_num}")

    print("[SUMMARY]")
    print(f"Checked samples : {checked}")
    print(f"Valid samples   : {valid}")
    if gt_counts:
        print(f"GT min/mean/max : {min(gt_counts)} / {sum(gt_counts)/len(gt_counts):.2f} / {max(gt_counts)}")
    print("Dataset OSM check done.")


if __name__ == "__main__":
    main()
