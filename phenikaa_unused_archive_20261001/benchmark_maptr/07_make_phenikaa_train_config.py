#!/usr/bin/env python3
"""Tao config train MapTR cho Phenikaa 12 camera voi GT OSM.

Config nay khac config smoke/inference o cho:
  - data.train.test_mode = False
  - data.train.map_ann_file tro toi normal_main.osm
  - train pipeline chi load/normalize/pad anh; GT vector duoc them trong
    CustomNuScenesLocalMapDataset.vectormap_pipeline tu OSM.
  - filter_empty_gt=True de bo frame nam ngoai vung OSM da ve.

Chay:
  python3 phd/phenikaa/benchmark_maptr/07_make_phenikaa_train_config.py
"""

from __future__ import annotations

import argparse
import os
import pprint
import sys
from pathlib import Path

import numpy as np


PHENIKAA_ROOT = Path(__file__).resolve().parents[1]
SURVEY_ROOT = PHENIKAA_ROOT.parents[1]
MAPTR_ROOT = PHENIKAA_ROOT / "benchmark_maptr" / "vendor" / "MapTR"
BASE_CONFIG = PHENIKAA_ROOT / "benchmark_maptr" / "config" / "maptr_tiny_r50_phenikaa_12cam.py"
DEFAULT_OSM = PHENIKAA_ROOT / "data" / "Normal" / "normal_main.osm"
DEFAULT_TRAIN_INFOS = PHENIKAA_ROOT / "outputs" / "benchmark_maptr" / "Normal" / "phenikaa_maptr_infos_train_osm.pkl"
DEFAULT_OUT_CONFIG = PHENIKAA_ROOT / "benchmark_maptr" / "config" / "maptr_tiny_r50_phenikaa_12cam_train_osm.py"
DEFAULT_WORK_DIR = PHENIKAA_ROOT / "outputs" / "benchmark_maptr" / "Normal" / "work_dirs" / "maptr_12cam_osm"
DEFAULT_RESNET = PHENIKAA_ROOT / "third_party" / "MapTR" / "ckpts" / "resnet50-19c8e357.pth"
DEFAULT_INIT_CKPT = PHENIKAA_ROOT / "outputs" / "benchmark_maptr" / "Normal" / "ckpts" / "maptr_init_12cam_partial.pth"
DEFAULT_TRAIN_IMAGE_SCALE = 0.25
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
from pipeline_config import expand_config_value, load_pipeline_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    pipe_cfg = load_pipeline_config()
    vars_cfg = pipe_cfg["_vars"]
    paths_cfg = pipe_cfg.get("_paths", {})
    train_cfg = pipe_cfg.get("train", {})
    parser = argparse.ArgumentParser(description="Tao config train MapTR Phenikaa OSM.")
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--osm", type=Path, default=Path(paths_cfg.get("gt_osm", DEFAULT_OSM)))
    parser.add_argument("--infos", type=Path, default=Path(vars_cfg.get("INFOS", DEFAULT_TRAIN_INFOS)))
    parser.add_argument("--out-config", type=Path, default=Path(vars_cfg.get("CONFIG", DEFAULT_OUT_CONFIG)))
    parser.add_argument("--work-dir", type=Path, default=Path(expand_config_value(train_cfg.get("work_dir", DEFAULT_WORK_DIR), pipe_cfg)))
    parser.add_argument("--resnet-pretrained", type=Path, default=DEFAULT_RESNET)
    parser.add_argument("--load-from", type=Path, default=Path(vars_cfg.get("INIT_CHECKPOINT", DEFAULT_INIT_CKPT)))
    parser.add_argument("--samples-per-gpu", type=int, default=int(train_cfg.get("samples_per_gpu", 1)))
    parser.add_argument("--workers-per-gpu", type=int, default=int(train_cfg.get("workers_per_gpu", 1)))
    parser.add_argument("--max-epochs", type=int, default=int(train_cfg.get("max_epochs", 12)))
    parser.add_argument("--lr", type=float, default=float(train_cfg.get("lr", 2e-4)))
    parser.add_argument(
        "--train-image-scale",
        type=float,
        default=float(train_cfg.get("image_scale", DEFAULT_TRAIN_IMAGE_SCALE)),
        help="Scale anh train. 12 cam 1920x1536 nen dung 0.25 cho GPU 8GB.",
    )
    return parser.parse_args()


def quote_path(path: Path) -> str:
    return str(path.expanduser().resolve()).replace("\\", "\\\\").replace("'", "\\'")


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


def add_maptr_paths() -> None:
    for path in (MAPTR_ROOT / "mmdetection3d", MAPTR_ROOT):
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)


def dump_config_dict(cfg_dict: dict) -> str:
    return "\n\n".join(
        f"{key} = {pprint.pformat(value, width=100, sort_dicts=False)}"
        for key, value in cfg_dict.items()
    ) + "\n"


def replace_key_recursive(obj, key: str, value) -> None:
    """Thay mot key trong dict/list nested cua mmcv config."""
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k == key:
                obj[k] = value
            else:
                replace_key_recursive(obj[k], key, value)
    elif isinstance(obj, list):
        for item in obj:
            replace_key_recursive(item, key, value)


def make_train_dataset_cfg(base_dataset, ann_file: Path, osm_file: Path, train_pipeline: list[dict], repeat: int = 1):
    """Tao dataset train cho mot scenario.

    Moi scenario phai giu dung cap `infos.pkl + gt.osm`, vi OSM va trajectory
    nam trong cung he toa do map cua scenario do. Ta khong gop OSM thanh mot
    file lon; thay vao do dung ConcatDataset/RepeatDataset o muc config.
    """
    dataset = base_dataset.copy()
    dataset.ann_file = quote_path(ann_file)
    dataset.map_ann_file = quote_path(osm_file)
    dataset.pipeline = train_pipeline
    dataset.test_mode = False
    dataset.filter_empty_gt = True
    dataset.queue_length = 1
    if int(repeat) > 1:
        return dict(type="RepeatDataset", times=int(repeat), dataset=dataset)
    return dataset


def train_datasets_from_yaml(pipe_cfg: dict, fallback_infos: Path, fallback_osm: Path) -> list[dict]:
    """Doc danh sach train_data.datasets tu YAML.

    Neu YAML chua khai bao multi-dataset, fallback ve mot dataset duy nhat de
    cac lenh cu van chay duoc.
    """
    datasets = pipe_cfg.get("train_data", {}).get("datasets", [])
    if not datasets:
        return [{"name": "single", "infos": fallback_infos, "gt_osm": fallback_osm, "repeat": 1}]
    out = []
    for item in datasets:
        name = str(item.get("name", f"dataset_{len(out)}"))
        infos = Path(expand_config_value(item["infos"], pipe_cfg)).expanduser().resolve()
        gt_osm = Path(expand_config_value(item["gt_osm"], pipe_cfg)).expanduser().resolve()
        repeat = int(item.get("repeat", 1))
        out.append({"name": name, "infos": infos, "gt_osm": gt_osm, "repeat": repeat})
    return out


def main() -> None:
    args = parse_args()
    pipe_cfg = load_pipeline_config()
    base_config = args.base_config.expanduser().resolve()
    osm_path = args.osm.expanduser().resolve()
    infos_path = args.infos.expanduser().resolve()
    out_config = args.out_config.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    resnet_path = args.resnet_pretrained.expanduser().resolve()
    load_from = args.load_from.expanduser().resolve()

    if not base_config.exists():
        raise FileNotFoundError(base_config)
    train_sources = train_datasets_from_yaml(pipe_cfg, infos_path, osm_path)
    for source in train_sources:
        if not source["gt_osm"].exists():
            raise FileNotFoundError(source["gt_osm"])
        if not source["infos"].exists():
            raise FileNotFoundError(source["infos"])

    make_matplotlib_cache()
    patch_numpy_legacy_aliases()
    add_maptr_paths()

    from mmcv import Config

    cfg = Config.fromfile(str(base_config))
    map_classes = list(DEFAULT_MAP_CLASSES)
    num_map_classes = len(map_classes)
    cfg.work_dir = quote_path(work_dir)
    cfg.total_epochs = args.max_epochs
    cfg.runner = dict(type="EpochBasedRunner", max_epochs=args.max_epochs)
    cfg.optimizer.lr = args.lr
    cfg.data.samples_per_gpu = args.samples_per_gpu
    cfg.data.workers_per_gpu = args.workers_per_gpu
    cfg.checkpoint_config = dict(interval=1, max_keep_ckpts=3)
    # TensorBoard cua torch 1.9/MMCV cu co the loi voi setuptools moi
    # (setuptools._distutils khong co version). Dung text log de train on dinh.
    cfg.log_config = dict(interval=10, hooks=[dict(type="TextLoggerHook")])
    cfg.evaluation = dict(interval=1, pipeline=cfg.test_pipeline, metric=None)
    cfg.map_classes = map_classes
    cfg.num_map_classes = num_map_classes
    cfg.model.pts_bbox_head.num_classes = num_map_classes
    cfg.model.pts_bbox_head.bbox_coder.num_classes = num_map_classes
    replace_key_recursive(cfg.data, "map_classes", map_classes)

    if resnet_path.exists():
        cfg.model.pretrained = dict(img=quote_path(resnet_path))
    else:
        cfg.model.pretrained = None

    # Pipeline tao config truoc roi moi tao init checkpoint, nen van ghi duong
    # dan load_from ke ca khi file chua ton tai tai thoi diem generate config.
    cfg.load_from = quote_path(load_from)
    cfg.resume_from = None

    train_pipeline = [
        dict(type="LoadMultiViewImageFromFiles", to_float32=True),
        dict(type="NormalizeMultiviewImage", **cfg.img_norm_cfg),
        dict(type="RandomScaleImageMultiViewImage", scales=[args.train_image_scale]),
        dict(type="PadMultiViewImage", size_divisor=32),
        dict(type="DefaultFormatBundle3D", class_names=cfg.class_names, with_label=False),
        dict(type="CustomCollect3D", keys=["img"]),
    ]
    cfg.train_pipeline = train_pipeline

    train_datasets = [
        make_train_dataset_cfg(
            cfg.data.train.copy(),
            source["infos"],
            source["gt_osm"],
            train_pipeline,
            repeat=source["repeat"],
        )
        for source in train_sources
    ]
    # MMCV/MMDetection build_dataset doc duoc ConcatDataset. RepeatDataset giup
    # can bang scenario nho ma khong can copy file anh/OSM.
    train_dataset = train_datasets[0] if len(train_datasets) == 1 else dict(type="ConcatDataset", datasets=train_datasets)

    val_dataset = cfg.data.val.copy()
    val_dataset.ann_file = quote_path(infos_path)
    val_dataset.map_ann_file = quote_path(osm_path)
    val_dataset.test_mode = True

    test_dataset = cfg.data.test.copy()
    test_dataset.ann_file = quote_path(infos_path)
    test_dataset.map_ann_file = quote_path(osm_path)
    test_dataset.test_mode = True

    cfg.data.train = train_dataset
    cfg.data.val = val_dataset
    cfg.data.test = test_dataset
    cfg.PHENIKAA_OSM_GT = quote_path(osm_path)
    cfg.PHENIKAA_TRAIN_INFOS = quote_path(infos_path)
    cfg.PHENIKAA_TRAIN_DATASETS = [
        {
            "name": source["name"],
            "infos": quote_path(source["infos"]),
            "gt_osm": quote_path(source["gt_osm"]),
            "repeat": source["repeat"],
        }
        for source in train_sources
    ]
    cfg.PHENIKAA_TRAIN_IMAGE_SCALE = args.train_image_scale
    cfg.PHENIKAA_MAP_CLASSES = map_classes

    out_config.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Auto-generated by benchmark_maptr/07_make_phenikaa_train_config.py\n"
        "# Train MapTR 12 camera voi GT OSM Phenikaa.\n\n"
    )
    out_config.write_text(header + dump_config_dict(cfg._cfg_dict.to_dict()), encoding="utf-8")

    print("[DONE]")
    print(f"Output config : {out_config}")
    print(f"OSM GT        : {osm_path}")
    print(f"Train infos   : {infos_path}")
    print("Train datasets:")
    for source in train_sources:
        print(f"  - {source['name']}: repeat={source['repeat']} infos={source['infos']} osm={source['gt_osm']}")
    print(f"Work dir      : {work_dir}")
    print(f"Load from     : {cfg.load_from}")
    print(f"Max epochs    : {args.max_epochs}")
    print(f"Train scale   : {args.train_image_scale}")
    print(f"Map classes   : {map_classes}")
    print("\nKiem tra train dataset bang:")
    print(f"python3 {Path(__file__).with_name('09_check_train_dataset_osm.py')} --config {out_config}")


if __name__ == "__main__":
    main()
