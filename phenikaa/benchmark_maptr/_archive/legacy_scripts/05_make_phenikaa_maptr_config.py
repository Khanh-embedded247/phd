#!/usr/bin/env python3
"""Tao config MapTR rieng cho Phenikaa 12 camera.

Buoc nay khong train va khong inference.
No chi tao file config Python de cac buoc sau dung lai.

Config sinh ra se:
  - Ke thua config goc maptr_tiny_r50_24e.py.
  - Doi num_cams tu mac dinh 6 sang 12.
  - Tro ann_file toi phenikaa_maptr_infos_val.pkl.
  - Dat test_mode=True va USE_GT=False.
  - Dat samples_per_gpu=1 de smoke test de kiem soat.

Chay:
  python3 phd/phenikaa/benchmark_maptr/05_make_phenikaa_maptr_config.py
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
MAPTR_ROOT = SURVEY_ROOT / "MapTR"

DEFAULT_BASE_CONFIG = MAPTR_ROOT / "projects" / "configs" / "maptr" / "maptr_tiny_r50_24e.py"
DEFAULT_INFOS = (
    PHENIKAA_ROOT
    / "outputs"
    / "benchmark_maptr"
    / "Normal"
    / "phenikaa_maptr_infos_val.pkl"
)
DEFAULT_NUSCENES_RAW = PHENIKAA_ROOT / "data" / "nuscenes" / "raw"
DEFAULT_OUTPUT_CONFIG = (
    PHENIKAA_ROOT
    / "benchmark_maptr"
    / "config"
    / "maptr_tiny_r50_phenikaa_12cam.py"
)


CAMERA_ORDER = [
    "CAM_P_F",
    "CAM_P_FL",
    "CAM_P_FR",
    "CAM_P_B",
    "CAM_P_L",
    "CAM_P_R",
    "CAM_P_LB",
    "CAM_P_RB",
    "CAM_F_F",
    "CAM_F_L",
    "CAM_F_R",
    "CAM_F_B",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tao config MapTR Phenikaa 12 camera.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--infos", type=Path, default=DEFAULT_INFOS)
    parser.add_argument("--out-config", type=Path, default=DEFAULT_OUTPUT_CONFIG)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_NUSCENES_RAW,
        help=(
            "data_root chi dung de CustomNuScenesLocalMapDataset init NuScenesMap. "
            "Anh/LiDAR Phenikaa trong pkl dang la absolute path."
        ),
    )
    parser.add_argument("--num-cams", type=int, default=len(CAMERA_ORDER))
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    return parser.parse_args()


def quote_path(path: Path) -> str:
    """Tao string path an toan trong file config."""
    return str(path.expanduser().resolve()).replace("\\", "\\\\").replace("'", "\\'")


def patch_numpy_legacy_aliases() -> None:
    """MapTR/MMCV cu doi khi con dung alias NumPy da bi xoa."""
    for name, value in {
        "long": int,
        "int": int,
        "float": float,
        "bool": bool,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def make_matplotlib_cache() -> None:
    """Tranh loi cache khi import config/plugin trong moi truong sandbox."""
    cache_dir = Path("/tmp") / "phenikaa_maptr_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def add_maptr_paths() -> None:
    """Them MapTR local va mmdetection3d local vao sys.path."""
    for path in (MAPTR_ROOT / "mmdetection3d", MAPTR_ROOT):
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)


def make_config_text_from_base(
    base_config: Path,
    infos_path: Path,
    data_root: Path,
    num_cams: int,
    workers_per_gpu: int,
) -> str:
    """Load config goc, sua truc tiep dict, roi xuat thanh config day du."""
    make_matplotlib_cache()
    patch_numpy_legacy_aliases()
    add_maptr_paths()

    from mmcv import Config

    cfg = Config.fromfile(str(base_config))

    cfg.model.pts_bbox_head.transformer.num_cams = num_cams
    cfg.model.pts_bbox_head.transformer.encoder.transformerlayers.attn_cfgs[1].num_cams = num_cams

    cfg.input_modality = dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=True,
    )

    common_dataset = dict(
        type=cfg.dataset_type,
        data_root=f"{quote_path(data_root)}/",
        ann_file=quote_path(infos_path),
        map_ann_file=None,
        pipeline=cfg.test_pipeline,
        bev_size=(cfg.bev_h_, cfg.bev_w_),
        pc_range=cfg.point_cloud_range,
        fixed_ptsnum_per_line=cfg.fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=cfg.eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=cfg.map_classes,
        classes=cfg.class_names,
        modality=cfg.input_modality,
        test_mode=True,
    )

    cfg.data = dict(
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
        train=common_dataset.copy(),
        val=common_dataset.copy(),
        test=common_dataset.copy(),
        shuffler_sampler=dict(type="DistributedGroupSampler"),
        nonshuffler_sampler=dict(type="DistributedSampler"),
    )
    cfg.evaluation = dict(interval=1, pipeline=cfg.test_pipeline, metric=None)
    cfg.PHENIKAA_USE_GT = False
    cfg.PHENIKAA_NUM_CAMS = num_cams
    cfg.PHENIKAA_CAMERA_ORDER = CAMERA_ORDER
    cfg.PHENIKAA_INFOS = quote_path(infos_path)
    cfg.plugin = True
    cfg.plugin_dir = "projects/mmdet3d_plugin/"

    cfg_dict = cfg._cfg_dict.to_dict()
    # Config.fromfile da merge _base_, nen config xuat ra la mot file doc lap.
    cfg_dict.pop("_base_", None)
    body_lines = []
    for key, value in cfg_dict.items():
        body_lines.append(f"{key} = {pprint.pformat(value, width=100, sort_dicts=False)}")
    text = "\n\n".join(body_lines) + "\n"
    infos_s = quote_path(infos_path)
    data_root_s = quote_path(data_root)
    camera_order_repr = repr(CAMERA_ORDER)

    header = f'''# Auto-generated by phd/phenikaa/benchmark_maptr/05_make_phenikaa_maptr_config.py
# Config MapTR cho Phenikaa VF6_02, 12 camera, USE_GT=False.
#
# Chay tu MapTR root de plugin_dir='projects/mmdet3d_plugin/' dung:
#   cd {quote_path(MAPTR_ROOT)}
#
# File nay chi phuc vu smoke test / inference input pipeline.
# Khi co GT lanelet/vector map, can tao config khac bat USE_GT=True va map_ann_file.

PHENIKAA_USE_GT = False
PHENIKAA_NUM_CAMS = {num_cams}
PHENIKAA_CAMERA_ORDER = {camera_order_repr}
PHENIKAA_INFOS = '{infos_s}'

# data_root nay dung cho NuScenesMap init trong dataset goc.
# Duong dan anh/LiDAR that da nam absolute trong PHENIKAA_INFOS.
data_root = '{data_root_s}/'

'''
    return header + "\n" + text


def main() -> None:
    args = parse_args()
    base_config = args.base_config.expanduser().resolve()
    infos_path = args.infos.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    out_config = args.out_config.expanduser().resolve()

    if not base_config.exists():
        raise FileNotFoundError(base_config)
    if not infos_path.exists():
        raise FileNotFoundError(infos_path)
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    out_config.parent.mkdir(parents=True, exist_ok=True)
    text = make_config_text_from_base(
        base_config=base_config,
        infos_path=infos_path,
        data_root=data_root,
        num_cams=args.num_cams,
        workers_per_gpu=args.workers_per_gpu,
    )
    out_config.write_text(text, encoding="utf-8")

    print("[DONE]")
    print(f"Output config : {out_config}")
    print(f"Base config   : {base_config}")
    print(f"Infos         : {infos_path}")
    print(f"Data root     : {data_root}")
    print(f"Num cameras   : {args.num_cams}")
    print(f"Camera order  : {CAMERA_ORDER}")
    print("\nKiem tra config bang buoc 04:")
    print(
        "python3 "
        f"{Path(__file__).with_name('04_check_maptr_dataset_load.py')} "
        f"--config {out_config}"
    )


if __name__ == "__main__":
    main()
