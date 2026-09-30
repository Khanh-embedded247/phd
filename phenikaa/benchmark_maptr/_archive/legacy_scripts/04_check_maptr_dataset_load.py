#!/usr/bin/env python3
"""Kiem tra MapTR co doc duoc infos Phenikaa hay chua.

Buoc nay CHUA train va CHUA inference.

No lam 2 tang kiem tra:
  1. Manual check: doc .pkl, mo anh, tinh lidar2img theo dung cong thuc MapTR.
     Tang nay khong phu thuoc vao mmdet3d/plugin, nen luon nen chay duoc.
  2. Official MapTR dataset check: thu import repo MapTR va build dataset bang config.
     Tang nay co the fail neu moi truong thieu mmdet3d ops/CUDA extension/nuScenes map.
     Neu fail, script se in ly do ro rang de sua tiep.

Chay:
  python3 phd/phenikaa/benchmark_maptr/04_check_maptr_dataset_load.py
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PHENIKAA_ROOT = Path(__file__).resolve().parents[1]
SURVEY_ROOT = PHENIKAA_ROOT.parents[1]
MAPTR_ROOT = SURVEY_ROOT / "MapTR"
MMDET3D_ROOT = MAPTR_ROOT / "mmdetection3d"

DEFAULT_INFOS = (
    PHENIKAA_ROOT
    / "outputs"
    / "benchmark_maptr"
    / "Normal"
    / "phenikaa_maptr_infos_val.pkl"
)
DEFAULT_MAPTR_CONFIG = MAPTR_ROOT / "projects" / "configs" / "maptr" / "maptr_tiny_r50_24e.py"
DEFAULT_NUSCENES_RAW = PHENIKAA_ROOT / "data" / "nuscenes" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiem tra MapTR load infos Phenikaa.")
    parser.add_argument("--infos", type=Path, default=DEFAULT_INFOS)
    parser.add_argument("--maptr-root", type=Path, default=MAPTR_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_MAPTR_CONFIG)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--skip-official",
        action="store_true",
        help="Chi manual check, bo qua build dataset MapTR that.",
    )
    return parser.parse_args()


def load_infos(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "infos" not in data:
        raise ValueError("File infos phai la dict va co key 'infos'")
    return data


def maptr_style_lidar2img(cam_info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tinh lidar2cam/lidar2img y het logic trong nuscenes_map_dataset.py.

    Trong pkl, sensor2lidar la camera -> lidar.
    MapTR dao nguoc no de lay lidar -> camera, sau do nhan voi K.
    """
    sensor2lidar_rotation = np.asarray(cam_info["sensor2lidar_rotation"], dtype=np.float64)
    sensor2lidar_translation = np.asarray(cam_info["sensor2lidar_translation"], dtype=np.float64)

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T

    lidar2cam_rt = np.eye(4, dtype=np.float64)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    lidar2cam = lidar2cam_rt.T

    intrinsic = np.asarray(cam_info["cam_intrinsic"], dtype=np.float64)
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    lidar2img = viewpad @ lidar2cam
    return lidar2cam, viewpad, lidar2img


def read_image_shape(path: Path) -> tuple[int, int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {path}")
    return image.shape


def manual_check(infos_path: Path, sample_index: int) -> dict[str, Any]:
    """Kiem tra truc tiep nhung field MapTR can tu .pkl."""
    data = load_infos(infos_path)
    infos = data["infos"]
    if not infos:
        raise ValueError("infos rong")
    if sample_index < 0 or sample_index >= len(infos):
        raise ValueError(f"sample-index ngoai khoang 0..{len(infos) - 1}")

    info = infos[sample_index]
    cams = info["cams"]
    if not isinstance(cams, dict) or not cams:
        raise ValueError("sample['cams'] phai la dict va khong rong")

    print("[MANUAL CHECK]")
    print(f"infos path       : {infos_path}")
    print(f"metadata         : {data.get('metadata')}")
    print(f"infos len        : {len(infos)}")
    print(f"sample index     : {sample_index}")
    print(f"sample token     : {info['token']}")
    print(f"sample timestamp : {info['timestamp']}")
    print(f"camera count     : {len(cams)}")
    print(f"camera order     : {list(cams.keys())}")

    lidar_path = Path(info["lidar_path"])
    print(f"lidar_path       : {lidar_path}")
    print(f"lidar exists     : {lidar_path.exists()}")
    if not lidar_path.exists():
        raise FileNotFoundError(lidar_path)

    can_bus = np.asarray(info["can_bus"])
    print(f"can_bus shape    : {can_bus.shape}")
    if can_bus.shape != (18,):
        raise ValueError(f"can_bus phai co shape (18,), hien tai {can_bus.shape}")

    image_shapes: list[tuple[int, int, int]] = []
    lidar2imgs: list[np.ndarray] = []
    cam_intrinsics: list[np.ndarray] = []
    lidar2cams: list[np.ndarray] = []

    for cam_name, cam_info in cams.items():
        image_path = Path(cam_info["data_path"])
        shape = read_image_shape(image_path)
        image_shapes.append(shape)

        K = np.asarray(cam_info["cam_intrinsic"])
        if K.shape != (3, 3):
            raise ValueError(f"{cam_name}.cam_intrinsic phai la 3x3, hien tai {K.shape}")
        R = np.asarray(cam_info["sensor2lidar_rotation"])
        t = np.asarray(cam_info["sensor2lidar_translation"])
        if R.shape != (3, 3):
            raise ValueError(f"{cam_name}.sensor2lidar_rotation phai la 3x3, hien tai {R.shape}")
        if t.shape != (3,):
            raise ValueError(f"{cam_name}.sensor2lidar_translation phai la (3,), hien tai {t.shape}")

        lidar2cam, K4, lidar2img = maptr_style_lidar2img(cam_info)
        lidar2cams.append(lidar2cam)
        cam_intrinsics.append(K4)
        lidar2imgs.append(lidar2img)

    print(f"image shapes     : {sorted(set(image_shapes))}")
    print(f"lidar2img count  : {len(lidar2imgs)}")
    print(f"lidar2img shape  : {lidar2imgs[0].shape}")
    print(f"cam_intrinsic4x4 : {cam_intrinsics[0].shape}")
    print(f"lidar2cam shape  : {lidar2cams[0].shape}")

    if len(lidar2imgs) != len(cams):
        raise ValueError("So lidar2img khong bang so camera")
    if any(mat.shape != (4, 4) for mat in lidar2imgs):
        raise ValueError("Tat ca lidar2img phai la 4x4")

    phenikaa = info.get("phenikaa", {})
    if phenikaa:
        print(f"sync method      : {phenikaa.get('sync_method')}")
        print(f"source lidar ts  : {phenikaa.get('source_lidar_timestamp_sec')}")
        print(f"synced lidar ts  : {phenikaa.get('synced_lidar_timestamp_sec')}")
        print(f"lidar dt sec     : {phenikaa.get('lidar_dt_sec')}")

    print("Manual check OK: .pkl co du du lieu de MapTR tinh image/lidar geometry.")
    return {
        "num_infos": len(infos),
        "num_cams": len(cams),
        "image_shapes": image_shapes,
        "lidar2img_count": len(lidar2imgs),
    }


def add_maptr_paths(maptr_root: Path) -> None:
    """Them MapTR va mmdetection3d local vao sys.path."""
    mmdet3d_root = maptr_root / "mmdetection3d"
    for path in (str(mmdet3d_root), str(maptr_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def make_matplotlib_cache() -> None:
    """Tranh loi matplotlib/fontconfig do home cache khong writable."""
    cache_dir = Path("/tmp") / "phenikaa_maptr_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def patch_numpy_legacy_aliases() -> None:
    """MapTR/MMDetection3D cu doi khi con dung alias cu cua NumPy.

    NumPy >= 1.24 da xoa np.long/np.int/np.float. Ta gan lai truoc khi
    import plugin de code cu chay duoc ma khong can downgrade moi truong.
    """
    legacy_aliases = {
        "long": int,
        "int": int,
        "float": float,
        "bool": bool,
    }
    for name, value in legacy_aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def patch_config_for_phenikaa(cfg: Any, infos_path: Path) -> Any:
    """Sua cfg test/val de tro vao pkl Phenikaa va chay test_mode."""
    dataset_cfg = cfg.data.test.copy()
    dataset_cfg.ann_file = str(infos_path)
    # CustomNuScenesLocalMapDataset luon khoi tao NuScenesMap trong __init__.
    # Du lieu Phenikaa dung absolute path trong pkl, nen data_root co the tro
    # toi nuScenes mini maps chi de vuot qua init khi USE_GT=False.
    dataset_cfg.data_root = str(DEFAULT_NUSCENES_RAW if DEFAULT_NUSCENES_RAW.exists() else PHENIKAA_ROOT)
    dataset_cfg.test_mode = True
    dataset_cfg.pop("samples_per_gpu", None)
    dataset_cfg.modality = dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=True,
    )
    # Buoc nay chua eval GT, nen map_ann_file khong quan trong.
    dataset_cfg.map_ann_file = None
    return dataset_cfg


def official_maptr_dataset_check(maptr_root: Path, config_path: Path, infos_path: Path) -> bool:
    """Thu build dataset bang stack MapTR that."""
    print("\n[OFFICIAL MAPTR DATASET CHECK]")
    print(f"maptr root : {maptr_root}")
    print(f"config     : {config_path}")

    make_matplotlib_cache()
    patch_numpy_legacy_aliases()
    add_maptr_paths(maptr_root)

    try:
        import mmcv
        from mmcv import Config
        from mmdet.datasets import build_dataset

        print(f"mmcv       : {mmcv.__version__}")

        # Chi import dataset/pipeline de register vao registry.
        # Khong import full projects.mmdet3d_plugin vi no keo ca model/BEVFormer/NMS,
        # trong khi buoc 04 chi can kiem data loader.
        import projects.mmdet3d_plugin.datasets  # noqa: F401
        import projects.mmdet3d_plugin.datasets.pipelines  # noqa: F401

        cfg = Config.fromfile(str(config_path))
        dataset_cfg = patch_config_for_phenikaa(cfg, infos_path)
        dataset = build_dataset(dataset_cfg)
        print(f"dataset type : {type(dataset).__name__}")
        print(f"dataset len  : {len(dataset)}")

        item = dataset[0]
        print(f"item type    : {type(item).__name__}")
        print(f"item keys    : {list(item.keys())}")
        if "img" in item:
            img = item["img"]
            print(f"img object   : {type(img).__name__}")
            data = getattr(img, "_data", img)
            if hasattr(data, "shape"):
                print(f"img shape    : {tuple(data.shape)}")
            elif isinstance(data, list):
                print(f"img list len : {len(data)}")
        if "img_metas" in item:
            metas = getattr(item["img_metas"], "_data", item["img_metas"])
            print(f"img_metas    : {type(metas).__name__}")
        print("Official MapTR dataset check OK.")
        return True
    except Exception as exc:
        print("Official MapTR dataset check FAILED.")
        print(f"error type : {type(exc).__name__}")
        print(f"error      : {exc}")
        print("\nTraceback ngan:")
        tb_lines = traceback.format_exc(limit=6).strip().splitlines()
        for line in tb_lines:
            print(line)
        print(
            "\nGoi y: manual check da du de xac nhan pkl/anh/calib dung format. "
            "Loi official thuong do mmdet3d ops chua build, plugin import model/ops, "
            "hoac dataset nuScenes map API chua phu hop Phenikaa."
        )
        return False


def main() -> None:
    args = parse_args()
    infos_path = args.infos.expanduser().resolve()
    maptr_root = args.maptr_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()

    manual_check(infos_path, args.sample_index)

    if args.skip_official:
        print("\nSkip official MapTR dataset check theo yeu cau.")
        return

    official_maptr_dataset_check(maptr_root, config_path, infos_path)


if __name__ == "__main__":
    main()
