#!/usr/bin/env python3
"""Kiem tra cau truc file infos .pkl theo dinh dang MapTR/nuScenes.

File nay chi doc va in thong tin, khong sua du lieu.
Muc dich:
  1. Soi file .pkl goc cua MapTR de biet chinh xac schema.
  2. Kiem tra file .pkl Phenikaa sau nay co giong dinh dang do khong.
  3. Xac nhan K camera trong .pkl la 3x3, con MapTR tu pad len 4x4 khi chay.

Chay mac dinh:
  python3 phd/phenikaa/benchmark_maptr/02_check_infos.py

Chay voi file khac:
  python3 phd/phenikaa/benchmark_maptr/02_check_infos.py --pkl /path/to/infos.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np


PHENIKAA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PKL = (
    PHENIKAA_ROOT
    / "data"
    / "nuscenes"
    / "raw"
    / "nuscenes_map_infos_temporal_val.pkl"
)


REQUIRED_TOP_KEYS = ("infos", "metadata")

REQUIRED_SAMPLE_KEYS = (
    "cams",
    "can_bus",
    "ego2global_rotation",
    "ego2global_translation",
    "frame_idx",
    "lidar2ego_rotation",
    "lidar2ego_translation",
    "lidar_path",
    "map_location",
    "next",
    "prev",
    "scene_token",
    "sweeps",
    "timestamp",
    "token",
)

REQUIRED_CAMERA_KEYS = (
    "cam_intrinsic",
    "data_path",
    "ego2global_rotation",
    "ego2global_translation",
    "sample_data_token",
    "sensor2ego_rotation",
    "sensor2ego_translation",
    "sensor2lidar_rotation",
    "sensor2lidar_translation",
    "timestamp",
    "type",
)


def describe_value(value: Any) -> str:
    """Tra ve mo ta ngan gon de in schema de doc."""
    if isinstance(value, np.ndarray):
        return f"ndarray shape={value.shape} dtype={value.dtype}"
    if isinstance(value, dict):
        keys = list(value.keys())
        return f"dict len={len(value)} keys={keys[:8]}"
    if isinstance(value, list):
        first_type = type(value[0]).__name__ if value else "None"
        return f"list len={len(value)} first_type={first_type}"
    return f"{type(value).__name__} value={repr(value)[:100]}"


def check_shape(name: str, value: Any, expected_shape: tuple[int, ...]) -> list[str]:
    """Kiem tra shape ndarray va tra ve danh sach loi."""
    errors: list[str] = []
    if not isinstance(value, np.ndarray):
        errors.append(f"{name}: phai la np.ndarray, hien tai la {type(value).__name__}")
        return errors
    if value.shape != expected_shape:
        errors.append(f"{name}: shape phai la {expected_shape}, hien tai la {value.shape}")
    return errors


def check_list_len(name: str, value: Any, expected_len: int) -> list[str]:
    """Kiem tra list/tuple co do dai mong muon."""
    errors: list[str] = []
    if not isinstance(value, (list, tuple)):
        errors.append(f"{name}: phai la list/tuple, hien tai la {type(value).__name__}")
        return errors
    if len(value) != expected_len:
        errors.append(f"{name}: do dai phai la {expected_len}, hien tai la {len(value)}")
    return errors


def check_required_keys(prefix: str, obj: dict[str, Any], required: tuple[str, ...]) -> list[str]:
    """Kiem tra cac key bat buoc co ton tai khong."""
    errors: list[str] = []
    for key in required:
        if key not in obj:
            errors.append(f"{prefix}: thieu key '{key}'")
    return errors


def inspect_annotation(info: dict[str, Any]) -> None:
    """In cau truc annotation neu file .pkl co san GT vector map."""
    ann = info.get("annotation")
    print("\n[ANNOTATION / GT VECTOR MAP]")
    if ann is None:
        print("Khong co key 'annotation'. Neu chi chay inference thi co the chap nhan.")
        return
    if not isinstance(ann, dict):
        print(f"annotation khong phai dict: {type(ann).__name__}")
        return

    print(f"annotation keys: {list(ann.keys())}")
    for cls_name, vectors in ann.items():
        if not isinstance(vectors, list):
            print(f"  {cls_name}: {type(vectors).__name__}, khong phai list")
            continue
        first_shape = None
        if vectors and isinstance(vectors[0], np.ndarray):
            first_shape = vectors[0].shape
        print(f"  {cls_name}: so vector={len(vectors)}, first_shape={first_shape}")


def inspect_one_sample(info: dict[str, Any], sample_index: int) -> list[str]:
    """In schema cua mot sample va tra ve cac loi phat hien."""
    errors: list[str] = []
    print(f"\n[SAMPLE {sample_index}]")
    print("sample keys:")
    for key in sorted(info.keys()):
        print(f"  {key}: {describe_value(info[key])}")

    errors.extend(check_required_keys("sample", info, REQUIRED_SAMPLE_KEYS))
    if errors:
        return errors

    errors.extend(check_list_len("ego2global_rotation", info["ego2global_rotation"], 4))
    errors.extend(check_list_len("ego2global_translation", info["ego2global_translation"], 3))
    errors.extend(check_list_len("lidar2ego_rotation", info["lidar2ego_rotation"], 4))
    errors.extend(check_list_len("lidar2ego_translation", info["lidar2ego_translation"], 3))
    errors.extend(check_shape("can_bus", info["can_bus"], (18,)))

    cams = info["cams"]
    if not isinstance(cams, dict):
        errors.append(f"cams: phai la dict, hien tai la {type(cams).__name__}")
        return errors
    if not cams:
        errors.append("cams: dict rong")
        return errors

    print("\n[CAMERAS]")
    print(f"camera count: {len(cams)}")
    print(f"camera order: {list(cams.keys())}")

    first_cam_name = next(iter(cams))
    first_cam = cams[first_cam_name]
    print(f"\n[first camera: {first_cam_name}]")
    for key in sorted(first_cam.keys()):
        print(f"  {key}: {describe_value(first_cam[key])}")

    for cam_name, cam_info in cams.items():
        if not isinstance(cam_info, dict):
            errors.append(f"cams[{cam_name}]: phai la dict, hien tai la {type(cam_info).__name__}")
            continue
        errors.extend(check_required_keys(f"cams[{cam_name}]", cam_info, REQUIRED_CAMERA_KEYS))
        if "cam_intrinsic" in cam_info:
            errors.extend(check_shape(f"cams[{cam_name}].cam_intrinsic", cam_info["cam_intrinsic"], (3, 3)))
        if "sensor2lidar_rotation" in cam_info:
            errors.extend(
                check_shape(
                    f"cams[{cam_name}].sensor2lidar_rotation",
                    cam_info["sensor2lidar_rotation"],
                    (3, 3),
                )
            )
        if "sensor2lidar_translation" in cam_info:
            errors.extend(
                check_shape(
                    f"cams[{cam_name}].sensor2lidar_translation",
                    cam_info["sensor2lidar_translation"],
                    (3,),
                )
            )
        if "sensor2ego_rotation" in cam_info:
            errors.extend(check_list_len(f"cams[{cam_name}].sensor2ego_rotation", cam_info["sensor2ego_rotation"], 4))
        if "sensor2ego_translation" in cam_info:
            errors.extend(check_list_len(f"cams[{cam_name}].sensor2ego_translation", cam_info["sensor2ego_translation"], 3))

    inspect_annotation(info)
    return errors


def load_infos(pkl_path: Path) -> dict[str, Any]:
    """Load file pkl bang pickle vi file MapTR la Python pickle."""
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Top-level phai la dict, hien tai la {type(data).__name__}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiem tra schema file infos .pkl cho MapTR.")
    parser.add_argument(
        "--pkl",
        type=Path,
        default=DEFAULT_PKL,
        help=f"Duong dan infos .pkl. Mac dinh: {DEFAULT_PKL}",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index sample can in chi tiet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pkl_path = args.pkl.expanduser().resolve()

    print("[INPUT]")
    print(f"pkl_path: {pkl_path}")
    if not pkl_path.exists():
        raise FileNotFoundError(f"Khong thay file: {pkl_path}")

    data = load_infos(pkl_path)

    print("\n[TOP LEVEL]")
    print(f"type: {type(data).__name__}")
    print(f"keys: {list(data.keys())}")

    errors: list[str] = []
    errors.extend(check_required_keys("top-level", data, REQUIRED_TOP_KEYS))
    if errors:
        print("\n[ERRORS]")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    infos = data["infos"]
    metadata = data["metadata"]
    print(f"metadata: {metadata}")
    print(f"infos type: {type(infos).__name__}")
    print(f"infos len: {len(infos)}")

    if not isinstance(infos, list) or not infos:
        raise SystemExit("infos phai la list va khong duoc rong")
    if args.sample_index < 0 or args.sample_index >= len(infos):
        raise SystemExit(f"sample-index nam ngoai khoang: 0..{len(infos) - 1}")

    sample = infos[args.sample_index]
    if not isinstance(sample, dict):
        raise SystemExit(f"sample phai la dict, hien tai la {type(sample).__name__}")

    errors.extend(inspect_one_sample(sample, args.sample_index))

    print("\n[SUMMARY]")
    if errors:
        print(f"FAILED: co {len(errors)} loi schema")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("OK: schema sample hop le theo cac field chinh MapTR dang doc.")
    print("Ghi nho: cam_intrinsic trong .pkl la 3x3; MapTR se tu pad len 4x4 khi tao lidar2img.")


if __name__ == "__main__":
    main()
