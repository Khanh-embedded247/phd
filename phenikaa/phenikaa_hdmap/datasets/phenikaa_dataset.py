"""
Phenikaa Dataset for MapTR inference.

Maps infos_residential.pkl format to MapTR's expected dataset interface.
This implementation returns fixed-shape camera arrays (10 cameras) padded where
necessary so the MapTR model can accept consistent inputs.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import torch
from mmdet.datasets import DATASETS

from phenikaa_hdmap.paths import INTRINSIC_JSON, EXTRINSIC_JSON


def load_image(path: Path) -> np.ndarray:
    """Load image as RGB, resized to 928x1656."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        # Return black image if missing/corrupted
        return np.zeros((928, 1656, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1656, 928), interpolation=cv2.INTER_LINEAR)
    return img


def load_lidar(path: Path) -> np.ndarray:
    """Load LiDAR as [N, 3]. Returns empty array if LAZ decompression fails."""
    try:
        las = laspy.read(str(path))
        return np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    except Exception as e:
        print(f"  WARNING: Could not load LiDAR from {path}: {e}")
        return np.zeros((0, 3), dtype=np.float32)


@DATASETS.register_module()
class PhenikkaaDataset:
    """Phenikaa dataset for MapTR inference (camera + LiDAR).

    Returns a dict with keys compatible with MapTR test pipeline:
      - `sample_idx` (str)
      - `img` (np.ndarray) shape [num_cams, H, W, C]
      - `lidar2img` (np.ndarray) shape [num_cams, 4, 4]
      - `cam_intrinsic` (np.ndarray) shape [num_cams, 4, 4]
      - `pts_filename` (str)
      - `ego2global_translation` (np.array)
      - `ego2global_rotation` (np.array)
      - `can_bus` (np.array)
    """

    CLASSES = ("divider", "ped_crossing", "boundary")

    def __init__(self, infos_path: str, test_mode: bool = False, **kwargs):
        self.infos_path = Path(infos_path)
        self.test_mode = test_mode

        # Load intrinsics/extrinsics JSONs
        with open(INTRINSIC_JSON) as f:
            self.intrinsics = json.load(f)
        with open(EXTRINSIC_JSON) as f:
            self.extrinsics = json.load(f)

        # Load infos
        with open(self.infos_path, "rb") as f:
            data = pickle.load(f)
        self.data_infos = data["infos"]

    def __len__(self) -> int:
        return len(self.data_infos)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get_data_info(index)

    def get_data_info(self, index: int) -> dict[str, Any]:
        info = self.data_infos[index]

        # Fixed camera order (10 cameras)
        cameras = [
            "CAM_P_L", "CAM_P_FL", "CAM_P_F", "CAM_P_FR", "CAM_P_R",
            "CAM_P_B", "CAM_F_L", "CAM_F_F", "CAM_F_R", "CAM_F_B",
        ]

        IMG_H, IMG_W = 928, 1656

        imgs = []
        lidar2img_rts = []
        cam_intrinsics = []

        for cam_name in cameras:
            cam_path = info["cams"].get(cam_name)
            if cam_path and Path(cam_path).exists():
                img = load_image(Path(cam_path))
            else:
                img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            imgs.append(img)

            # Intrinsic
            if cam_name in self.intrinsics:
                try:
                    K = np.array(self.intrinsics[cam_name]["camera_matrix"], dtype=np.float32).reshape(3, 3)
                except Exception:
                    K = np.eye(3, dtype=np.float32)
            else:
                K = np.eye(3, dtype=np.float32)

            # Extrinsic (lidar->cam). JSON may store 4x4 or 3x4 or flat 12
            lidar2cam_rt = np.zeros((3, 4), dtype=np.float32)
            if cam_name in self.extrinsics:
                try:
                    raw = np.array(self.extrinsics[cam_name], dtype=np.float32)
                    if raw.shape == (4, 4):
                        lidar2cam_rt = raw[:3, :]
                    elif raw.shape == (3, 4):
                        lidar2cam_rt = raw
                    elif raw.size == 12:
                        lidar2cam_rt = raw.reshape(3, 4)
                except Exception:
                    pass

            # Build projection matrix
            lidar2cam_4x4 = np.eye(4, dtype=np.float32)
            lidar2cam_4x4[:3, :] = lidar2cam_rt
            K_homo = np.eye(4, dtype=np.float32)
            K_homo[:3, :3] = K
            lidar2img = (K_homo @ lidar2cam_4x4).astype(np.float32)
            lidar2img_rts.append(lidar2img)

            cam_K = np.eye(4, dtype=np.float32)
            cam_K[:3, :3] = K
            cam_intrinsics.append(cam_K.astype(np.float32))

        imgs_array = np.stack(imgs, axis=0).astype(np.float32)  # [num_cams, H, W, C]
        lidar2img_arr = np.stack(lidar2img_rts, axis=0).astype(np.float32)  # [num_cams,4,4]
        cam_intrinsics_arr = np.stack(cam_intrinsics, axis=0).astype(np.float32)

        # LiDAR points (may be empty if LAZ backend missing)
        points = load_lidar(Path(info.get("lidar_path", "")))

        input_dict = {
            "sample_idx": int(index),
            "img": imgs_array,  # [num_cams, H, W, C]
            "lidar2img": lidar2img_arr,
            "cam_intrinsic": cam_intrinsics_arr,
            "pts_filename": info.get("lidar_path", ""),
            "ego2global_translation": np.array(info.get("ego_translation", [0, 0, 0]), dtype=np.float32),
            "ego2global_rotation": np.array(info.get("ego2global_rotation", [0, 0, 0, 1]), dtype=np.float32),
            "can_bus": np.zeros(18, dtype=np.float32),
            "timestamp": float(info.get("timestamp", 0.0)),
        }

        return input_dict

    def prepare_test_data(self, index: int) -> dict:
        return self.get_data_info(index)
