"""Load Phenikaa camera intrinsics / extrinsics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from phenikaa_hdmap.paths import EXTRINSIC_JSON, INTRINSIC_JSON


def load_intrinsics(path: Path = INTRINSIC_JSON) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_extrinsics(path: Path = EXTRINSIC_JSON) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def get_camera_K_D(intrinsics: dict, camera: str) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return K (3x3), D, is_fisheye."""
    cam = intrinsics[camera]
    K = np.array(cam["camera_matrix"], dtype=np.float64).reshape(3, 3)
    D = np.array(cam["distortion_coefficients"], dtype=np.float64)
    is_fisheye = camera.startswith("CAM_F")
    if is_fisheye and len(D) > 4:
        D = D[:4]
    return K, D, is_fisheye


def get_lidar_to_camera_Rt(extrinsics: dict, camera: str) -> np.ndarray:
    """3x4 LiDAR → camera transform."""
    return np.array(extrinsics[camera], dtype=np.float64)[:3, :]
