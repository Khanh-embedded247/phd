"""Phenikaa Pose helpers (3x4 map ← ego/sensor)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_pose_3x4(path: Path) -> np.ndarray:
    """Read a Pose/*.txt file as 3x4 matrix."""
    vals = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if vals.size != 12:
        raise ValueError(f"Expected 12 numbers in pose file, got {vals.size}: {path}")
    return vals.reshape(3, 4)


def pose_to_4x4(pose_3x4: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :] = pose_3x4
    return T


def translation(pose_3x4: np.ndarray) -> np.ndarray:
    return pose_3x4[:, 3].copy()


def invert_pose_4x4(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv
