"""
Phenikaa HD Map — canonical paths.

Mọi script trong dự án import từ đây, không hard-code đường dẫn tuyệt đối rải rác.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Project roots
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
PHENIKAA_DATA = DATA_ROOT / "phenikaa"
NUSCENES_DATA = DATA_ROOT / "nuscenes"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
THIRD_PARTY_MAPTR = PROJECT_ROOT / "third_party" / "MapTR"
TUTORIALS_ROOT = PROJECT_ROOT / "tutorials"

# ---------------------------------------------------------------------------
# Phenikaa sensors & sequence
# ---------------------------------------------------------------------------
CALIB_ROOT = PHENIKAA_DATA / "calib"
INTRINSIC_JSON = CALIB_ROOT / "Camera_Intrinsics.json"
EXTRINSIC_JSON = CALIB_ROOT / "Sensor_Extrinsics.json"

SEQUENCE_NAME = "RESIDENTIAL_AREA"
SEQUENCE_FOLDER = PHENIKAA_DATA / "sequences" / SEQUENCE_NAME

MAPS_ROOT = PHENIKAA_DATA / "maps"
# Sau khi verify, đặt/symlink file này:
RESIDENTIAL_MAP_PCD = MAPS_ROOT / "residential.pcd"
MAP_CANDIDATES = [
    MAPS_ROOT / "map_candidate.pcd",
    MAPS_ROOT / "pnkx_candidate.pcd",
]

REF_CAMERA = "CAM_P_F"
CAMERAS = [
    "CAM_P_L",
    "CAM_P_FL",
    "CAM_P_F",
    "CAM_P_FR",
    "CAM_P_R",
    "CAM_P_B",
    "CAM_F_L",
    "CAM_F_F",
    "CAM_F_R",
    "CAM_F_B",
]

_LABEL_STAMP_RE = re.compile(r"^(\d{10}-\d{9})\.txt$")


def collect_timestamps(
    sequence_folder: Path = SEQUENCE_FOLDER,
    *,
    require_lidar: bool = True,
    require_image: bool = True,
    require_pose: bool = True,
    ref_camera: str = REF_CAMERA,
) -> list[str]:
    """Sorted timestamps from Label/, optionally requiring matching modalities."""
    label_dir = sequence_folder / "Label"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label folder not found: {label_dir}")

    stamps: list[str] = []
    for label_file in sorted(label_dir.glob("*.txt")):
        match = _LABEL_STAMP_RE.match(label_file.name)
        if not match:
            continue
        ts = match.group(1)
        if require_lidar and not (sequence_folder / "Lidar" / f"{ts}.laz").exists():
            continue
        if require_image and not (
            sequence_folder / "Image" / ref_camera / f"{ts}.jpg"
        ).exists():
            continue
        if require_pose and not (sequence_folder / "Pose" / f"{ts}.txt").exists():
            continue
        stamps.append(ts)

    if not stamps:
        raise ValueError(f"No valid timestamps under {sequence_folder}")
    return stamps


def require_file(path: Path, hint: str = "") -> Path:
    if not path.is_file():
        msg = f"Missing file: {path}"
        if hint:
            msg += f"\n  hint: {hint}"
        raise FileNotFoundError(msg)
    return path


def require_dir(path: Path, hint: str = "") -> Path:
    if not path.is_dir():
        msg = f"Missing directory: {path}"
        if hint:
            msg += f"\n  hint: {hint}"
        raise FileNotFoundError(msg)
    return path
