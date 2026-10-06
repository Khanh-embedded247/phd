"""
Shared paths for Phenikaa tutorials (calib / projection demos).

Owned by the Phenikaa project — not part of upstream MapTR.
"""

from __future__ import annotations

import re
from pathlib import Path

# phenikaa/tutorials/ → phenikaa/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINI_DATASET_ROOT = PROJECT_ROOT / "data" / "phenikaa" / "sequences"
CALIB_ROOT = PROJECT_ROOT / "data" / "phenikaa" / "calib"
# CALIB_ROOT = PROJECT_ROOT / "calib" / "vf_06_06"
RESULT_ROOT = PROJECT_ROOT / "outputs" / "phenikaa_vis" / "tutorials"
# RESULT_ROOT = PROJECT_ROOT / "outputs" / "Normal"

# Raw VF6_02 validation data uses the original folder layout, not sequences/.
RAW_VF_06_02_CALIB_DIR = PROJECT_ROOT / "calib" / "vf_06_02"
RAW_VF_06_02_INTRINSIC_JSON = RAW_VF_06_02_CALIB_DIR / "VF6_02_Intrinsics.json"
RAW_VF_06_02_EXTRINSIC_JSON = RAW_VF_06_02_CALIB_DIR / "VF6_02_Extrinsics_By_Dates.json"
RAW_VF_06_02_IMAGE_ROOT = PROJECT_ROOT / "data" / "Normal" / "CAMERA"
RAW_VF_06_02_LIDAR_ROOT = PROJECT_ROOT / "data" / "Normal" / "LIDAR"
RAW_VF_06_02_RESULT_ROOT = PROJECT_ROOT / "outputs" / "raw_vf_06_02_validation"

# Folder name under sequences/
SEQUENCE_NAME = "RESIDENTIAL_AREA"
# SEQUENCE_NAME = ""

REF_CAMERA = "CAM_F_F"
_LABEL_STAMP_RE = re.compile(r"^(\d{10}-\d{9})\.txt$")


def list_sequence_names(root: Path = MINI_DATASET_ROOT) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def collect_timestamps(
    sequence_folder: Path,
    *,
    require_lidar: bool = True,
    require_image: bool = True,
    ref_camera: str = REF_CAMERA,
) -> list[str]:
    """Sorted timestamps from Label/, optionally requiring matching Lidar/Image."""
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
        stamps.append(ts)

    if not stamps:
        raise ValueError(
            f"No valid timestamps under {sequence_folder} "
            f"(Label + optional Lidar/Image checks)."
        )
    return stamps


def _resolve_sequence_folder() -> Path:
    folder = MINI_DATASET_ROOT / SEQUENCE_NAME
    if not folder.is_dir():
        available = list_sequence_names()
        raise FileNotFoundError(
            f"Sequence not found: {folder}\n"
            f"Available under {MINI_DATASET_ROOT}: {available}"
        )
    return folder


def _resolve_calib() -> tuple[Path, Path]:
    intrinsic = CALIB_ROOT / "Camera_Intrinsics.json"
    extrinsic = CALIB_ROOT / "Sensor_Extrinsics.json"
    missing = [p for p in (intrinsic, extrinsic) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Calibration file(s) missing: {missing}")
    return intrinsic, extrinsic


SEQUENCE_FOLDER = _resolve_sequence_folder()
INTRINSIC_JSON, EXTRINSIC_JSON = _resolve_calib()

TIMESTAMPS = collect_timestamps(SEQUENCE_FOLDER)
TIMESTAMP_FIRST = TIMESTAMPS[0]
TIMESTAMP_LAST = TIMESTAMPS[-1]
# Single-frame tutorials (01, 04, 07, …) use an early frame in the sequence.
TIMESTAMP = TIMESTAMPS[min(10, len(TIMESTAMPS) - 1)]

DEVKIT_ROOT = PROJECT_ROOT

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
