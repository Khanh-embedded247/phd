#!/usr/bin/env python3
"""
raw_vf_06_02_projection.py

Project raw VF6_02 LiDAR points onto a raw CAM_P_F image.

RAW VF6_02 EXTRINSIC CONVENTION
================================
For this raw VF6_02 dump, visual validation shows the LiDAR matrix should be
used directly in the LiDAR -> base part of the projection chain:

    T_lidar_to_cam = T_base_to_cam @ T_lidar_to_base

Projection policy:
    Use calibration distortion D to undistort the image first, then project
    LiDAR points with the undistorted K and zero distortion.
    CAM_F_* uses OpenCV fisheye undistortion; other cameras use normal
    OpenCV pinhole undistortion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import laspy
import numpy as np

from phenikaa_paths import (
    PROJECT_ROOT,
    RAW_VF_06_02_CALIB_DIR,
    RAW_VF_06_02_EXTRINSIC_JSON,
    RAW_VF_06_02_IMAGE_ROOT,
    RAW_VF_06_02_INTRINSIC_JSON,
    RAW_VF_06_02_LIDAR_ROOT,
    RAW_VF_06_02_RESULT_ROOT,
)


# ============================================================
# PATHS / DEFAULTS
# ============================================================

ROOT = PROJECT_ROOT
CALIB_DIR = RAW_VF_06_02_CALIB_DIR
INTRINSIC_JSON = RAW_VF_06_02_INTRINSIC_JSON
EXTRINSIC_JSON = RAW_VF_06_02_EXTRINSIC_JSON
IMAGE_ROOT = RAW_VF_06_02_IMAGE_ROOT
LIDAR_ROOT = RAW_VF_06_02_LIDAR_ROOT
OUTPUT_ROOT = RAW_VF_06_02_RESULT_ROOT

CAM_NAME = "CAM_P_F"
LIDAR_NAME = "LIDAR_TOP"
DEFAULT_IMAGE = IMAGE_ROOT / "1781509259-100421069.jpg"
DEFAULT_LIDAR = LIDAR_ROOT / "1781509259-100004911.laz"

SYNC_WARNING_MS = 50.0
UNDISTORT_BALANCE = 1.0
MAX_DRAW_POINTS = 120_000


# ============================================================
# UTILITIES
# ============================================================

def timestamp_from_name(filename: str) -> float:
    """Parse SEC-NANOSEC.xxx into floating-point seconds."""
    stem = Path(filename).stem
    sec_text, ns_text = stem.split("-", 1)
    return int(sec_text) + int(ns_text) / (10 ** len(ns_text))


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_sensor_table(obj, camera_name: str, lidar_name: str):
    """
    Recursively find the extrinsic block containing both requested sensors.

    This keeps compatibility with an Extrinsics_By_Dates JSON in which
    the sensor table may be nested under a date/calibration-period object.
    """
    if isinstance(obj, dict):
        if camera_name in obj and lidar_name in obj:
            return obj

        for value in obj.values():
            try:
                return find_sensor_table(value, camera_name, lidar_name)
            except KeyError:
                pass

    elif isinstance(obj, list):
        for value in obj:
            try:
                return find_sensor_table(value, camera_name, lidar_name)
            except KeyError:
                pass

    raise KeyError(
        f"Could not find an extrinsic block containing "
        f"{camera_name} and {lidar_name}"
    )


def get_base_to_sensor(extrinsics, sensor_name: str) -> np.ndarray:
    """
    Read a 4x4 sensor extrinsic matrix from JSON.

    Raw VF6_02 naming/convention is ambiguous. The caller decides whether this
    matrix is used directly or inverted for the LiDAR side of the chain.
    """
    if sensor_name not in extrinsics:
        raise KeyError(f"{sensor_name} not found in extrinsics")

    T = np.asarray(extrinsics[sensor_name], dtype=np.float64)

    if T.shape != (4, 4):
        raise ValueError(
            f"{sensor_name}: expected 4x4 matrix, got {T.shape}"
        )

    return T


def nearest_file(reference: Path, candidates: list[Path]):
    """Find candidate having the nearest filename timestamp."""
    t_ref = timestamp_from_name(reference.name)

    choice = min(
        candidates,
        key=lambda p: abs(timestamp_from_name(p.name) - t_ref),
    )

    dt = abs(timestamp_from_name(choice.name) - t_ref)
    return choice, dt


def resolve_sensor_folder(root: Path, sensor_name: str) -> Path:
    """
    Resolve either flat Normal/CAMERA/*.jpg layout or sensor subfolder layout.
    """
    sensor_dir = root / sensor_name
    if sensor_dir.is_dir():
        return sensor_dir
    return root


def print_matrix(name: str, matrix: np.ndarray):
    """Pretty-print a calibration/transform matrix."""
    print(f"\n{name} =")
    print(
        np.array2string(
            matrix,
            precision=9,
            suppress_small=False,
            floatmode="fixed",
        )
    )


def scale_camera_matrix(
    K: np.ndarray,
    calib_width: int,
    calib_height: int,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Scale intrinsics when calibration and image resolutions differ."""
    if calib_width <= 0 or calib_height <= 0:
        raise ValueError(
            f"Invalid calibration image size: {calib_width} x {calib_height}"
        )

    sx = image_width / calib_width
    sy = image_height / calib_height

    K_scaled = K.copy()
    K_scaled[0, :] *= sx
    K_scaled[1, :] *= sy

    return K_scaled


def prepare_projection_image(
    image: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    mode: str,
    *,
    is_fisheye: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select projection target.

    raw:
        Keep the image unchanged and project with OpenCV distortion D.
    rectified:
        Keep the image unchanged and project with zero distortion.
    undistorted:
        Rectify the image first, then project into that rectified image with
        K_new and zero distortion.
    """
    if mode == "raw":
        return image, K, D

    if mode == "rectified":
        return image, K, np.zeros(5, dtype=np.float64)

    if mode != "undistorted":
        raise ValueError(f"Unknown projection image mode: {mode}")

    h, w = image.shape[:2]
    if is_fisheye:
        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            D,
            (w, h),
            np.eye(3),
            balance=UNDISTORT_BALANCE,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K,
            D,
            np.eye(3),
            K_new,
            (w, h),
            cv2.CV_16SC2,
        )
        image_undistorted = cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
        )
        D_zero = np.zeros(5, dtype=np.float64)
        return image_undistorted, K_new, D_zero
    else:
        K_new, _ = cv2.getOptimalNewCameraMatrix(
            K,
            D,
            (w, h),
            UNDISTORT_BALANCE,
            (w, h),
        )
        image_undistorted = cv2.undistort(image, K, D, None, K_new)

    D_zero = np.zeros(5, dtype=np.float64)
    return image_undistorted, K_new, D_zero


def project_points(
    points_lidar: np.ndarray,
    T_lidar_to_cam: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    use_fisheye: bool,
) -> tuple[np.ndarray, np.ndarray]:
    points_h = np.column_stack((points_lidar, np.ones(len(points_lidar))))
    points_cam = (T_lidar_to_cam[:3, :] @ points_h.T).T
    front = np.isfinite(points_cam).all(axis=1) & (points_cam[:, 2] > 0.1)
    points_cam = points_cam[front]

    depths = points_cam[:, 2].copy()
    obj_pts = points_cam.reshape(-1, 1, 3).astype(np.float32)
    rvec = np.zeros(3, dtype=np.float32)
    tvec = np.zeros(3, dtype=np.float32)

    if use_fisheye:
        uv, _ = cv2.fisheye.projectPoints(
            obj_pts,
            rvec,
            tvec,
            K.astype(np.float32),
            D.astype(np.float32),
        )
    else:
        uv, _ = cv2.projectPoints(
            obj_pts,
            rvec,
            tvec,
            K.astype(np.float32),
            D.astype(np.float32),
        )

    return uv.reshape(-1, 2), depths


def draw_depth_overlay(
    image: np.ndarray,
    uv: np.ndarray,
    depths: np.ndarray,
) -> tuple[np.ndarray, int]:
    h, w = image.shape[:2]
    valid = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    uv = uv[valid]
    depths = depths[valid]
    projected_count = len(uv)

    if len(uv) > MAX_DRAW_POINTS:
        ids = np.linspace(0, len(uv) - 1, MAX_DRAW_POINTS, dtype=np.int64)
        uv = uv[ids]
        depths = depths[ids]

    if len(uv) == 0:
        return image.copy(), projected_count

    lo, hi = np.percentile(depths, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    depth_norm = np.clip((depths - lo) / (hi - lo), 0.0, 1.0)
    colors = cv2.applyColorMap(
        (255.0 * (1.0 - depth_norm)).astype(np.uint8).reshape(-1, 1),
        cv2.COLORMAP_TURBO,
    ).reshape(-1, 3)

    out = image.copy()
    uv_i = np.round(uv).astype(np.int32)
    for (u, v), color in zip(uv_i, colors):
        cv2.circle(out, (int(u), int(v)), 1, tuple(int(c) for c in color), -1)

    return out, projected_count


# ============================================================
# CORE
# ============================================================

def project_lidar_to_camera(
    img_path: Path,
    lidar_path: Path,
    camera_name: str,
    lidar_name: str,
    extrinsic_convention: str,
    projection_image: str,
):
    intrinsics = load_json(INTRINSIC_JSON)
    extrinsics_raw = load_json(EXTRINSIC_JSON)
    extrinsics = find_sensor_table(
        extrinsics_raw,
        camera_name,
        lidar_name,
    )

    if camera_name not in intrinsics:
        raise KeyError(
            f"{camera_name} not found in {INTRINSIC_JSON}"
        )

    camera_calib = intrinsics[camera_name]

    K_calib = np.asarray(
        camera_calib["camera_matrix"],
        dtype=np.float64,
    )

    D = np.asarray(
        camera_calib["distortion_coefficients"],
        dtype=np.float64,
    ).reshape(-1)
    is_fisheye = camera_name.startswith("CAM_F")
    if is_fisheye and D.size > 4:
        D = D[:4]

    if K_calib.shape != (3, 3):
        raise ValueError(
            f"camera_matrix must be 3x3, got {K_calib.shape}"
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------
    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"Cannot read image: {img_path}"
        )
    image_h, image_w = image.shape[:2]

    K = scale_camera_matrix(
        K_calib,
        int(camera_calib["image_width"]),
        int(camera_calib["image_height"]),
        image_w,
        image_h,
    )

    # --------------------------------------------------------
    # Read LiDAR
    # --------------------------------------------------------
    las = laspy.read(str(lidar_path))

    points_lidar = np.column_stack(
        (las.x, las.y, las.z)
    ).astype(np.float64)

    valid_lidar = np.isfinite(points_lidar).all(axis=1)
    points_lidar = points_lidar[valid_lidar]

    if len(points_lidar) == 0:
        raise ValueError(
            f"No valid LiDAR points in {lidar_path}"
        )

    # --------------------------------------------------------
    # INPUT EXTRINSIC MATRICES.
    # Raw VF6_02 calibration naming is ambiguous, so keep the two conventions
    # selectable. The visual default is "direct_lidar_to_base".
    # --------------------------------------------------------
    T_base_to_cam = get_base_to_sensor(
        extrinsics,
        camera_name,
    )

    T_base_to_lidar = get_base_to_sensor(
        extrinsics,
        lidar_name,
    )

    # --------------------------------------------------------
    # DERIVE OUTPUT TRANSFORM: LIDAR -> CAMERA
    # --------------------------------------------------------
    if extrinsic_convention == "direct_lidar_to_base":
        T_lidar_to_base = T_base_to_lidar
        formula = (
            f"T_{lidar_name}_to_{camera_name}"
            f" = T_base_to_{camera_name}"
            f" @ T_{lidar_name}_to_base"
        )
        chain_text = (
            f"  {lidar_name}"
            f" --T_{lidar_name}_to_base--> base_link"
            f" --T_base_to_{camera_name}--> {camera_name}"
        )
    elif extrinsic_convention == "base_to_sensor":
        T_lidar_to_base = np.linalg.inv(T_base_to_lidar)
        formula = (
            f"T_{lidar_name}_to_{camera_name}"
            f" = T_base_to_{camera_name}"
            f" @ inv(T_base_to_{lidar_name})"
        )
        chain_text = (
            f"  {lidar_name}"
            f" --inv(T_base_to_{lidar_name})--> base_link"
            f" --T_base_to_{camera_name}--> {camera_name}"
        )
    else:
        raise ValueError(
            f"Unknown extrinsic convention: {extrinsic_convention}"
        )

    T_lidar_to_cam = (
        T_base_to_cam
        @ T_lidar_to_base
    )

    # ========================================================
    # PRINT ALL IMPORTANT INPUT / OUTPUT MATRICES
    # ========================================================
    print("\n" + "=" * 78)
    print("CALIBRATION / TRANSFORM DEBUG")
    print("=" * 78)

    print(f"Image          : {img_path}")
    print(f"LiDAR          : {lidar_path}")
    print(f"Camera sensor  : {camera_name}")
    print(f"LiDAR sensor   : {lidar_name}")
    print(f"Camera model   : {'fisheye' if is_fisheye else 'pinhole'}")
    print(
        f"Calib size     : "
        f"{camera_calib['image_width']} x {camera_calib['image_height']}"
    )
    print(f"Image size     : {image_w} x {image_h}")
    print(f"Undistort alpha: {UNDISTORT_BALANCE}")

    print("\nConvention:")
    print(f"  {extrinsic_convention}")

    print("\nTransform chain:")
    print(chain_text)

    print_matrix(
        f"[INPUT] T_base_to_{camera_name}",
        T_base_to_cam,
    )

    print_matrix(
        f"[INPUT] T_base_to_{lidar_name}",
        T_base_to_lidar,
    )

    print_matrix(
        f"[INTERMEDIATE] T_{lidar_name}_to_base",
        T_lidar_to_base,
    )

    print_matrix(
        f"[OUTPUT] T_{lidar_name}_to_{camera_name}",
        T_lidar_to_cam,
    )

    print_matrix(
        f"[INPUT] K_calib_{camera_name}",
        K_calib,
    )

    print_matrix(
        f"[INPUT] K_scaled_{camera_name}",
        K,
    )

    print(
        f"\n[INPUT] D_{camera_name} "
        f"[k1, k2, p1, p2, k3, ...] ="
    )
    print(
        np.array2string(
            D,
            precision=9,
            suppress_small=False,
            floatmode="fixed",
        )
    )

    print("\nFormula:")
    print(f"  {formula}")

    print("=" * 78 + "\n")

    projection_mode = projection_image
    if projection_mode == "auto":
        projection_mode = "undistorted"

    image_project, K_project, D_project = prepare_projection_image(
        image,
        K,
        D,
        projection_mode,
        is_fisheye=is_fisheye,
    )
    is_fisheye_project = is_fisheye and projection_mode == "raw"

    print_matrix(
        f"[PROJECTION] K_output_{camera_name}",
        K_project,
    )

    h, w = image_project.shape[:2]
    uv, depths = project_points(
        points_lidar,
        T_lidar_to_cam[:3, :],
        K_project,
        D_project,
        use_fisheye=is_fisheye_project,
    )
    overlay, projected_count = draw_depth_overlay(image_project, uv, depths)

    print("PROJECTION STATISTICS")
    print("-" * 78)
    print(f"Projection image mode          : {projection_mode}")
    print(
        "Projection camera model        : "
        f"{'fisheye' if is_fisheye_project else 'pinhole'}"
    )
    print(f"Image size                     : {w} x {h}")
    print(f"Raw valid LiDAR points         : {len(points_lidar)}")
    print(f"Projected pixels inside image  : {projected_count}")
    print("-" * 78)

    return overlay, image_project, projected_count, projection_mode


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Project VF6_02 raw LiDAR points onto "
            "a raw pinhole camera image."
        )
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Specific JPG image path.",
    )

    parser.add_argument(
        "--lidar",
        type=Path,
        default=DEFAULT_LIDAR,
        help="Specific LAZ point-cloud path.",
    )

    parser.add_argument(
        "--camera",
        default=CAM_NAME,
        help=f"Camera sensor name (default: {CAM_NAME}).",
    )

    parser.add_argument(
        "--lidar-sensor",
        default=LIDAR_NAME,
        help=f"LiDAR sensor name (default: {LIDAR_NAME}).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Overlay output directory.",
    )

    parser.add_argument(
        "--first-n",
        type=int,
        default=1,
        help="Project the first N camera frames using nearest LiDAR timestamps.",
    )

    parser.add_argument(
        "--auto-pair",
        action="store_true",
        help=(
            "Ignore --image/--lidar and use the first N frames from the "
            "camera/lidar sensor folders."
        ),
    )

    parser.add_argument(
        "--extrinsic-convention",
        choices=("direct_lidar_to_base", "base_to_sensor"),
        default="direct_lidar_to_base",
        help=(
            "How to use the LiDAR extrinsic matrix. "
            "direct_lidar_to_base matches raw VF6_02 visual validation; "
            "base_to_sensor applies inv(T_base_to_lidar)."
        ),
    )

    parser.add_argument(
        "--projection-image",
        choices=("auto", "raw", "rectified", "undistorted"),
        default="auto",
        help=(
            "auto undistorts the image with calibration D, then projects with "
            "zero distortion. raw keeps the original JPG and projects with D. "
            "rectified keeps the original JPG and projects with zero distortion."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    print("\nPATHS")
    print("-" * 78)
    print(f"ROOT       : {ROOT}")
    print(f"CALIB_DIR  : {CALIB_DIR}")
    print(f"IMAGE_ROOT : {IMAGE_ROOT}")
    print(f"LIDAR_ROOT : {LIDAR_ROOT}")
    print(f"OUTPUT_ROOT: {args.output_dir}")
    print("-" * 78)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.auto_pair:
        args.image = None
        args.lidar = None

    if (args.image is None) != (args.lidar is None):
        raise ValueError(
            "--image and --lidar must be supplied together"
        )

    # --------------------------------------------------------
    # Explicit pair
    # --------------------------------------------------------
    if args.image is not None:
        pairs = [
            (args.image, args.lidar)
        ]

    # --------------------------------------------------------
    # Automatically choose nearest LiDAR for first N images
    # --------------------------------------------------------
    else:
        image_dir = resolve_sensor_folder(
            IMAGE_ROOT,
            args.camera,
        )

        lidar_dir = resolve_sensor_folder(
            LIDAR_ROOT,
            args.lidar_sensor,
        )

        images = sorted(
            image_dir.glob("*.jpg"),
            key=lambda p: timestamp_from_name(p.name),
        )

        lidars = sorted(
            lidar_dir.glob("*.laz"),
            key=lambda p: timestamp_from_name(p.name),
        )

        if not images:
            raise FileNotFoundError(
                f"No JPG in {image_dir}"
            )

        if not lidars:
            raise FileNotFoundError(
                f"No LAZ in {lidar_dir}"
            )

        count = min(
            args.first_n,
            len(images),
        )

        pairs = []

        for img in images[:count]:
            lidar, _ = nearest_file(
                img,
                lidars,
            )
            pairs.append(
                (img, lidar)
            )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------
    for index, (img, lidar) in enumerate(
        pairs,
        start=1,
    ):
        t_img = timestamp_from_name(img.name)
        t_lidar = timestamp_from_name(lidar.name)

        dt_ms = abs(
            t_img - t_lidar
        ) * 1000.0

        print("\n" + "#" * 78)
        print(
            f"PAIR {index}/{len(pairs)}"
        )
        print("#" * 78)

        print(f"Image timestamp : {t_img:.9f}")
        print(f"LiDAR timestamp : {t_lidar:.9f}")
        print(f"Difference      : {dt_ms:.3f} ms")

        if dt_ms > SYNC_WARNING_MS:
            print(
                f"[WARNING] Timestamp difference exceeds "
                f"{SYNC_WARNING_MS:.1f} ms"
            )

        overlay, image_project, projected_count, projection_mode = (
            project_lidar_to_camera(
                img,
                lidar,
                args.camera,
                args.lidar_sensor,
                args.extrinsic_convention,
                args.projection_image,
            )
        )

        undistorted_path = (
            args.output_dir
            / (
                f"{index:04d}_"
                f"{img.stem}_"
                f"{lidar.stem}_"
                f"{projection_mode}.jpg"
            )
        )

        output_path = (
            args.output_dir
            / (
                f"{index:04d}_"
                f"{img.stem}_"
                f"{lidar.stem}_overlay.jpg"
            )
        )

        success = cv2.imwrite(
            str(undistorted_path),
            image_project,
        )

        if not success:
            raise IOError(
                f"Cannot write undistorted image: {undistorted_path}"
            )

        success = cv2.imwrite(
            str(output_path),
            overlay,
        )

        if not success:
            raise IOError(
                f"Cannot write output image: {output_path}"
            )

        print(f"\nSaved image      : {undistorted_path}")
        print(f"Saved overlay    : {output_path}")
        print(f"Projected points : {projected_count}")


if __name__ == "__main__":
    main()
