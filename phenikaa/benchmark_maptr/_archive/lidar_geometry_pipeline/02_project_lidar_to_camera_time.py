#!/usr/bin/env python3
"""
Bước 02: đồng bộ LiDAR-camera theo timestamp và project để kiểm tra alignment.

Mục tiêu của file này là kiểm tra sensor sync/calib bằng mắt:
1. Đọc một sample từ index.jsonl.
2. Undistort ảnh camera.
3. Nội suy pose LiDAR tại timestamp camera và timestamp LiDAR.
4. Đưa point LiDAR về hệ `lidar_at_t_cam`.
5. Dùng extrinsic LiDAR -> camera để chiếu point lên ảnh đã undistort.

Hệ cuối của benchmark vẫn là MAP. File này chỉ dùng `lidar_at_t_cam`
làm hệ trung gian để chiếu lên ảnh camera.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import (
    DEFAULT_CAMERA,
    DEFAULT_LIDAR,
    DEFAULT_SCENARIO,
    draw_depth_overlay,
    ensure_output_dir,
    interpolate_pose,
    load_intrinsic,
    load_laz_xyz_intensity,
    load_lidar_to_camera,
    load_lidar_trajectory,
    project_points_to_image,
    timestamp_from_name,
    transform_points,
    undistort_for_projection,
)


def parse_args() -> argparse.Namespace:
    """Đọc tham số CLI: scenario, camera, lidar, dòng index cần kiểm tra."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--lidar", default=DEFAULT_LIDAR)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Dòng index cần chạy. Nếu bỏ trống, tự chọn pair có dt nhỏ nhất.",
    )
    parser.add_argument(
        "--lidar-frame-space",
        choices=("local", "map"),
        default="local",
        help="Use local if dump/frames/laz points are in LiDAR frame; map if already in map frame.",
    )
    return parser.parse_args()


def read_index_row(path: Path, index: int) -> dict:
    """Lấy một dòng JSON từ index.jsonl theo số thứ tự."""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"Index {index} not found in {path}")


def read_best_sync_row(path: Path) -> tuple[int, dict]:
    """Chọn dòng có |camera_lidar_dt_sec| nhỏ nhất trong index."""
    best_index = -1
    best_row = None
    best_dt = float("inf")
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            dt = abs(float(row["camera_lidar_dt_sec"]))
            if dt < best_dt:
                best_index = i
                best_row = row
                best_dt = dt
    if best_row is None:
        raise ValueError(f"Empty index: {path}")
    return best_index, best_row


def main() -> None:
    args = parse_args()
    index_path = args.index_path
    if index_path is None:
        index_path = ensure_output_dir(args.scenario) / "index.jsonl"

    if args.index is None:
        row_index, row = read_best_sync_row(index_path)
    else:
        row_index = args.index
        row = read_index_row(index_path, args.index)

    # Một sample benchmark gồm ảnh camera, LiDAR deskew gần nhất và trajectory.
    image_path = Path(row["image_path"])
    lidar_path = Path(row["deskew_lidar_path"])
    traj_path = Path(row["traj_path"])
    t_cam = float(row["image_timestamp"])
    t_lidar = float(row["deskew_lidar_timestamp"])

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)

    # Undistort ảnh trước. Sau bước này projection sẽ dùng K_project và D = 0.
    K_calib, D, calib_w, calib_h = load_intrinsic(args.camera)
    image_project, K_project, D_project = undistort_for_projection(
        image,
        K_calib,
        D,
        calib_w,
        calib_h,
        is_fisheye=args.camera.startswith("CAM_F"),
    )

    # Nội suy pose xe/LiDAR trong map tại 2 thời điểm:
    # - t_cam: thời điểm camera chụp ảnh.
    # - t_lidar: timestamp frame LiDAR deskew.
    poses = load_lidar_trajectory(traj_path)
    T_map_lidar_cam = interpolate_pose(poses, t_cam)
    T_map_lidar_frame = interpolate_pose(poses, t_lidar)

    points, _ = load_laz_xyz_intensity(lidar_path)
    if args.lidar_frame_space == "local":
        # Nếu point trong file LAZ đang ở local LiDAR frame:
        #
        #   P_map       = T_map_lidar(t_lidar) @ P_lidar_t
        #   P_lidarcam  = inv(T_map_lidar(t_cam)) @ P_map
        #
        # Gộp lại:
        #   P_lidarcam =
        #       inv(T_map_lidar(t_cam)) @ T_map_lidar(t_lidar) @ P_lidar_t
        T_lidar_frame_to_lidar_cam = np.linalg.inv(T_map_lidar_cam) @ T_map_lidar_frame
        points_lidar_cam = transform_points(T_lidar_frame_to_lidar_cam, points)
    else:
        # Nếu point đã nằm trong hệ map:
        #   P_lidarcam = inv(T_map_lidar(t_cam)) @ P_map
        points_lidar_cam = transform_points(np.linalg.inv(T_map_lidar_cam), points)

    # Từ hệ lidar_at_t_cam sang camera rồi project lên ảnh đã undistort.
    T_lidar_to_cam = load_lidar_to_camera(args.camera, args.lidar)
    uv, depths = project_points_to_image(
        points_lidar_cam,
        T_lidar_to_cam,
        K_project,
        D_project,
    )
    overlay, count = draw_depth_overlay(image_project, uv, depths)

    #Địa chỉ lưu ảnh undistorted và overlay.
    undistorted_dir = ensure_output_dir(args.scenario, "projection", "undistorted")
    overlay_dir = ensure_output_dir(args.scenario, "projection", "overlay")
    stem = f"{row_index:06d}_{image_path.stem}_{lidar_path.stem}"
    #Tên file lưu ảnh undistorted và overlay.
    undistorted_path = undistorted_dir / f"{stem}_undistorted.jpg"
    overlay_path = overlay_dir / f"{stem}_overlay.jpg"
    #Lưu ảnh undistorted và overlay.
    cv2.imwrite(str(undistorted_path), image_project)
    cv2.imwrite(str(overlay_path), overlay)

    print(f"Index           : {row_index}")
    print(f"Image timestamp : {t_cam:.9f}")
    print(f"LiDAR timestamp : {t_lidar:.9f}")
    print(f"dt              : {(t_cam - t_lidar) * 1000.0:.3f} ms")
    print(f"LiDAR points    : {len(points):,}")
    print(f"Projected inside: {count:,}")
    print(f"Saved image     : {undistorted_path}")
    print(f"Saved overlay   : {overlay_path}")


if __name__ == "__main__":
    main()
