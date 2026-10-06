#!/usr/bin/env python3
"""
Bước 03: tích lũy nhiều frame LiDAR vào hệ MAP.

Đây là bước gần với bài toán cuối của thesis hơn bước projection:
- Vector HD map là đối tượng tĩnh trong thế giới.
- Vì vậy point/evidence từ nhiều timestamp phải được đưa về cùng hệ MAP.
- File này tạo một map crop quanh một camera frame để các bước sau detect lane,
  fuse camera-LiDAR, và vectorize lanelet.

Output là file .npz chứa:
- points_map: point cloud đã nằm trong hệ map từ traj_lidar.txt.
- intensity: intensity tương ứng nếu file LAZ có.
- center_timestamp: timestamp camera trung tâm.
- frame_count: số frame LiDAR được tích lũy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import (
    DEFAULT_SCENARIO,
    ensure_output_dir,
    interpolate_pose,
    load_laz_xyz_intensity,
    load_lidar_trajectory,
    timestamp_from_name,
    transform_points,
)


def parse_args() -> argparse.Namespace:
    """Đọc tham số CLI cho cửa sổ tích lũy map quanh một index camera."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--window-sec", type=float, default=0.5)
    parser.add_argument(
        "--lidar-frame-space",
        choices=("local", "map"),
        default="local",
    )
    parser.add_argument("--max-points-per-frame", type=int, default=120_000)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    """Đọc toàn bộ index.jsonl để chọn các frame LiDAR quanh timestamp trung tâm."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    args = parse_args()
    index_path = args.index_path
    if index_path is None:
        index_path = ensure_output_dir(args.scenario) / "index.jsonl"

    rows = read_rows(index_path)

    # Camera frame trung tâm quyết định vùng thời gian cần tích lũy.
    center = rows[args.index]
    t_center = float(center["image_timestamp"])
    traj_path = Path(center["traj_path"])
    poses = load_lidar_trajectory(traj_path)

    # Lấy các frame LiDAR có timestamp nằm trong ±window-sec quanh ảnh camera.
    selected = [
        row for row in rows
        if abs(float(row["deskew_lidar_timestamp"]) - t_center) <= args.window_sec
    ]

    map_points = []
    map_intensities = []
    for row in selected:
        lidar_path = Path(row["deskew_lidar_path"])
        t_lidar = timestamp_from_name(lidar_path)
        points, intensity = load_laz_xyz_intensity(lidar_path)

        # Giới hạn số point mỗi frame để output không quá lớn trong giai đoạn thử.
        if len(points) > args.max_points_per_frame:
            ids = np.linspace(0, len(points) - 1, args.max_points_per_frame, dtype=np.int64)
            points = points[ids]
            if intensity is not None:
                intensity = intensity[ids]

        if args.lidar_frame_space == "local":
            # Nếu point còn ở hệ LiDAR local:
            #   P_map = T_map_lidar(t_lidar) @ P_lidar
            T_map_lidar = interpolate_pose(poses, t_lidar)
            points_map = transform_points(T_map_lidar, points)
        else:
            # Nếu dump LAZ đã là point trong hệ map thì giữ nguyên.
            points_map = points
        map_points.append(points_map.astype(np.float32))
        if intensity is not None:
            map_intensities.append(intensity.astype(np.float32))

    if not map_points:
        raise RuntimeError("No LiDAR frames selected")

    points_out = np.concatenate(map_points, axis=0)
    intensities_out = (
        np.concatenate(map_intensities, axis=0)
        if len(map_intensities) == len(map_points)
        else np.empty((0,), dtype=np.float32)
    )

    out_dir = ensure_output_dir(args.scenario, "map_accumulation")
    out_path = out_dir / f"{args.index:06d}_map_points_{args.window_sec:.2f}s.npz"

    # File này là dữ liệu trung gian cho bước detect/fusion/vectorization sau.
    np.savez_compressed(
        out_path,
        points_map=points_out,
        intensity=intensities_out,
        center_timestamp=t_center,
        frame_count=len(selected),
        map_frame="traj_lidar_map",
    )

    print(f"Selected frames : {len(selected)}")
    print(f"Map points      : {len(points_out):,}")
    print(f"Saved           : {out_path}")


if __name__ == "__main__":
    main()
