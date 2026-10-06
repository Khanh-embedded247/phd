#!/usr/bin/env python3
"""
Bước 07: lọc ground/mặt đường chuẩn từ từng frame LiDAR deskew.

Đây là pipeline chuẩn hơn so với lọc trực tiếp từ `map - Cloud.pcd`.

Vì sao:
- Mỗi file LAZ deskew có timestamp trùng timestamp pose trong `traj_lidar.txt`.
- Vì vậy với frame LiDAR nào, ta dùng đúng pose `T_map_lidar` của frame đó.
- Sau đó dùng calib `T_base_lidar` để đưa point về `base_link` thật của xe.
- Lọc ground theo z trong hệ `base_link`.
- Point nào là ground thì đưa về hệ `map` bằng đúng `T_map_lidar`.

Input:
    data/Normal/dump/frames/laz/*.laz
    data/Normal/dump/traj_lidar.txt
    calib/vf_06_02/VF6_02_Extrinsics_By_Dates.json

Output:
    outputs/benchmark/Normal/ground_road_base_link/
        ground_road_points_map.npz
        report.json
        bev_ground_density.png
        bev_ground_intensity.png
        bev_ground_z_base.png

Quy ước transform:
- `traj_lidar.txt` là T_map_lidar_top: đổi point từ LiDAR_TOP sang map.
- extrinsic `LIDAR_TOP` đang được dùng là T_base_lidar_top:
      point_base = T_base_lidar_top @ point_lidar
- Khi cần pose base trong map:
      T_map_base = T_map_lidar_top @ inv(T_base_lidar_top)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import laspy
import numpy as np

from common import (
    DEFAULT_LIDAR,
    DEFAULT_SCENARIO,
    EXTRINSIC_JSON,
    PHENIKAA_ROOT,
    ensure_output_dir,
    interpolate_pose,
    load_json,
    load_trajectory,
    timestamp_from_name,
    transform_points,
)


# ============================================================
# USER CONFIG
# ============================================================

CONFIG_SCENARIO = DEFAULT_SCENARIO
CONFIG_LIDAR = DEFAULT_LIDAR
CONFIG_LAZ_DIR = PHENIKAA_ROOT / "data" / CONFIG_SCENARIO / "dump" / "frames" / "laz"
CONFIG_TRAJ = PHENIKAA_ROOT / "data" / CONFIG_SCENARIO / "dump" / "traj_lidar.txt"

# None = chạy toàn bộ frame. Khi test nhanh có thể đặt 50, 100, 200...
CONFIG_START_INDEX = 0
CONFIG_MAX_FRAMES = None

# ROI trong hệ base_link. x là trước/sau xe, y là trái/phải xe.
CONFIG_MIN_X_BASE = -25.0
CONFIG_MAX_X_BASE = 60.0
CONFIG_MAX_ABS_Y_BASE = 20.0

# Ground quanh base_link. Base_link nằm sát mặt đất nên mặt đường thường quanh z=0.
CONFIG_MIN_Z_BASE = -0.45
CONFIG_MAX_Z_BASE = 0.35

# Voxel trong hệ map để giảm trùng lặp khi gom nhiều frame.
# Đặt 0 hoặc âm nếu muốn giữ toàn bộ point, nhưng file output có thể rất lớn.
CONFIG_VOXEL_SIZE = 0.12

# Chỉ dùng frame LAZ có timestamp trùng pose lidar trong traj trong sai số này.
# Nếu frame nằm ngoài traj thì bỏ qua, không dùng pose clamp ở đầu/cuối.
CONFIG_POSE_TIMESTAMP_TOLERANCE_SEC = 1e-6

# Mỗi lần buffer đạt số point này thì downsample để tránh ăn RAM.
CONFIG_FLUSH_POINTS = 1_500_000

# BEV output.
CONFIG_BEV_RESOLUTION = 0.10
CONFIG_MAX_BEV_PIXELS = 25_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=CONFIG_SCENARIO)
    parser.add_argument("--lidar", default=CONFIG_LIDAR)
    parser.add_argument("--laz-dir", type=Path, default=CONFIG_LAZ_DIR)
    parser.add_argument("--traj", type=Path, default=CONFIG_TRAJ)
    parser.add_argument("--start-index", type=int, default=CONFIG_START_INDEX)
    parser.add_argument("--max-frames", type=int, default=CONFIG_MAX_FRAMES)
    parser.add_argument("--min-x-base", type=float, default=CONFIG_MIN_X_BASE)
    parser.add_argument("--max-x-base", type=float, default=CONFIG_MAX_X_BASE)
    parser.add_argument("--max-abs-y-base", type=float, default=CONFIG_MAX_ABS_Y_BASE)
    parser.add_argument("--min-z-base", type=float, default=CONFIG_MIN_Z_BASE)
    parser.add_argument("--max-z-base", type=float, default=CONFIG_MAX_Z_BASE)
    parser.add_argument("--voxel-size", type=float, default=CONFIG_VOXEL_SIZE)
    parser.add_argument("--pose-timestamp-tolerance-sec", type=float, default=CONFIG_POSE_TIMESTAMP_TOLERANCE_SEC)
    parser.add_argument("--flush-points", type=int, default=CONFIG_FLUSH_POINTS)
    parser.add_argument("--bev-resolution", type=float, default=CONFIG_BEV_RESOLUTION)
    parser.add_argument("--max-bev-pixels", type=int, default=CONFIG_MAX_BEV_PIXELS)
    return parser.parse_args()


def load_laz_xyz_intensity(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Đọc một frame LAZ thành XYZ trong hệ LiDAR_TOP và intensity nếu có."""
    las = laspy.read(str(path))
    points = np.column_stack((las.x, las.y, las.z)).astype(np.float32)
    intensity = np.asarray(las.intensity, dtype=np.float32) if hasattr(las, "intensity") else None
    return points, intensity


def select_laz_files(laz_dir: Path, start_index: int, max_frames: int | None) -> list[Path]:
    """Lấy danh sách LAZ theo thứ tự timestamp."""
    paths = sorted(laz_dir.glob("*.laz"), key=timestamp_from_name)
    if start_index > 0:
        paths = paths[start_index:]
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise ValueError(f"No LAZ files selected from {laz_dir}")
    return paths


def voxel_downsample(
    points_map: np.ndarray,
    local_xyz_base: np.ndarray,
    intensity: np.ndarray | None,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Downsample bằng voxel trong hệ map.

    Bản đầu này giữ một point đại diện cho mỗi voxel để pipeline nhẹ và dễ kiểm tra.
    Sau này có thể đổi sang lấy trung bình trong voxel nếu cần mượt hơn.
    """
    if voxel_size <= 0.0 or len(points_map) == 0:
        return points_map, local_xyz_base, intensity

    keys = np.floor(points_map / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    unique_idx.sort()
    points_ds = points_map[unique_idx]
    local_ds = local_xyz_base[unique_idx]
    intensity_ds = intensity[unique_idx] if intensity is not None else None
    return points_ds, local_ds, intensity_ds


def merge_and_downsample(
    point_blocks: list[np.ndarray],
    local_blocks: list[np.ndarray],
    intensity_blocks: list[np.ndarray],
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Gộp buffer rồi voxel downsample."""
    points = np.concatenate(point_blocks, axis=0)
    local = np.concatenate(local_blocks, axis=0)
    intensity = np.concatenate(intensity_blocks, axis=0) if intensity_blocks else None
    return voxel_downsample(points, local, intensity, voxel_size)


def normalize_to_u8(values: np.ndarray, clip_percentile: tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    """Chuẩn hóa layer float thành ảnh uint8."""
    finite = np.isfinite(values)
    out = np.zeros(values.shape, dtype=np.uint8)
    if not finite.any():
        return out
    lo, hi = np.percentile(values[finite], clip_percentile)
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    out[finite] = (scaled[finite] * 255.0).astype(np.uint8)
    return out


def save_colormap(path: Path, image_u8: np.ndarray) -> None:
    """Lưu ảnh màu dễ nhìn."""
    cv2.imwrite(str(path), cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO))


def build_bev_layers(
    points_map: np.ndarray,
    z_base: np.ndarray,
    intensity: np.ndarray | None,
    resolution: float,
    max_pixels: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Tạo BEV để kiểm tra ground map bằng mắt."""
    min_xy = points_map[:, :2].min(axis=0)
    max_xy = points_map[:, :2].max(axis=0)
    span = max_xy - min_xy

    width = int(np.ceil(span[0] / resolution)) + 1
    height = int(np.ceil(span[1] / resolution)) + 1
    used_resolution = resolution
    pixels = width * height
    if pixels > max_pixels:
        scale = np.sqrt(pixels / max_pixels)
        used_resolution = resolution * scale
        width = int(np.ceil(span[0] / used_resolution)) + 1
        height = int(np.ceil(span[1] / used_resolution)) + 1

    ix = np.floor((points_map[:, 0] - min_xy[0]) / used_resolution).astype(np.int32)
    iy = np.floor((points_map[:, 1] - min_xy[1]) / used_resolution).astype(np.int32)
    ix = np.clip(ix, 0, width - 1)
    iy = np.clip(iy, 0, height - 1)
    row = height - 1 - iy
    col = ix

    density = np.zeros((height, width), dtype=np.float32)
    z_sum = np.zeros((height, width), dtype=np.float64)
    z_count = np.zeros((height, width), dtype=np.float64)
    np.add.at(density, (row, col), 1.0)
    np.add.at(z_sum, (row, col), z_base.astype(np.float64))
    np.add.at(z_count, (row, col), 1.0)
    z_mean = np.divide(
        z_sum,
        z_count,
        out=np.full_like(z_sum, np.nan),
        where=z_count > 0,
    ).astype(np.float32)

    layers: dict[str, np.ndarray] = {
        "density": density,
        "z_base_mean": z_mean,
    }

    if intensity is not None:
        intensity_sum = np.zeros((height, width), dtype=np.float64)
        intensity_count = np.zeros((height, width), dtype=np.float64)
        np.add.at(intensity_sum, (row, col), intensity.astype(np.float64))
        np.add.at(intensity_count, (row, col), 1.0)
        layers["intensity_mean"] = np.divide(
            intensity_sum,
            intensity_count,
            out=np.full_like(intensity_sum, np.nan),
            where=intensity_count > 0,
        ).astype(np.float32)

    info = {
        "min_xy": min_xy.tolist(),
        "max_xy": max_xy.tolist(),
        "resolution_requested": float(resolution),
        "resolution_used": float(used_resolution),
        "bev_width": int(width),
        "bev_height": int(height),
    }
    return layers, info


def flush_if_needed(
    final_points: list[np.ndarray],
    final_local: list[np.ndarray],
    final_intensity: list[np.ndarray],
    buffer_points: list[np.ndarray],
    buffer_local: list[np.ndarray],
    buffer_intensity: list[np.ndarray],
    voxel_size: float,
    force: bool = False,
) -> int:
    """Flush buffer ground point và downsample để giảm RAM."""
    count = sum(len(block) for block in buffer_points)
    if count == 0 or (not force and count < 1):
        return 0

    points_ds, local_ds, intensity_ds = merge_and_downsample(
        buffer_points,
        buffer_local,
        buffer_intensity,
        voxel_size,
    )
    final_points.append(points_ds)
    final_local.append(local_ds)
    if intensity_ds is not None:
        final_intensity.append(intensity_ds)
    buffer_points.clear()
    buffer_local.clear()
    buffer_intensity.clear()
    return int(len(points_ds))


def main() -> None:
    args = parse_args()
    if not args.laz_dir.is_dir():
        raise FileNotFoundError(args.laz_dir)
    if not args.traj.is_file():
        raise FileNotFoundError(args.traj)

    out_dir = ensure_output_dir(args.scenario, "ground_road_base_link")
    laz_paths = select_laz_files(args.laz_dir, args.start_index, args.max_frames)
    trajectory = load_trajectory(args.traj)
    T_base_lidar = np.asarray(load_json(EXTRINSIC_JSON)[args.lidar], dtype=np.float64)
    T_lidar_base = np.linalg.inv(T_base_lidar)

    final_points: list[np.ndarray] = []
    final_local: list[np.ndarray] = []
    final_intensity: list[np.ndarray] = []
    buffer_points: list[np.ndarray] = []
    buffer_local: list[np.ndarray] = []
    buffer_intensity: list[np.ndarray] = []

    total_points = 0
    total_ground_before_voxel = 0
    max_pose_dt = 0.0
    max_used_pose_dt = 0.0
    skipped_no_pose = []

    for frame_idx, laz_path in enumerate(laz_paths):
        ts = timestamp_from_name(laz_path)

        # Chỉ chấp nhận frame có pose trùng timestamp. Nếu không có pose đúng,
        # bỏ qua frame để tránh dùng pose clamp/interpolate sai thời điểm.
        nearest_i = int(np.argmin(np.abs(trajectory.timestamps - ts)))
        pose_dt = abs(float(trajectory.timestamps[nearest_i] - ts))
        max_pose_dt = max(max_pose_dt, pose_dt)
        if pose_dt > args.pose_timestamp_tolerance_sec:
            skipped_no_pose.append(
                {
                    "laz": str(laz_path),
                    "timestamp": float(ts),
                    "nearest_pose_timestamp": float(trajectory.timestamps[nearest_i]),
                    "dt_sec": float(pose_dt),
                }
            )
            continue
        max_used_pose_dt = max(max_used_pose_dt, pose_dt)

        # Interpolate pose từ traj_lidar.txt.
        T_map_lidar = interpolate_pose(trajectory, ts)

        points_lidar, intensity = load_laz_xyz_intensity(laz_path)
        total_points += int(len(points_lidar))

        points_base = transform_points(T_base_lidar, points_lidar).astype(np.float32)
        roi = (
            (points_base[:, 0] >= args.min_x_base)
            & (points_base[:, 0] <= args.max_x_base)
            & (np.abs(points_base[:, 1]) <= args.max_abs_y_base)
            & (points_base[:, 2] >= args.min_z_base)
            & (points_base[:, 2] <= args.max_z_base)
            & np.isfinite(points_base).all(axis=1)
        )
        if not roi.any():
            continue

        ground_lidar = points_lidar[roi]
        ground_base = points_base[roi]
        # Đổi tọa độ từ LiDAR_TOP sang map.
        ground_map = transform_points(T_map_lidar, ground_lidar).astype(np.float32)
        total_ground_before_voxel += int(len(ground_map))

        buffer_points.append(ground_map)
        buffer_local.append(ground_base)
        if intensity is not None:
            buffer_intensity.append(intensity[roi].astype(np.float32))

        buffered = sum(len(block) for block in buffer_points)
        if buffered >= args.flush_points:
            flush_if_needed(
                final_points,
                final_local,
                final_intensity,
                buffer_points,
                buffer_local,
                buffer_intensity,
                args.voxel_size,
                force=True,
            )

        if (frame_idx + 1) % 100 == 0:
            print(f"Processed {frame_idx + 1}/{len(laz_paths)} frames")

    flush_if_needed(
        final_points,
        final_local,
        final_intensity,
        buffer_points,
        buffer_local,
        buffer_intensity,
        args.voxel_size,
        force=True,
    )

    ground_points_map, local_xyz_base, ground_intensity = merge_and_downsample(
        final_points,
        final_local,
        final_intensity,
        args.voxel_size,
    )

    npz_path = out_dir / "ground_road_points_map.npz"
    save_data = {
        "points_map": ground_points_map.astype(np.float32),
        "local_xyz_base": local_xyz_base.astype(np.float32),
    }
    if ground_intensity is not None:
        save_data["intensity"] = ground_intensity.astype(np.float32)
    np.savez_compressed(npz_path, **save_data)

    layers, bev_info = build_bev_layers(
        ground_points_map,
        local_xyz_base[:, 2],
        ground_intensity,
        args.bev_resolution,
        args.max_bev_pixels,
    )

    density_u8 = normalize_to_u8(np.log1p(layers["density"]), (1, 99.5))
    save_colormap(out_dir / "bev_ground_density.png", density_u8)

    z_base_u8 = normalize_to_u8(layers["z_base_mean"], (1, 99))
    save_colormap(out_dir / "bev_ground_z_base.png", z_base_u8)

    if "intensity_mean" in layers:
        intensity_u8 = normalize_to_u8(layers["intensity_mean"], (1, 99))
        cv2.imwrite(str(out_dir / "bev_ground_intensity.png"), intensity_u8)
        save_colormap(out_dir / "bev_ground_intensity_color.png", intensity_u8)

    report = {
        "method": "deskew_laz_exact_timestamp_pose_to_base_link_then_map",
        "scenario": args.scenario,
        "lidar": args.lidar,
        "laz_dir": str(args.laz_dir),
        "traj": str(args.traj),
        "frame_count": int(len(laz_paths)),
        "start_index": int(args.start_index),
        "max_frames": args.max_frames,
        "total_lidar_points": int(total_points),
        "ground_points_before_voxel": int(total_ground_before_voxel),
        "ground_points_after_voxel": int(len(ground_points_map)),
        "max_pose_timestamp_abs_dt_sec": float(max_pose_dt),
        "max_used_pose_timestamp_abs_dt_sec": float(max_used_pose_dt),
        "skipped_no_exact_pose_count": int(len(skipped_no_pose)),
        "skipped_no_exact_pose": skipped_no_pose,
        "assumption": {
            "traj_lidar": "T_map_lidar_top",
            "extrinsic_lidar": "T_base_lidar_top",
            "point_base": "T_base_lidar_top @ point_lidar",
            "point_map": "T_map_lidar_top @ point_lidar",
            "T_map_base": "T_map_lidar_top @ inv(T_base_lidar_top)",
        },
        "thresholds": {
            "min_x_base": float(args.min_x_base),
            "max_x_base": float(args.max_x_base),
            "max_abs_y_base": float(args.max_abs_y_base),
            "min_z_base": float(args.min_z_base),
            "max_z_base": float(args.max_z_base),
            "voxel_size": float(args.voxel_size),
            "pose_timestamp_tolerance_sec": float(args.pose_timestamp_tolerance_sec),
        },
        "bev": bev_info,
        "outputs": {
            "ground_points_npz": str(npz_path),
            "bev_ground_density": str(out_dir / "bev_ground_density.png"),
            "bev_ground_z_base": str(out_dir / "bev_ground_z_base.png"),
            "bev_ground_intensity": str(out_dir / "bev_ground_intensity.png"),
        },
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Frames                  : {len(laz_paths):,}")
    print(f"Raw LiDAR points         : {total_points:,}")
    print(f"Ground before voxel      : {total_ground_before_voxel:,}")
    print(f"Ground after voxel       : {len(ground_points_map):,}")
    print(f"Max pose timestamp dt    : {max_pose_dt * 1000.0:.6f} ms")
    print(f"Max used pose dt         : {max_used_pose_dt * 1000.0:.6f} ms")
    print(f"Skipped no exact pose    : {len(skipped_no_pose):,}")
    print(f"Output dir               : {out_dir}")
    print(f"NPZ                      : {npz_path}")


if __name__ == "__main__":
    main()
