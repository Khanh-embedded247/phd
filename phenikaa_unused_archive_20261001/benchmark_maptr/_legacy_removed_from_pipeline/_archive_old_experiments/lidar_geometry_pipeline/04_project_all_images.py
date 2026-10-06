#!/usr/bin/env python3
"""
Bước 04: project LiDAR lên nhiều ảnh để tạo QA report cho toàn scenario.

Mục tiêu:
- Chạy qua toàn bộ index.jsonl.
- Ghi CSV cho tất cả frame: timestamp, LiDAR nearest, số point, số point lọt ảnh, ratio.
- Chỉ lưu overlay cho một số frame đại diện, mặc định tối đa 100 ảnh.
- Lưu thêm các frame overlay liên tiếp trong 20s đầu để bước 05 ghép video.

Lý do:
- Benchmark cần số liệu đầy đủ cho toàn scenario.
- Nhưng overlay JPG cho mọi frame sẽ rất tốn dung lượng.
"""

from __future__ import annotations

import argparse
import csv
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
    transform_points,
    undistort_for_projection,
)


# ============================================================
# USER CONFIG
# ============================================================
# Bạn có thể chỉnh trực tiếp các giá trị này trong code rồi chạy file.
# CLI args vẫn có thể override nếu cần.

CONFIG_SCENARIO = DEFAULT_SCENARIO
CONFIG_CAMERA = DEFAULT_CAMERA
CONFIG_LIDAR = DEFAULT_LIDAR

# Bắt đầu từ dòng nào trong index.jsonl.
CONFIG_START_INDEX = 0

# Chạy bao nhiêu dòng index. None = toàn bộ index.
CONFIG_MAX_FRAMES = None

# Lưu tối đa bao nhiêu ảnh overlay đại diện trong projection_all/overlays.
CONFIG_MAX_OVERLAYS = 100

# Thời lượng chuỗi frame liên tiếp để file 05 ghép video.
CONFIG_SEQUENCE_DURATION_SEC = 20.0

# "local": point trong LAZ đang ở hệ LiDAR tại timestamp frame.
# "map": point trong LAZ đã nằm trong hệ map.
CONFIG_LIDAR_FRAME_SPACE = "local"

# Có lưu chuỗi frame liên tiếp cho video hay không.
CONFIG_SAVE_SEQUENCE_FRAMES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=CONFIG_SCENARIO)
    parser.add_argument("--camera", default=CONFIG_CAMERA)
    parser.add_argument("--lidar", default=CONFIG_LIDAR)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument(
        "--start-index",
        type=int,
        default=CONFIG_START_INDEX,
        help="Bắt đầu xử lý từ dòng index này.",
    )
    parser.add_argument("--max-overlays", type=int, default=CONFIG_MAX_OVERLAYS)
    parser.add_argument(
        "--sequence-duration-sec",
        type=float,
        default=CONFIG_SEQUENCE_DURATION_SEC,
        help="Lưu overlay liên tiếp trong N giây đầu theo timeline camera.",
    )
    parser.add_argument(
        "--sequence-frame-dir",
        type=Path,
        default=None,
        help="Thư mục lưu frame liên tiếp cho video. Mặc định nằm trong projection_all/sequence_20s/frames.",
    )
    parser.add_argument(
        "--no-sequence-frames",
        action="store_true",
        default=not CONFIG_SAVE_SEQUENCE_FRAMES,
        help="Tắt lưu frame liên tiếp 20s, chỉ ghi CSV/overlay đại diện.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=CONFIG_MAX_FRAMES,
        help="Giới hạn số frame để test nhanh. Mặc định chạy toàn bộ index.",
    )
    parser.add_argument(
        "--lidar-frame-space",
        choices=("local", "map"),
        default=CONFIG_LIDAR_FRAME_SPACE,
    )
    return parser.parse_args()


def read_rows(path: Path, start_index: int, max_frames: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index < start_index:
                continue
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_index"] = index
            rows.append(row)
            if max_frames is not None and len(rows) >= max_frames:
                break
    return rows


def choose_overlay_indices(total: int, max_overlays: int) -> set[int]:
    """Chọn tối đa max_overlays index cách đều để lưu ảnh overlay."""
    if max_overlays <= 0 or total <= 0:
        return set()
    if total <= max_overlays:
        return set(range(total))
    return set(np.linspace(0, total - 1, max_overlays, dtype=np.int64).tolist())


def main() -> None:
    args = parse_args()

    index_path = args.index_path
    if index_path is None:
        index_path = ensure_output_dir(args.scenario) / "index.jsonl"

    rows = read_rows(index_path, args.start_index, args.max_frames)
    if not rows:
        raise ValueError(f"Empty index: {index_path}")

    # Các dữ liệu calibration/trajectory không đổi cho cả scenario, load một lần.
    first_row = rows[0]
    traj_path = Path(first_row["traj_path"])

    poses = load_lidar_trajectory(traj_path)
    K_calib, D, calib_w, calib_h = load_intrinsic(args.camera)
    T_lidar_to_cam = load_lidar_to_camera(args.camera, args.lidar)

    out_dir = ensure_output_dir(args.scenario, "projection_all")
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.sequence_frame_dir is None:
        sequence_frame_dir = (
            out_dir
            / f"sequence_{args.sequence_duration_sec:.1f}s"
            / "frames"
        )
    else:
        sequence_frame_dir = args.sequence_frame_dir
    if not args.no_sequence_frames:
        sequence_frame_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in sequence_frame_dir.glob("frame_*.jpg"):
            old_frame.unlink()
    report_path = out_dir / "projection_report.csv"

    overlay_indices = choose_overlay_indices(len(rows), args.max_overlays)
    sequence_start_ts = float(rows[0]["image_timestamp"])
    sequence_end_ts = sequence_start_ts + args.sequence_duration_sec
    sequence_count = 0

    fieldnames = [
        "index",
        "scenario",
        "camera",
        "lidar",
        "lidar_frame_space",
        "sync_method",
        "sequence_duration_sec",
        "image_path",
        "lidar_path",
        "image_timestamp",
        "lidar_timestamp",
        "dt_ms",
        "lidar_points",
        "projected_inside",
        "projected_ratio",
        "overlay_saved",
        "overlay_path",
        "sequence_saved",
        "sequence_frame_index",
        "sequence_frame_path",
    ]

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for index, row in enumerate(rows):
            source_index = int(row["_source_index"])
            image_path = Path(row["image_path"])
            lidar_path = Path(row["deskew_lidar_path"])
            t_cam = float(row["image_timestamp"])
            t_lidar = float(row["deskew_lidar_timestamp"])

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)

            # Ảnh dùng để project luôn là ảnh đã undistort.
            image_project, K_project, D_project = undistort_for_projection(
                image,
                K_calib,
                D,
                calib_w,
                calib_h,
                is_fisheye=args.camera.startswith("CAM_F"),
            )

            # Đồng bộ point về hệ LiDAR tại timestamp camera.
            # Đây là phần cốt lõi:
            # - Chọn một LiDAR deskew frame gần t_cam nhất từ index.jsonl.
            # - Không coi nó là cùng timestamp với camera.
            # - Warp toàn bộ cloud deskew từ t_lidar sang lidar_at_t_cam:
            #
            #   P_lidar_at_t_cam =
            #       inv(T_map_lidar(t_cam))
            #       @ T_map_lidar(t_lidar)
            #       @ P_lidar_at_t_lidar
            T_map_lidar_cam = interpolate_pose(poses, t_cam)
            points, _ = load_laz_xyz_intensity(lidar_path)
            total_lidar_points = len(points)

            if args.lidar_frame_space == "local":
                T_map_lidar_frame = interpolate_pose(poses, t_lidar)
                T_lidar_frame_to_lidar_cam = (
                    np.linalg.inv(T_map_lidar_cam) @ T_map_lidar_frame
                )
                points_lidar_cam = transform_points(
                    T_lidar_frame_to_lidar_cam,
                    points,
                )
            else:
                points_lidar_cam = transform_points(
                    np.linalg.inv(T_map_lidar_cam),
                    points,
                )

            dt_ms = (t_cam - t_lidar) * 1000.0

            uv, depths = project_points_to_image(
                points_lidar_cam,
                T_lidar_to_cam,
                K_project,
                D_project,
            )

            overlay_path = ""
            sequence_frame_path = ""
            sequence_frame_index = ""
            overlay_saved = index in overlay_indices
            sequence_saved = (
                not args.no_sequence_frames
                and sequence_start_ts <= t_cam <= sequence_end_ts
            )
            need_overlay = overlay_saved or sequence_saved

            if need_overlay:
                overlay, projected_inside = draw_depth_overlay(
                    image_project,
                    uv,
                    depths,
                )
                if overlay_saved:
                    overlay_path_obj = overlay_dir / (
                        f"{source_index:06d}_{image_path.stem}_overlay.jpg"
                    )
                    cv2.imwrite(str(overlay_path_obj), overlay)
                    overlay_path = str(overlay_path_obj)

                if sequence_saved:
                    sequence_frame_index = sequence_count
                    sequence_path_obj = sequence_frame_dir / (
                        f"frame_{sequence_count:06d}.jpg"
                    )
                    cv2.imwrite(str(sequence_path_obj), overlay)
                    sequence_frame_path = str(sequence_path_obj)
                    sequence_count += 1

            else:
                h, w = image_project.shape[:2]
                valid = (
                    np.isfinite(uv).all(axis=1)
                    & (uv[:, 0] >= 0)
                    & (uv[:, 0] < w)
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < h)
                )
                projected_inside = int(valid.sum())

            ratio = (
                projected_inside / total_lidar_points
                if total_lidar_points
                else 0.0
            )

            writer.writerow(
                {
                    "index": source_index,
                    "scenario": args.scenario,
                    "camera": args.camera,
                    "lidar": args.lidar,
                    "lidar_frame_space": args.lidar_frame_space,
                    "sync_method": "nearest_lidar_warp_to_camera_time",
                    "sequence_duration_sec": f"{args.sequence_duration_sec:.3f}",
                    "image_path": str(image_path),
                    "lidar_path": str(lidar_path),
                    "image_timestamp": f"{t_cam:.9f}",
                    "lidar_timestamp": f"{t_lidar:.9f}",
                    "dt_ms": f"{dt_ms:.3f}",
                    "lidar_points": total_lidar_points,
                    "projected_inside": projected_inside,
                    "projected_ratio": f"{ratio:.6f}",
                    "overlay_saved": int(overlay_saved),
                    "overlay_path": overlay_path,
                    "sequence_saved": int(sequence_saved),
                    "sequence_frame_index": sequence_frame_index,
                    "sequence_frame_path": sequence_frame_path,
                }
            )

            if (index + 1) % 100 == 0 or index + 1 == len(rows):
                print(
                    f"Processed {index + 1}/{len(rows)} "
                    f"(overlays saved: {sum(i <= index for i in overlay_indices)})"
                )

    print(f"Report saved : {report_path}")
    print(f"Overlay dir  : {overlay_dir}")
    print(f"Overlay count: {len(overlay_indices)}")
    if not args.no_sequence_frames:
        print(f"Sequence dir : {sequence_frame_dir}")
        print(f"Sequence frames saved: {sequence_count}")


if __name__ == "__main__":
    main()
