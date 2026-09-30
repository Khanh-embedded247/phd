#!/usr/bin/env python3
"""
Bước 01: tạo index benchmark cho một scenario.

File này không xử lý point cloud. Nó chỉ gom metadata:
- ảnh camera nào,
- timestamp ảnh,
- frame LiDAR deskew gần nhất,
- timestamp LiDAR,
- đường dẫn traj_lidar.txt.

Index này là "xương sống" cho các bước sau: projection, tích lũy map,
detect lane, vectorize, và benchmark theo scenario.

Lưu ý: camera có thể bắt đầu sớm hơn dump/frames/laz. Vì vậy script mặc định
bỏ các pair có lệch timestamp quá lớn để tránh sample đầu tiên bị lệch > 1s.
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

from common import (
    DEFAULT_CAMERA,
    DEFAULT_SCENARIO,
    ensure_output_dir,
    scenario_root,
    timestamp_from_name,
)


def parse_args() -> argparse.Namespace:
    """Đọc tham số CLI cho scenario/camera cần benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--output-name", default="index.jsonl")
    parser.add_argument(
        "--max-dt-sec",
        type=float,
        default=0.05,
        help="Chỉ giữ camera-LiDAR pair có lệch timestamp <= ngưỡng này.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Layout dữ liệu hiện tại:
    # data/Normal/CAMERA/CAM_P_F/*.jpg
    # data/Normal/dump/frames/laz/*.laz
    # data/Normal/dump/traj_lidar.txt
    root = scenario_root(args.scenario)
    camera_dir = root / "CAMERA" / args.camera
    dump_dir = root / "dump"
    deskew_lidar_dir = dump_dir / "frames" / "laz"
    traj_path = dump_dir / "traj_lidar.txt"

    if not camera_dir.is_dir():
        raise FileNotFoundError(camera_dir)
    if not deskew_lidar_dir.is_dir():
        raise FileNotFoundError(deskew_lidar_dir)
    if not traj_path.is_file():
        raise FileNotFoundError(traj_path)

    images = sorted(camera_dir.glob("*.jpg"), key=timestamp_from_name)
    lidars = sorted(deskew_lidar_dir.glob("*.laz"), key=timestamp_from_name)
    lidar_times = [timestamp_from_name(p) for p in lidars]

    out_dir = ensure_output_dir(args.scenario)
    out_path = out_dir / args.output_name

    count = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for image_path in images:
            t_cam = timestamp_from_name(image_path)

            # Tìm frame LiDAR deskew gần timestamp camera nhất.
            # Bước 02 vẫn sẽ dùng trajectory để bù pose nếu còn lệch timestamp.
            insert = bisect.bisect_left(lidar_times, t_cam)
            candidates = []
            if insert > 0:
                candidates.append((abs(lidar_times[insert - 1] - t_cam), lidars[insert - 1]))
            if insert < len(lidars):
                candidates.append((abs(lidar_times[insert] - t_cam), lidars[insert]))
            dt, lidar_path = min(candidates, key=lambda item: item[0])

            # Bỏ các ảnh nằm ngoài khoảng thời gian có LiDAR deskew tương ứng.
            # Ví dụ frame camera đầu tiên của Normal sớm hơn LiDAR deskew đầu tiên
            # khoảng 1.07s, không nên đưa vào benchmark sync/projection.
            if dt > args.max_dt_sec:
                skipped += 1
                continue

            row = {
                # Mỗi dòng JSONL là một sample benchmark.
                # Giữ index nhẹ: pose sẽ được nội suy lại khi cần ở bước sau.
                "scenario": args.scenario,
                "camera": args.camera,
                "image_path": str(image_path),
                "image_timestamp": t_cam,
                "deskew_lidar_path": str(lidar_path),
                "deskew_lidar_timestamp": timestamp_from_name(lidar_path),
                "camera_lidar_dt_sec": dt,
                "traj_path": str(traj_path),
            }
            f.write(json.dumps(row) + "\n")
            count += 1

    print(f"Wrote {count} rows: {out_path}")
    print(f"Skipped {skipped} rows with dt > {args.max_dt_sec:.3f}s")


if __name__ == "__main__":
    main()
