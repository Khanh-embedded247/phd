#!/usr/bin/env python3
"""
Bước 05: ghép video từ các frame overlay đã tạo ở bước 04.

File này KHÔNG project lại LiDAR.
Nó chỉ đọc chuỗi ảnh liên tiếp:

    outputs/benchmark/<scenario>/projection_all/sequence_20.0s/frames/frame_*.jpg

rồi ghép thành MP4. Cách này giúp video mượt và đúng timeline ảnh camera,
vì toàn bộ frame đã được tạo liên tiếp ở bước 04.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common import DEFAULT_SCENARIO, ensure_output_dir


# ============================================================
# USER CONFIG
# ============================================================
# File 05 chỉ ghép video từ ảnh overlay đã tạo bởi file 04.
# Chỉnh các giá trị này nếu muốn thay duration/fps/tên video trong code.

CONFIG_SCENARIO = DEFAULT_SCENARIO
CONFIG_SEQUENCE_DURATION_SEC = 20.0
CONFIG_VIDEO_FPS = 30.0
CONFIG_OUTPUT_NAME = "projection_sequence_20s.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=CONFIG_SCENARIO)
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=CONFIG_VIDEO_FPS)
    parser.add_argument("--duration-sec", type=float, default=CONFIG_SEQUENCE_DURATION_SEC)
    parser.add_argument("--output-name", default=CONFIG_OUTPUT_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.frame_dir is None:
        args.frame_dir = (
            ensure_output_dir(args.scenario)
            / "projection_all"
            / f"sequence_{args.duration_sec:.1f}s"
            / "frames"
        )

    frame_paths = sorted(args.frame_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(
            f"No frame_*.jpg found in {args.frame_dir}. "
            "Run 04_project_all_images.py first."
        )

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(frame_paths[0])
    h, w = first.shape[:2]

    out_dir = ensure_output_dir(args.scenario, "videos")
    output_path = out_dir / args.output_name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (w, h))
    if not writer.isOpened():
        raise IOError(f"Cannot create video: {output_path}")

    try:
        for index, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(frame_path)
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            if (index + 1) % 100 == 0 or index + 1 == len(frame_paths):
                print(f"Added {index + 1}/{len(frame_paths)} frames")
    finally:
        writer.release()

    print(f"Frame dir : {args.frame_dir}")
    print(f"Frames    : {len(frame_paths)}")
    print(f"FPS       : {args.fps}")
    print(f"Video     : {output_path}")


if __name__ == "__main__":
    main()
