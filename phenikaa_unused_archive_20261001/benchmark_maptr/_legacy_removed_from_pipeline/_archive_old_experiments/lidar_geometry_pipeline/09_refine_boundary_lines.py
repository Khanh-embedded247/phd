#!/usr/bin/env python3
"""
Bước 09: lọc, nối và làm mượt boundary/curb thô từ bước 08.

Input:
    outputs/benchmark/Normal/road_boundary_curb/boundary_curb_candidates_map.npz

Output:
    outputs/benchmark/Normal/boundary_refined/
        refined_boundaries_map.npz
        refined_boundaries.json
        report.json
        bev_refined_boundaries.png

Ý tưởng:
- Bước 08 cho ta các điểm ứng viên mép đường/curb, nhưng còn vụn và nhiễu.
- Mỗi candidate đã có `local_xyz_base` từ bước 07, tức tọa độ trong base_link
  tại frame LiDAR thật. Vì vậy ta dùng y_base để tách trái/phải.
- Quỹ đạo chỉ dùng để tạo trục dọc đường `s`, giúp sắp thứ tự và gom theo từng mét.
- Trong mỗi đoạn dọc đường, lấy một điểm boundary đại diện bằng thống kê bền vững.
- Bỏ điểm nhảy ngang bất thường, nối gap ngắn, rồi làm mượt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import medfilt, savgol_filter
from scipy.spatial import cKDTree

from common import (
    DEFAULT_LIDAR,
    DEFAULT_SCENARIO,
    EXTRINSIC_JSON,
    OUTPUT_ROOT,
    ensure_output_dir,
    load_json,
    load_trajectory,
)


# ============================================================
# USER CONFIG
# ============================================================

CONFIG_SCENARIO = DEFAULT_SCENARIO
CONFIG_LIDAR = DEFAULT_LIDAR
CONFIG_INPUT_NPZ = (
    OUTPUT_ROOT
    / CONFIG_SCENARIO
    / "road_boundary_curb"
    / "boundary_curb_candidates_map.npz"
)
CONFIG_TRAJ = (
    OUTPUT_ROOT.parents[1]
    / "data"
    / CONFIG_SCENARIO
    / "dump"
    / "traj_lidar.txt"
)

# Bỏ vùng quá gần tâm xe và quá xa xe. Đây là cổng lọc rộng, không ép bề rộng đường.
CONFIG_MIN_ABS_Y_BASE = 0.8
CONFIG_MAX_ABS_Y_BASE = 22.0

# Gom candidate theo đoạn dọc quỹ đạo.
CONFIG_BIN_SIZE_M = 1.0
CONFIG_MIN_POINTS_PER_BIN = 4

# Với phía trái lấy y lớn hơn, phía phải lấy y nhỏ hơn.
CONFIG_LEFT_Y_PERCENTILE = 75.0
CONFIG_RIGHT_Y_PERCENTILE = 25.0

# Bỏ các đại diện boundary nhảy ngang quá mạnh so với lân cận để tránh outlier.
CONFIG_MEDIAN_FILTER_KERNEL = 7
CONFIG_MAX_Y_DEVIATION_M = 3.0

# Nối gap ngắn, gap dài giữ nguyên là đứt đoạn.
CONFIG_MAX_INTERPOLATE_GAP_M = 6.0

# Làm mượt đường sau khi đã nối gap ngắn.
CONFIG_SAVGOL_WINDOW = 15
CONFIG_SAVGOL_POLYORDER = 2

# Bỏ segment quá ngắn sau refine.
CONFIG_MIN_SEGMENT_LENGTH_M = 8.0
CONFIG_MIN_SEGMENT_POINTS = 8

# Ảnh BEV kiểm tra.
CONFIG_BEV_RESOLUTION = 0.20
CONFIG_MAX_BEV_PIXELS = 25_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=CONFIG_SCENARIO)
    parser.add_argument("--lidar", default=CONFIG_LIDAR)
    parser.add_argument("--input-npz", type=Path, default=CONFIG_INPUT_NPZ)
    parser.add_argument("--traj", type=Path, default=CONFIG_TRAJ)
    parser.add_argument("--min-abs-y-base", type=float, default=CONFIG_MIN_ABS_Y_BASE)
    parser.add_argument("--max-abs-y-base", type=float, default=CONFIG_MAX_ABS_Y_BASE)
    parser.add_argument("--bin-size-m", type=float, default=CONFIG_BIN_SIZE_M)
    parser.add_argument("--min-points-per-bin", type=int, default=CONFIG_MIN_POINTS_PER_BIN)
    parser.add_argument("--left-y-percentile", type=float, default=CONFIG_LEFT_Y_PERCENTILE)
    parser.add_argument("--right-y-percentile", type=float, default=CONFIG_RIGHT_Y_PERCENTILE)
    parser.add_argument("--median-filter-kernel", type=int, default=CONFIG_MEDIAN_FILTER_KERNEL)
    parser.add_argument("--max-y-deviation-m", type=float, default=CONFIG_MAX_Y_DEVIATION_M)
    parser.add_argument("--max-interpolate-gap-m", type=float, default=CONFIG_MAX_INTERPOLATE_GAP_M)
    parser.add_argument("--savgol-window", type=int, default=CONFIG_SAVGOL_WINDOW)
    parser.add_argument("--savgol-polyorder", type=int, default=CONFIG_SAVGOL_POLYORDER)
    parser.add_argument("--min-segment-length-m", type=float, default=CONFIG_MIN_SEGMENT_LENGTH_M)
    parser.add_argument("--min-segment-points", type=int, default=CONFIG_MIN_SEGMENT_POINTS)
    parser.add_argument("--bev-resolution", type=float, default=CONFIG_BEV_RESOLUTION)
    parser.add_argument("--max-bev-pixels", type=int, default=CONFIG_MAX_BEV_PIXELS)
    return parser.parse_args()


def load_base_trajectory(traj_path: Path, lidar_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Đọc traj LiDAR và suy ra vị trí base_link trong hệ map."""
    trajectory = load_trajectory(traj_path)
    T_base_lidar = np.asarray(load_json(EXTRINSIC_JSON)[lidar_name], dtype=np.float64)
    T_lidar_base = np.linalg.inv(T_base_lidar)
    T_map_base = np.stack([pose.T_map_lidar @ T_lidar_base for pose in trajectory.poses], axis=0)
    base_xy = T_map_base[:, :2, 3].astype(np.float64)
    step = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(step)))
    return base_xy, s


def assign_progress_s(points_map: np.ndarray, base_xy: np.ndarray, base_s: np.ndarray) -> np.ndarray:
    """Gán mỗi candidate vào đoạn gần nhất trên quỹ đạo để có tọa độ dọc đường s."""
    tree = cKDTree(base_xy)
    _, idx = tree.query(points_map[:, :2], k=1, workers=-1)
    return base_s[idx]


def boundary_representatives(
    points_map: np.ndarray,
    local_xyz: np.ndarray,
    progress_s: np.ndarray,
    side_mask: np.ndarray,
    side: str,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Gom candidate theo bin dọc đường và lấy điểm đại diện từng bin."""
    pts = points_map[side_mask]
    local = local_xyz[side_mask]
    s = progress_s[side_mask]
    if len(pts) == 0:
        return empty_boundary()

    s_min = float(np.floor(s.min() / args.bin_size_m) * args.bin_size_m)
    bin_id = np.floor((s - s_min) / args.bin_size_m).astype(np.int64)
    reps = []
    percentile = args.left_y_percentile if side == "left" else args.right_y_percentile

    for bid in np.unique(bin_id):
        ids = np.flatnonzero(bin_id == bid)
        if len(ids) < args.min_points_per_bin:
            continue
        y = local[ids, 1]
        target_y = np.percentile(y, percentile)

        # Lấy nhóm gần target_y để giảm ảnh hưởng point lẻ xa bất thường.
        dist_y = np.abs(y - target_y)
        keep_count = max(args.min_points_per_bin, int(np.ceil(0.35 * len(ids))))
        near_local = ids[np.argsort(dist_y)[:keep_count]]

        rep_map = np.median(pts[near_local], axis=0)
        rep_local = np.median(local[near_local], axis=0)
        reps.append(
            [
                s_min + (bid + 0.5) * args.bin_size_m,
                rep_map[0],
                rep_map[1],
                rep_map[2],
                rep_local[0],
                rep_local[1],
                rep_local[2],
                len(ids),
            ]
        )

    if not reps:
        return empty_boundary()
    arr = np.asarray(reps, dtype=np.float64)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return {
        "s": arr[:, 0],
        "points_map": arr[:, 1:4],
        "local_xyz_base": arr[:, 4:7],
        "count": arr[:, 7].astype(np.int32),
        "observed": np.ones(len(arr), dtype=bool),
    }


def empty_boundary() -> dict[str, np.ndarray]:
    """Boundary rỗng dùng khi một phía không có dữ liệu."""
    return {
        "s": np.empty((0,), dtype=np.float64),
        "points_map": np.empty((0, 3), dtype=np.float64),
        "local_xyz_base": np.empty((0, 3), dtype=np.float64),
        "count": np.empty((0,), dtype=np.int32),
        "observed": np.empty((0,), dtype=bool),
    }


def remove_y_outliers(boundary: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Bỏ các điểm đại diện có y_base nhảy bất thường so với median lân cận."""
    n = len(boundary["s"])
    if n < max(args.median_filter_kernel, 3):
        return boundary

    kernel = args.median_filter_kernel
    if kernel % 2 == 0:
        kernel += 1
    kernel = min(kernel, n if n % 2 == 1 else n - 1)
    if kernel < 3:
        return boundary

    y = boundary["local_xyz_base"][:, 1]
    y_med = medfilt(y, kernel_size=kernel)
    keep = np.abs(y - y_med) <= args.max_y_deviation_m
    return {key: value[keep] for key, value in boundary.items()}


def interpolate_small_gaps(boundary: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Nối các gap ngắn bằng nội suy tuyến tính, đồng thời đánh dấu observed/interpolated."""
    if len(boundary["s"]) < 2:
        return boundary

    s = boundary["s"]
    points = boundary["points_map"]
    local = boundary["local_xyz_base"]
    count = boundary["count"]

    out_s = [s[0]]
    out_points = [points[0]]
    out_local = [local[0]]
    out_count = [count[0]]
    out_observed = [True]

    max_gap_bins = max(1, int(np.floor(args.max_interpolate_gap_m / args.bin_size_m)))
    for i in range(1, len(s)):
        gap_bins = int(round((s[i] - s[i - 1]) / args.bin_size_m)) - 1
        if 0 < gap_bins <= max_gap_bins:
            for j in range(1, gap_bins + 1):
                alpha = j / (gap_bins + 1)
                out_s.append((1.0 - alpha) * s[i - 1] + alpha * s[i])
                out_points.append((1.0 - alpha) * points[i - 1] + alpha * points[i])
                out_local.append((1.0 - alpha) * local[i - 1] + alpha * local[i])
                out_count.append(0)
                out_observed.append(False)
        out_s.append(s[i])
        out_points.append(points[i])
        out_local.append(local[i])
        out_count.append(count[i])
        out_observed.append(True)

    return {
        "s": np.asarray(out_s, dtype=np.float64),
        "points_map": np.asarray(out_points, dtype=np.float64),
        "local_xyz_base": np.asarray(out_local, dtype=np.float64),
        "count": np.asarray(out_count, dtype=np.int32),
        "observed": np.asarray(out_observed, dtype=bool),
    }


def continuous_segments(s: np.ndarray, max_step: float) -> list[slice]:
    """Tách boundary thành các segment liên tục theo s."""
    if len(s) == 0:
        return []
    breaks = np.flatnonzero(np.diff(s) > max_step)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [len(s)]))
    return [slice(int(a), int(b)) for a, b in zip(starts, ends)]


def smooth_segments(boundary: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Làm mượt từng segment bằng Savitzky-Golay, bỏ segment quá ngắn."""
    if len(boundary["s"]) == 0:
        return boundary

    keep_blocks = []
    for seg in continuous_segments(boundary["s"], args.max_interpolate_gap_m + args.bin_size_m * 1.5):
        seg_len = seg.stop - seg.start
        if seg_len < args.min_segment_points:
            continue
        length_m = boundary["s"][seg.stop - 1] - boundary["s"][seg.start]
        if length_m < args.min_segment_length_m:
            continue
        keep_blocks.append(seg)

    if not keep_blocks:
        return empty_boundary()

    smoothed = {key: [] for key in boundary}
    for seg in keep_blocks:
        for key, value in boundary.items():
            block = value[seg].copy()
            if key in {"points_map", "local_xyz_base"}:
                window = args.savgol_window
                if window % 2 == 0:
                    window += 1
                max_window = len(block) if len(block) % 2 == 1 else len(block) - 1
                window = min(window, max_window)
                if window > args.savgol_polyorder + 2 and window >= 5:
                    block = savgol_filter(
                        block,
                        window_length=window,
                        polyorder=args.savgol_polyorder,
                        axis=0,
                        mode="interp",
                    )
            smoothed[key].append(block)

    return {key: np.concatenate(blocks, axis=0) for key, blocks in smoothed.items()}


def refine_one_side(
    points_map: np.ndarray,
    local_xyz: np.ndarray,
    progress_s: np.ndarray,
    side_mask: np.ndarray,
    side: str,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Chạy toàn bộ refine cho một phía trái/phải."""
    boundary = boundary_representatives(points_map, local_xyz, progress_s, side_mask, side, args)
    boundary = remove_y_outliers(boundary, args)
    boundary = interpolate_small_gaps(boundary, args)
    boundary = smooth_segments(boundary, args)
    return boundary


def boundary_to_json(name: str, boundary: dict[str, np.ndarray]) -> dict:
    """Đổi boundary sang dict JSON nhẹ để xem nhanh."""
    points = boundary["points_map"]
    return {
        "name": name,
        "coordinate_frame": "map",
        "point_count": int(len(points)),
        "observed_count": int(boundary["observed"].sum()) if len(points) else 0,
        "interpolated_count": int((~boundary["observed"]).sum()) if len(points) else 0,
        "xyz_map": points.tolist(),
        "s": boundary["s"].tolist(),
        "observed": boundary["observed"].astype(bool).tolist(),
    }


def draw_refined_bev(
    out_path: Path,
    candidates_map: np.ndarray,
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    resolution: float,
    max_pixels: int,
) -> dict:
    """Vẽ ảnh BEV: candidate nền xám, left đỏ, right xanh."""
    all_points = candidates_map
    min_xy = all_points[:, :2].min(axis=0)
    max_xy = all_points[:, :2].max(axis=0)
    span = max_xy - min_xy
    width = int(np.ceil(span[0] / resolution)) + 1
    height = int(np.ceil(span[1] / resolution)) + 1
    used_resolution = resolution
    if width * height > max_pixels:
        scale = np.sqrt((width * height) / max_pixels)
        used_resolution = resolution * scale
        width = int(np.ceil(span[0] / used_resolution)) + 1
        height = int(np.ceil(span[1] / used_resolution)) + 1

    img = np.zeros((height, width, 3), dtype=np.uint8)
    cols = np.floor((all_points[:, 0] - min_xy[0]) / used_resolution).astype(np.int32)
    rows = height - 1 - np.floor((all_points[:, 1] - min_xy[1]) / used_resolution).astype(np.int32)
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    img[rows[valid], cols[valid]] = (65, 65, 65)

    def to_pixel(points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.int32)
        col = np.floor((points[:, 0] - min_xy[0]) / used_resolution).astype(np.int32)
        row = height - 1 - np.floor((points[:, 1] - min_xy[1]) / used_resolution).astype(np.int32)
        return np.column_stack((col, row))

    for boundary, color in ((left, (0, 0, 255)), (right, (255, 80, 0))):
        pix = to_pixel(boundary["points_map"])
        if len(pix) >= 2:
            cv2.polylines(img, [pix.reshape(-1, 1, 2)], False, color, 2, lineType=cv2.LINE_AA)
            for p, observed in zip(pix, boundary["observed"]):
                dot_color = color if observed else (0, 255, 255)
                cv2.circle(img, tuple(int(v) for v in p), 2, dot_color, -1)

    cv2.imwrite(str(out_path), img)
    return {
        "min_xy": min_xy.tolist(),
        "max_xy": max_xy.tolist(),
        "resolution_used": float(used_resolution),
        "bev_width": int(width),
        "bev_height": int(height),
    }


def main() -> None:
    args = parse_args()
    if not args.input_npz.is_file():
        raise FileNotFoundError(args.input_npz)
    if not args.traj.is_file():
        raise FileNotFoundError(args.traj)

    out_dir = ensure_output_dir(args.scenario, "boundary_refined")
    data = np.load(args.input_npz)
    points_map = data["points_map"].astype(np.float64)
    local_xyz = data["local_xyz_base"].astype(np.float64)

    base_xy, base_s = load_base_trajectory(args.traj, args.lidar)
    progress_s = assign_progress_s(points_map, base_xy, base_s)

    y = local_xyz[:, 1]
    abs_y = np.abs(y)
    valid_width = (abs_y >= args.min_abs_y_base) & (abs_y <= args.max_abs_y_base)
    left_mask = valid_width & (y > 0.0)
    right_mask = valid_width & (y < 0.0)

    left = refine_one_side(points_map, local_xyz, progress_s, left_mask, "left", args)
    right = refine_one_side(points_map, local_xyz, progress_s, right_mask, "right", args)

    npz_path = out_dir / "refined_boundaries_map.npz"
    np.savez_compressed(
        npz_path,
        left_points_map=left["points_map"].astype(np.float32),
        left_s=left["s"].astype(np.float32),
        left_observed=left["observed"],
        right_points_map=right["points_map"].astype(np.float32),
        right_s=right["s"].astype(np.float32),
        right_observed=right["observed"],
    )

    json_path = out_dir / "refined_boundaries.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "boundary_type": "refined_road_boundary_candidate",
                "left": boundary_to_json("left", left),
                "right": boundary_to_json("right", right),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    bev_path = out_dir / "bev_refined_boundaries.png"
    bev_info = draw_refined_bev(
        bev_path,
        points_map,
        left,
        right,
        args.bev_resolution,
        args.max_bev_pixels,
    )

    report = {
        "input_npz": str(args.input_npz),
        "candidate_point_count": int(len(points_map)),
        "left_candidate_count": int(left_mask.sum()),
        "right_candidate_count": int(right_mask.sum()),
        "left_refined_point_count": int(len(left["points_map"])),
        "right_refined_point_count": int(len(right["points_map"])),
        "thresholds": {
            "min_abs_y_base": float(args.min_abs_y_base),
            "max_abs_y_base": float(args.max_abs_y_base),
            "bin_size_m": float(args.bin_size_m),
            "min_points_per_bin": int(args.min_points_per_bin),
            "left_y_percentile": float(args.left_y_percentile),
            "right_y_percentile": float(args.right_y_percentile),
            "max_y_deviation_m": float(args.max_y_deviation_m),
            "max_interpolate_gap_m": float(args.max_interpolate_gap_m),
            "min_segment_length_m": float(args.min_segment_length_m),
        },
        "bev": bev_info,
        "outputs": {
            "refined_npz": str(npz_path),
            "refined_json": str(json_path),
            "bev_refined_boundaries": str(bev_path),
        },
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Candidates        : {len(points_map):,}")
    print(f"Left candidates   : {left_mask.sum():,}")
    print(f"Right candidates  : {right_mask.sum():,}")
    print(f"Left refined      : {len(left['points_map']):,}")
    print(f"Right refined     : {len(right['points_map']):,}")
    print(f"Output dir        : {out_dir}")


if __name__ == "__main__":
    main()
