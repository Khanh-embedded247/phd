#!/usr/bin/env python3
"""
Bước 10: tạo lanelet thô và centerline từ boundary trái/phải đã refine.

Input:
    outputs/benchmark/Normal/boundary_refined/refined_boundaries_map.npz

Output:
    outputs/benchmark/Normal/lanelet_draft/
        lanelet_draft_map.npz
        lanelet_draft.json
        lanelet_draft.geojson
        bev_lanelet_draft.png
        report.json

Ý tưởng:
- Bước 09 đã tạo boundary trái/phải theo cùng đại lượng `s` dọc quỹ đạo.
- Bước 10 lấy đoạn `s` mà cả trái và phải đều có dữ liệu.
- Nội suy boundary trái/phải tại cùng các giá trị `s`.
- Centerline = trung điểm của boundary trái và phải.
- Width = khoảng cách trái-phải.
- Đây là lanelet thô, chưa phải Lanelet2 OSM cuối cùng.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

from common import DEFAULT_SCENARIO, OUTPUT_ROOT, ensure_output_dir


# ============================================================
# USER CONFIG
# ============================================================

CONFIG_SCENARIO = DEFAULT_SCENARIO
CONFIG_INPUT_NPZ = (
    OUTPUT_ROOT
    / CONFIG_SCENARIO
    / "boundary_refined"
    / "refined_boundaries_map.npz"
)

# Khoảng cách giữa các điểm vector đầu ra.
CONFIG_SAMPLE_STEP_M = 1.0

# Giới hạn width hợp lý để bỏ đoạn trái/phải ghép sai.
CONFIG_MIN_LANELET_WIDTH_M = 3.0
CONFIG_MAX_LANELET_WIDTH_M = 35.0

# Smooth centerline/boundary sau khi resample.
CONFIG_SMOOTH_WINDOW = 21
CONFIG_SMOOTH_POLYORDER = 2

# Bỏ segment lanelet quá ngắn.
CONFIG_MIN_SEGMENT_LENGTH_M = 20.0
CONFIG_MIN_SEGMENT_POINTS = 15

# Ảnh BEV kiểm tra.
CONFIG_BEV_RESOLUTION = 0.20
CONFIG_MAX_BEV_PIXELS = 25_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=CONFIG_SCENARIO)
    parser.add_argument("--input-npz", type=Path, default=CONFIG_INPUT_NPZ)
    parser.add_argument("--sample-step-m", type=float, default=CONFIG_SAMPLE_STEP_M)
    parser.add_argument("--min-lanelet-width-m", type=float, default=CONFIG_MIN_LANELET_WIDTH_M)
    parser.add_argument("--max-lanelet-width-m", type=float, default=CONFIG_MAX_LANELET_WIDTH_M)
    parser.add_argument("--smooth-window", type=int, default=CONFIG_SMOOTH_WINDOW)
    parser.add_argument("--smooth-polyorder", type=int, default=CONFIG_SMOOTH_POLYORDER)
    parser.add_argument("--min-segment-length-m", type=float, default=CONFIG_MIN_SEGMENT_LENGTH_M)
    parser.add_argument("--min-segment-points", type=int, default=CONFIG_MIN_SEGMENT_POINTS)
    parser.add_argument("--bev-resolution", type=float, default=CONFIG_BEV_RESOLUTION)
    parser.add_argument("--max-bev-pixels", type=int, default=CONFIG_MAX_BEV_PIXELS)
    return parser.parse_args()


def interp_polyline(s_src: np.ndarray, xyz_src: np.ndarray, s_dst: np.ndarray) -> np.ndarray:
    """Nội suy polyline 3D theo s."""
    out = np.column_stack(
        [
            np.interp(s_dst, s_src, xyz_src[:, 0]),
            np.interp(s_dst, s_src, xyz_src[:, 1]),
            np.interp(s_dst, s_src, xyz_src[:, 2]),
        ]
    )
    return out.astype(np.float64)


def interp_observed(s_src: np.ndarray, observed_src: np.ndarray, s_dst: np.ndarray) -> np.ndarray:
    """Gán trạng thái observed gần nhất cho s mới."""
    idx = np.searchsorted(s_src, s_dst, side="left")
    idx = np.clip(idx, 0, len(s_src) - 1)
    left_idx = np.clip(idx - 1, 0, len(s_src) - 1)
    choose_left = np.abs(s_dst - s_src[left_idx]) <= np.abs(s_dst - s_src[idx])
    nearest = np.where(choose_left, left_idx, idx)
    return observed_src[nearest].astype(bool)


def smooth_xyz(xyz: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Làm mượt polyline, giữ nguyên nếu segment quá ngắn."""
    if len(xyz) < max(5, polyorder + 3):
        return xyz
    if window % 2 == 0:
        window += 1
    max_window = len(xyz) if len(xyz) % 2 == 1 else len(xyz) - 1
    window = min(window, max_window)
    if window < polyorder + 3 or window < 5:
        return xyz
    return savgol_filter(xyz, window_length=window, polyorder=polyorder, axis=0, mode="interp")


def continuous_segments(mask: np.ndarray, s: np.ndarray, max_step: float) -> list[slice]:
    """Tách các đoạn liên tục theo mask hợp lệ và khoảng cách s."""
    valid_idx = np.flatnonzero(mask)
    if len(valid_idx) == 0:
        return []
    breaks = []
    for i in range(1, len(valid_idx)):
        prev_i = valid_idx[i - 1]
        cur_i = valid_idx[i]
        if cur_i != prev_i + 1 or (s[cur_i] - s[prev_i]) > max_step:
            breaks.append(i)
    starts = np.concatenate(([0], np.asarray(breaks, dtype=np.int64)))
    ends = np.concatenate((np.asarray(breaks, dtype=np.int64), [len(valid_idx)]))
    return [slice(int(valid_idx[a]), int(valid_idx[b - 1]) + 1) for a, b in zip(starts, ends)]


def build_lanelet_segments(
    s: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    left_observed: np.ndarray,
    right_observed: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, np.ndarray]]:
    """Tạo các segment lanelet hợp lệ từ boundary trái/phải đã đồng bộ."""
    width = np.linalg.norm(left[:, :2] - right[:, :2], axis=1)
    valid = (
        np.isfinite(left).all(axis=1)
        & np.isfinite(right).all(axis=1)
        & (width >= args.min_lanelet_width_m)
        & (width <= args.max_lanelet_width_m)
    )

    segments = []
    for seg in continuous_segments(valid, s, args.sample_step_m * 1.5):
        seg_s = s[seg]
        if len(seg_s) < args.min_segment_points:
            continue
        if seg_s[-1] - seg_s[0] < args.min_segment_length_m:
            continue

        left_seg = smooth_xyz(left[seg], args.smooth_window, args.smooth_polyorder)
        right_seg = smooth_xyz(right[seg], args.smooth_window, args.smooth_polyorder)
        center_seg = smooth_xyz((left_seg + right_seg) * 0.5, args.smooth_window, args.smooth_polyorder)
        width_seg = np.linalg.norm(left_seg[:, :2] - right_seg[:, :2], axis=1)
        observed_seg = left_observed[seg] & right_observed[seg]

        segments.append(
            {
                "s": seg_s,
                "left_boundary_map": left_seg,
                "right_boundary_map": right_seg,
                "centerline_map": center_seg,
                "width_m": width_seg,
                "observed": observed_seg,
            }
        )
    return segments


def segments_to_json(segments: list[dict[str, np.ndarray]]) -> dict:
    """Đổi lanelet segment sang JSON dễ đọc."""
    items = []
    for idx, seg in enumerate(segments):
        items.append(
            {
                "id": idx,
                "coordinate_frame": "map",
                "point_count": int(len(seg["s"])),
                "length_m": float(seg["s"][-1] - seg["s"][0]) if len(seg["s"]) else 0.0,
                "width_mean_m": float(np.mean(seg["width_m"])) if len(seg["width_m"]) else 0.0,
                "width_min_m": float(np.min(seg["width_m"])) if len(seg["width_m"]) else 0.0,
                "width_max_m": float(np.max(seg["width_m"])) if len(seg["width_m"]) else 0.0,
                "observed_ratio": float(np.mean(seg["observed"])) if len(seg["observed"]) else 0.0,
                "left_boundary_map": seg["left_boundary_map"].tolist(),
                "right_boundary_map": seg["right_boundary_map"].tolist(),
                "centerline_map": seg["centerline_map"].tolist(),
                "s": seg["s"].tolist(),
                "observed": seg["observed"].astype(bool).tolist(),
            }
        )
    return {
        "coordinate_frame": "map",
        "lanelet_type": "draft_from_refined_boundaries",
        "count": len(items),
        "lanelets": items,
    }


def segments_to_geojson(segments: list[dict[str, np.ndarray]]) -> dict:
    """
    Xuất GeoJSON trong tọa độ map XY.

    Đây chưa phải tọa độ GPS, chỉ là GeoJSON tiện mở bằng tool GIS/plot.
    """
    features = []
    for idx, seg in enumerate(segments):
        for name, key in (
            ("left_boundary", "left_boundary_map"),
            ("right_boundary", "right_boundary_map"),
            ("centerline", "centerline_map"),
        ):
            coords = seg[key][:, :2].tolist()
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "lanelet_id": idx,
                        "role": name,
                        "frame": "map_xy_not_lonlat",
                        "width_mean_m": float(np.mean(seg["width_m"])),
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords,
                    },
                }
            )
    return {
        "type": "FeatureCollection",
        "name": "lanelet_draft_map_xy",
        "features": features,
    }


def draw_bev(out_path: Path, segments: list[dict[str, np.ndarray]], resolution: float, max_pixels: int) -> dict:
    """Vẽ BEV lanelet: trái đỏ, phải xanh, center vàng."""
    if not segments:
        raise RuntimeError("No lanelet segment to draw")

    all_points = np.concatenate(
        [
            np.concatenate(
                [seg["left_boundary_map"], seg["right_boundary_map"], seg["centerline_map"]],
                axis=0,
            )
            for seg in segments
        ],
        axis=0,
    )
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

    def to_pixel(points: np.ndarray) -> np.ndarray:
        col = np.floor((points[:, 0] - min_xy[0]) / used_resolution).astype(np.int32)
        row = height - 1 - np.floor((points[:, 1] - min_xy[1]) / used_resolution).astype(np.int32)
        return np.column_stack((col, row))

    for seg in segments:
        for key, color, thickness in (
            ("left_boundary_map", (0, 0, 255), 2),
            ("right_boundary_map", (255, 80, 0), 2),
            ("centerline_map", (0, 255, 255), 1),
        ):
            pix = to_pixel(seg[key])
            cv2.polylines(img, [pix.reshape(-1, 1, 2)], False, color, thickness, lineType=cv2.LINE_AA)

        # Vẽ vài đường ngang để nhìn cặp trái/phải ghép với nhau.
        step = max(1, len(seg["s"]) // 80)
        left_pix = to_pixel(seg["left_boundary_map"])
        right_pix = to_pixel(seg["right_boundary_map"])
        for lpt, rpt in zip(left_pix[::step], right_pix[::step]):
            cv2.line(img, tuple(lpt), tuple(rpt), (50, 50, 50), 1, lineType=cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    return {
        "min_xy": min_xy.tolist(),
        "max_xy": max_xy.tolist(),
        "resolution_used": float(used_resolution),
        "bev_width": int(width),
        "bev_height": int(height),
    }


def save_npz(path: Path, segments: list[dict[str, np.ndarray]]) -> None:
    """Lưu nhiều segment vào NPZ bằng key có id."""
    save_data = {}
    for idx, seg in enumerate(segments):
        prefix = f"lanelet_{idx:03d}"
        save_data[f"{prefix}_s"] = seg["s"].astype(np.float32)
        save_data[f"{prefix}_left_boundary_map"] = seg["left_boundary_map"].astype(np.float32)
        save_data[f"{prefix}_right_boundary_map"] = seg["right_boundary_map"].astype(np.float32)
        save_data[f"{prefix}_centerline_map"] = seg["centerline_map"].astype(np.float32)
        save_data[f"{prefix}_width_m"] = seg["width_m"].astype(np.float32)
        save_data[f"{prefix}_observed"] = seg["observed"].astype(bool)
    save_data["lanelet_count"] = np.asarray([len(segments)], dtype=np.int32)
    np.savez_compressed(path, **save_data)


def main() -> None:
    args = parse_args()
    if not args.input_npz.is_file():
        raise FileNotFoundError(args.input_npz)

    data = np.load(args.input_npz)
    left_s = data["left_s"].astype(np.float64)
    right_s = data["right_s"].astype(np.float64)
    left_points = data["left_points_map"].astype(np.float64)
    right_points = data["right_points_map"].astype(np.float64)
    left_observed = data["left_observed"].astype(bool)
    right_observed = data["right_observed"].astype(bool)

    s_start = max(float(left_s.min()), float(right_s.min()))
    s_end = min(float(left_s.max()), float(right_s.max()))
    if s_end <= s_start:
        raise RuntimeError("Left/right boundaries do not overlap in s")

    common_s = np.arange(s_start, s_end + args.sample_step_m * 0.5, args.sample_step_m)
    left_interp = interp_polyline(left_s, left_points, common_s)
    right_interp = interp_polyline(right_s, right_points, common_s)
    left_obs = interp_observed(left_s, left_observed, common_s)
    right_obs = interp_observed(right_s, right_observed, common_s)

    segments = build_lanelet_segments(common_s, left_interp, right_interp, left_obs, right_obs, args)
    if not segments:
        raise RuntimeError("No valid lanelet segment. Try widening min/max width thresholds.")

    out_dir = ensure_output_dir(args.scenario, "lanelet_draft")
    npz_path = out_dir / "lanelet_draft_map.npz"
    json_path = out_dir / "lanelet_draft.json"
    geojson_path = out_dir / "lanelet_draft.geojson"
    bev_path = out_dir / "bev_lanelet_draft.png"

    save_npz(npz_path, segments)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(segments_to_json(segments), f, indent=2, ensure_ascii=False)
    with geojson_path.open("w", encoding="utf-8") as f:
        json.dump(segments_to_geojson(segments), f, indent=2, ensure_ascii=False)
    bev_info = draw_bev(bev_path, segments, args.bev_resolution, args.max_bev_pixels)

    lengths = [float(seg["s"][-1] - seg["s"][0]) for seg in segments]
    widths = np.concatenate([seg["width_m"] for seg in segments])
    observed = np.concatenate([seg["observed"] for seg in segments])
    report = {
        "input_npz": str(args.input_npz),
        "common_s_start": float(s_start),
        "common_s_end": float(s_end),
        "sample_step_m": float(args.sample_step_m),
        "lanelet_count": int(len(segments)),
        "total_length_m": float(np.sum(lengths)),
        "width_mean_m": float(np.mean(widths)),
        "width_min_m": float(np.min(widths)),
        "width_max_m": float(np.max(widths)),
        "observed_ratio": float(np.mean(observed)),
        "thresholds": {
            "min_lanelet_width_m": float(args.min_lanelet_width_m),
            "max_lanelet_width_m": float(args.max_lanelet_width_m),
            "min_segment_length_m": float(args.min_segment_length_m),
            "min_segment_points": int(args.min_segment_points),
        },
        "bev": bev_info,
        "outputs": {
            "npz": str(npz_path),
            "json": str(json_path),
            "geojson": str(geojson_path),
            "bev": str(bev_path),
        },
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Lanelets       : {len(segments):,}")
    print(f"Total length   : {np.sum(lengths):.2f} m")
    print(f"Width mean     : {np.mean(widths):.2f} m")
    print(f"Width min/max  : {np.min(widths):.2f} / {np.max(widths):.2f} m")
    print(f"Observed ratio : {np.mean(observed):.3f}")
    print(f"Output dir     : {out_dir}")


if __name__ == "__main__":
    main()
