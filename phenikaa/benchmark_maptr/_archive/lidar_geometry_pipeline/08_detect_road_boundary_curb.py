#!/usr/bin/env python3
"""
Bước 08: tìm ứng viên mép đường / curb từ điểm mặt đường đã lọc ở bước 07.

Input:
    outputs/benchmark/Normal/ground_road_base_link/ground_road_points_map.npz

Output:
    outputs/benchmark/Normal/road_boundary_curb/
        boundary_curb_candidates_map.npz
        boundary_polylines_map.json
        report.json
        bev_road_mask.png
        bev_boundary_candidate.png
        bev_boundary_on_intensity.png

Ý tưởng:
- Bước 07 đã giữ lại điểm mặt đất/mặt đường trong hệ `map`.
- Bước 08 chuyển các điểm này sang ảnh BEV dạng lưới.
- Ô nào có đủ điểm thì coi là vùng mặt đường quan sát được.
- Mép đường/curb thường nằm ở rìa vùng mặt đường hoặc nơi z_base thay đổi.
- Kết quả của bước này là candidate thô để xem và chỉnh tiếp, chưa phải lanelet cuối.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import DEFAULT_SCENARIO, OUTPUT_ROOT, ensure_output_dir


# ============================================================
# USER CONFIG
# ============================================================

CONFIG_SCENARIO = DEFAULT_SCENARIO
CONFIG_INPUT_NPZ = (
    OUTPUT_ROOT
    / CONFIG_SCENARIO
    / "ground_road_base_link"
    / "ground_road_points_map.npz"
)

# Độ phân giải BEV cho bước tìm boundary.
CONFIG_BEV_RESOLUTION = 0.15
CONFIG_MAX_BEV_PIXELS = 25_000_000

# Một ô BEV cần tối thiểu bao nhiêu điểm để coi là có mặt đường.
CONFIG_MIN_POINTS_PER_CELL = 1

# Làm kín mask mặt đường để giảm lỗ nhỏ do point cloud thưa.
CONFIG_CLOSE_KERNEL_SIZE = 5
CONFIG_OPEN_KERNEL_SIZE = 3

# Giữ contour đủ dài, bỏ các mảnh vụn nhỏ.
CONFIG_MIN_CONTOUR_POINTS = 30
CONFIG_MIN_CONTOUR_LENGTH_M = 5.0

# Nếu z_base thay đổi nhiều quanh vùng rìa thì tăng độ tin cậy curb.
CONFIG_Z_EDGE_PERCENTILE = 85.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=CONFIG_SCENARIO)
    parser.add_argument("--input-npz", type=Path, default=CONFIG_INPUT_NPZ)
    parser.add_argument("--resolution", type=float, default=CONFIG_BEV_RESOLUTION)
    parser.add_argument("--max-bev-pixels", type=int, default=CONFIG_MAX_BEV_PIXELS)
    parser.add_argument("--min-points-per-cell", type=int, default=CONFIG_MIN_POINTS_PER_CELL)
    parser.add_argument("--close-kernel-size", type=int, default=CONFIG_CLOSE_KERNEL_SIZE)
    parser.add_argument("--open-kernel-size", type=int, default=CONFIG_OPEN_KERNEL_SIZE)
    parser.add_argument("--min-contour-points", type=int, default=CONFIG_MIN_CONTOUR_POINTS)
    parser.add_argument("--min-contour-length-m", type=float, default=CONFIG_MIN_CONTOUR_LENGTH_M)
    parser.add_argument("--z-edge-percentile", type=float, default=CONFIG_Z_EDGE_PERCENTILE)
    return parser.parse_args()


def normalize_to_u8(values: np.ndarray, clip_percentile: tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    """Chuẩn hóa ảnh float thành uint8 để lưu PNG."""
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


def build_bev_grid(
    points_map: np.ndarray,
    z_base: np.ndarray,
    intensity: np.ndarray | None,
    resolution: float,
    max_pixels: int,
) -> tuple[dict[str, np.ndarray], dict, np.ndarray, np.ndarray, np.ndarray]:
    """Đưa point map thành lưới BEV và trả về index ô của từng point."""
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
    z_min = np.full((height, width), np.inf, dtype=np.float32)
    z_max = np.full((height, width), -np.inf, dtype=np.float32)

    np.add.at(density, (row, col), 1.0)
    np.add.at(z_sum, (row, col), z_base.astype(np.float64))
    np.add.at(z_count, (row, col), 1.0)
    np.minimum.at(z_min, (row, col), z_base.astype(np.float32))
    np.maximum.at(z_max, (row, col), z_base.astype(np.float32))

    z_mean = np.divide(
        z_sum,
        z_count,
        out=np.full_like(z_sum, np.nan),
        where=z_count > 0,
    ).astype(np.float32)
    z_range = z_max - z_min
    z_range[~np.isfinite(z_range)] = np.nan

    layers: dict[str, np.ndarray] = {
        "density": density,
        "z_mean": z_mean,
        "z_range": z_range,
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
    return layers, info, row, col, min_xy


def mask_cleanup(mask: np.ndarray, close_kernel_size: int, open_kernel_size: int) -> np.ndarray:
    """Làm sạch mask: đóng lỗ nhỏ, bỏ nhiễu nhỏ."""
    out = mask.astype(np.uint8)
    if close_kernel_size > 1:
        kernel = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    if open_kernel_size > 1:
        kernel = np.ones((open_kernel_size, open_kernel_size), dtype=np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    return out


def find_boundary_mask(road_mask: np.ndarray, z_range: np.ndarray, z_edge_percentile: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Tìm candidate mép đường.

    boundary_mask:
        rìa ngoài của vùng mặt đường.

    curb_score_mask:
        rìa ngoài cộng thêm vùng z_range cao, vì curb thường có thay đổi cao độ.
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(road_mask.astype(np.uint8), kernel, iterations=1)
    boundary = ((road_mask > 0) & (eroded == 0)).astype(np.uint8)

    z_valid = np.isfinite(z_range) & (road_mask > 0)
    z_edge = np.zeros_like(road_mask, dtype=np.uint8)
    if z_valid.any():
        thr = float(np.percentile(z_range[z_valid], z_edge_percentile))
        z_edge = ((z_range >= thr) & z_valid).astype(np.uint8)
        z_edge = cv2.dilate(z_edge, kernel, iterations=1)

    curb_score = ((boundary > 0) | ((boundary > 0) & (z_edge > 0))).astype(np.uint8)
    return boundary, curb_score


def contour_length_pixels(contour: np.ndarray) -> float:
    """Tính độ dài contour theo pixel."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def contours_to_polylines(
    boundary_mask: np.ndarray,
    min_xy: np.ndarray,
    resolution: float,
    min_points: int,
    min_length_m: float,
) -> list[dict]:
    """Đổi contour ảnh BEV thành polyline thô trong hệ map."""
    contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    height = boundary_mask.shape[0]
    polylines = []
    for contour in contours:
        if len(contour) < min_points:
            continue
        length_m = contour_length_pixels(contour) * resolution
        if length_m < min_length_m:
            continue

        pts = contour.reshape(-1, 2).astype(np.float64)
        col = pts[:, 0]
        row = pts[:, 1]
        x = min_xy[0] + (col + 0.5) * resolution
        y = min_xy[1] + ((height - 1 - row) + 0.5) * resolution
        polyline = np.column_stack((x, y)).astype(float)

        # Giảm bớt số điểm để JSON không quá nặng.
        if len(polyline) > 600:
            ids = np.linspace(0, len(polyline) - 1, 600, dtype=np.int32)
            polyline = polyline[ids]

        polylines.append(
            {
                "length_m": length_m,
                "num_points": int(len(polyline)),
                "xy_map": polyline.tolist(),
            }
        )
    polylines.sort(key=lambda item: item["length_m"], reverse=True)
    return polylines


def main() -> None:
    args = parse_args()
    if not args.input_npz.is_file():
        raise FileNotFoundError(args.input_npz)

    out_dir = ensure_output_dir(args.scenario, "road_boundary_curb")
    data = np.load(args.input_npz)
    points_map = data["points_map"].astype(np.float32)
    local_xyz = data["local_xyz_base"].astype(np.float32)
    intensity = data["intensity"].astype(np.float32) if "intensity" in data.files else None

    layers, bev_info, row, col, min_xy = build_bev_grid(
        points_map,
        local_xyz[:, 2],
        intensity,
        args.resolution,
        args.max_bev_pixels,
    )

    road_mask_raw = (layers["density"] >= args.min_points_per_cell).astype(np.uint8)
    road_mask = mask_cleanup(road_mask_raw, args.close_kernel_size, args.open_kernel_size)
    boundary_mask, curb_score_mask = find_boundary_mask(
        road_mask,
        layers["z_range"],
        args.z_edge_percentile,
    )

    point_boundary_flag = boundary_mask[row, col] > 0
    boundary_points = points_map[point_boundary_flag]
    boundary_local = local_xyz[point_boundary_flag]
    boundary_intensity = intensity[point_boundary_flag] if intensity is not None else None

    candidate_npz = out_dir / "boundary_curb_candidates_map.npz"
    save_data = {
        "points_map": boundary_points.astype(np.float32),
        "local_xyz_base": boundary_local.astype(np.float32),
    }
    if boundary_intensity is not None:
        save_data["intensity"] = boundary_intensity.astype(np.float32)
    np.savez_compressed(candidate_npz, **save_data)

    polylines = contours_to_polylines(
        boundary_mask,
        min_xy,
        bev_info["resolution_used"],
        args.min_contour_points,
        args.min_contour_length_m,
    )
    polylines_path = out_dir / "boundary_polylines_map.json"
    with polylines_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "coordinate_frame": "map",
                "polyline_type": "rough_boundary_candidate",
                "count": len(polylines),
                "polylines": polylines,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    road_u8 = (road_mask * 255).astype(np.uint8)
    boundary_u8 = (boundary_mask * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "bev_road_mask.png"), road_u8)
    cv2.imwrite(str(out_dir / "bev_boundary_candidate.png"), boundary_u8)
    save_colormap(out_dir / "bev_density.png", normalize_to_u8(np.log1p(layers["density"]), (1, 99.5)))
    save_colormap(out_dir / "bev_z_range.png", normalize_to_u8(layers["z_range"], (1, 99)))

    if "intensity_mean" in layers:
        intensity_u8 = normalize_to_u8(layers["intensity_mean"], (1, 99))
        overlay = cv2.cvtColor(intensity_u8, cv2.COLOR_GRAY2BGR)
        overlay[boundary_mask > 0] = (0, 0, 255)
        overlay[curb_score_mask > 0] = (0, 255, 255)
        cv2.imwrite(str(out_dir / "bev_boundary_on_intensity.png"), overlay)

    report = {
        "input_npz": str(args.input_npz),
        "point_count": int(len(points_map)),
        "boundary_candidate_point_count": int(len(boundary_points)),
        "polyline_count": int(len(polylines)),
        "bev": bev_info,
        "thresholds": {
            "min_points_per_cell": int(args.min_points_per_cell),
            "close_kernel_size": int(args.close_kernel_size),
            "open_kernel_size": int(args.open_kernel_size),
            "min_contour_points": int(args.min_contour_points),
            "min_contour_length_m": float(args.min_contour_length_m),
            "z_edge_percentile": float(args.z_edge_percentile),
        },
        "outputs": {
            "candidate_npz": str(candidate_npz),
            "polylines_json": str(polylines_path),
            "bev_road_mask": str(out_dir / "bev_road_mask.png"),
            "bev_boundary_candidate": str(out_dir / "bev_boundary_candidate.png"),
            "bev_boundary_on_intensity": str(out_dir / "bev_boundary_on_intensity.png"),
        },
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Ground/Road points : {len(points_map):,}")
    print(f"Boundary candidates: {len(boundary_points):,}")
    print(f"Rough polylines    : {len(polylines):,}")
    print(f"Output dir         : {out_dir}")


if __name__ == "__main__":
    main()
