#!/usr/bin/env python3
"""
Bước 06: kiểm tra map GLIM đã tạo sẵn và tạo ảnh BEV để nhìn nhanh.

Input:
    data/Normal/map - Cloud.pcd

Output:
    outputs/benchmark/Normal/glim_map/
        stats.json
        bev_density.png
        bev_height_max.png
        bev_height_range.png
        bev_intensity_mean.png  nếu PCD có intensity

Ý nghĩa:
- File PCD này đã nằm trong hệ map, nên chưa cần traj_lidar để gom map nữa.
- Bước này giúp nhìn xem map có rõ mặt đường, lề đường, vỉa hè, vạch phản xạ không.
- Sau khi inspect xong mới quyết định cách lọc ground / curb / lane marking.
bev_density.png
-> nơi nào nhiều point

bev_height_max.png
-> độ cao lớn nhất mỗi ô

bev_height_range.png
-> chênh cao trong mỗi ô, hữu ích để nhìn curb/vỉa hè/mép đường

bev_intensity_mean.png
-> phản xạ trung bình, có thể giúp nhìn vạch đường nếu vạch phản xạ rõ
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import DEFAULT_SCENARIO, PHENIKAA_ROOT, ensure_output_dir


# ============================================================
# USER CONFIG
# ============================================================

CONFIG_SCENARIO = DEFAULT_SCENARIO
CONFIG_MAP_PCD = PHENIKAA_ROOT / "data" / CONFIG_SCENARIO / "map - Cloud.pcd"

# Độ phân giải BEV tính bằng mét/pixel. Nếu map quá lớn, script tự tăng
# resolution để tránh tạo ảnh quá khổng lồ.
CONFIG_BEV_RESOLUTION = 0.10
CONFIG_MAX_BEV_PIXELS = 25_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=CONFIG_SCENARIO)
    parser.add_argument("--map-pcd", type=Path, default=CONFIG_MAP_PCD)
    parser.add_argument("--resolution", type=float, default=CONFIG_BEV_RESOLUTION)
    parser.add_argument("--max-bev-pixels", type=int, default=CONFIG_MAX_BEV_PIXELS)
    return parser.parse_args()


def parse_pcd_header(path: Path) -> tuple[dict, int]:
    """Đọc header PCD và trả về metadata + offset bắt đầu binary data."""
    meta: dict[str, object] = {}
    offset = 0
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Missing DATA line in {path}")
            offset += len(line)
            text = line.decode("utf-8", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            key = key.upper()
            if key in {"FIELDS", "TYPE"}:
                meta[key] = values
            elif key in {"SIZE", "COUNT"}:
                meta[key] = [int(v) for v in values]
            elif key in {"WIDTH", "HEIGHT", "POINTS"}:
                meta[key] = int(values[0])
            elif key == "DATA":
                meta[key] = values[0].lower()
                break
            else:
                meta[key] = values
    return meta, offset


def pcd_numpy_type(type_code: str, size: int) -> np.dtype:
    if type_code == "F" and size == 4:
        return np.dtype("<f4")
    if type_code == "F" and size == 8:
        return np.dtype("<f8")
    if type_code == "U" and size == 1:
        return np.dtype("u1")
    if type_code == "U" and size == 2:
        return np.dtype("<u2")
    if type_code == "U" and size == 4:
        return np.dtype("<u4")
    if type_code == "I" and size == 1:
        return np.dtype("i1")
    if type_code == "I" and size == 2:
        return np.dtype("<i2")
    if type_code == "I" and size == 4:
        return np.dtype("<i4")
    raise ValueError(f"Unsupported PCD field type: TYPE={type_code}, SIZE={size}")


def pcd_field_offsets(meta: dict) -> tuple[dict[str, int], int]:
    """Tính byte offset của từng field trong một point record."""
    fields = meta["FIELDS"]
    sizes = meta["SIZE"]
    counts = meta.get("COUNT", [1] * len(fields))

    offsets = {}
    offset = 0
    for field, size, count in zip(fields, sizes, counts):
        if field not in offsets:
            offsets[field] = offset
        offset += size * count
    return offsets, offset


def read_binary_pcd_xyz_intensity(path: Path) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Đọc PCD binary thành points Nx3 và intensity nếu có."""
    meta, data_offset = parse_pcd_header(path)
    if meta.get("DATA") != "binary":
        raise ValueError(f"Only binary PCD is supported for now, got DATA={meta.get('DATA')}")

    fields = meta["FIELDS"]
    sizes = meta["SIZE"]
    types = meta["TYPE"]
    points_count = int(meta["POINTS"])
    offsets, point_step = pcd_field_offsets(meta)

    required = {"x", "y", "z"}
    missing = required.difference(offsets)
    if missing:
        raise ValueError(f"PCD missing required fields: {sorted(missing)}")

    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=data_offset)
    expected_bytes = points_count * point_step
    if raw.size < expected_bytes:
        raise ValueError(
            f"PCD binary block is smaller than expected: {raw.size} < {expected_bytes}"
        )

    field_type = {field: pcd_numpy_type(t, s) for field, t, s in zip(fields, types, sizes)}

    def read_field(name: str) -> np.ndarray:
        return np.ndarray(
            shape=(points_count,),
            dtype=field_type[name],
            buffer=raw,
            offset=offsets[name],
            strides=(point_step,),
        ).astype(np.float32, copy=True)

    x = read_field("x")
    y = read_field("y")
    z = read_field("z")
    points = np.column_stack((x, y, z))

    intensity = read_field("intensity") if "intensity" in offsets else None
    return points, intensity, meta


def normalize_to_u8(values: np.ndarray, *, clip_percentile: tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
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


def save_colormap(path: Path, image_u8: np.ndarray, colormap: int = cv2.COLORMAP_TURBO) -> None:
    colored = cv2.applyColorMap(image_u8, colormap)
    cv2.imwrite(str(path), colored)


def build_bev_layers(
    points: np.ndarray,
    intensity: np.ndarray | None,
    resolution: float,
    max_pixels: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Tạo các lớp BEV cơ bản từ point cloud map."""
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if intensity is not None:
        intensity = intensity[valid]

    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    span_xy = max_xyz[:2] - min_xyz[:2]

    width = int(np.ceil(span_xy[0] / resolution)) + 1
    height = int(np.ceil(span_xy[1] / resolution)) + 1
    pixels = width * height
    used_resolution = resolution
    if pixels > max_pixels:
        scale = np.sqrt(pixels / max_pixels)
        used_resolution = resolution * scale
        width = int(np.ceil(span_xy[0] / used_resolution)) + 1
        height = int(np.ceil(span_xy[1] / used_resolution)) + 1

    ix = np.floor((points[:, 0] - min_xyz[0]) / used_resolution).astype(np.int32)
    iy = np.floor((points[:, 1] - min_xyz[1]) / used_resolution).astype(np.int32)
    ix = np.clip(ix, 0, width - 1)
    iy = np.clip(iy, 0, height - 1)
    row = height - 1 - iy
    col = ix

    density = np.zeros((height, width), dtype=np.float32)
    z_max = np.full((height, width), -np.inf, dtype=np.float32)
    z_min = np.full((height, width), np.inf, dtype=np.float32)
    np.add.at(density, (row, col), 1.0)
    np.maximum.at(z_max, (row, col), points[:, 2].astype(np.float32))
    np.minimum.at(z_min, (row, col), points[:, 2].astype(np.float32))

    z_range = z_max - z_min
    z_max[~np.isfinite(z_max)] = np.nan
    z_min[~np.isfinite(z_min)] = np.nan
    z_range[~np.isfinite(z_range)] = np.nan

    layers = {
        "density": density,
        "z_max": z_max,
        "z_range": z_range,
    }

    if intensity is not None:
        intensity_sum = np.zeros((height, width), dtype=np.float64)
        intensity_count = np.zeros((height, width), dtype=np.float64)
        np.add.at(intensity_sum, (row, col), intensity.astype(np.float64))
        np.add.at(intensity_count, (row, col), 1.0)
        intensity_mean = np.divide(
            intensity_sum,
            intensity_count,
            out=np.full_like(intensity_sum, np.nan),
            where=intensity_count > 0,
        )
        layers["intensity_mean"] = intensity_mean.astype(np.float32)

    info = {
        "point_count": int(len(points)),
        "min_xyz": min_xyz.tolist(),
        "max_xyz": max_xyz.tolist(),
        "resolution_requested": float(resolution),
        "resolution_used": float(used_resolution),
        "bev_width": int(width),
        "bev_height": int(height),
        "bev_pixels": int(width * height),
    }
    return layers, info


def main() -> None:
    args = parse_args()
    if not args.map_pcd.is_file():
        raise FileNotFoundError(args.map_pcd)

    out_dir = ensure_output_dir(args.scenario, "glim_map")
    points, intensity, meta = read_binary_pcd_xyz_intensity(args.map_pcd)
    layers, info = build_bev_layers(
        points,
        intensity,
        args.resolution,
        args.max_bev_pixels,
    )

    stats = {
        "map_pcd": str(args.map_pcd),
        "pcd_fields": meta.get("FIELDS"),
        "pcd_points_header": meta.get("POINTS"),
        "has_intensity": intensity is not None,
        **info,
    }

    stats_path = out_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    density_u8 = normalize_to_u8(np.log1p(layers["density"]), clip_percentile=(1, 99.5))
    cv2.imwrite(str(out_dir / "bev_density.png"), density_u8)
    save_colormap(out_dir / "bev_density_color.png", density_u8)

    z_max_u8 = normalize_to_u8(layers["z_max"], clip_percentile=(1, 99))
    save_colormap(out_dir / "bev_height_max.png", z_max_u8)

    z_range_u8 = normalize_to_u8(layers["z_range"], clip_percentile=(1, 99))
    save_colormap(out_dir / "bev_height_range.png", z_range_u8)

    if "intensity_mean" in layers:
        intensity_u8 = normalize_to_u8(layers["intensity_mean"], clip_percentile=(1, 99))
        cv2.imwrite(str(out_dir / "bev_intensity_mean.png"), intensity_u8)
        save_colormap(out_dir / "bev_intensity_mean_color.png", intensity_u8)

    print(f"PCD             : {args.map_pcd}")
    print(f"Points          : {stats['point_count']:,}")
    print(f"XYZ min         : {np.array(stats['min_xyz'])}")
    print(f"XYZ max         : {np.array(stats['max_xyz'])}")
    print(f"BEV resolution  : {stats['resolution_used']:.3f} m/pixel")
    print(f"BEV size        : {stats['bev_width']} x {stats['bev_height']}")
    print(f"Output dir      : {out_dir}")


if __name__ == "__main__":
    main()

