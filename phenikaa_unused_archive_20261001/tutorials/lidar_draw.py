"""LiDAR point overlay styles for camera projection tutorials."""

from __future__ import annotations

import cv2
import numpy as np

# circle  — chấm tròn đồng kích thước
# square  — ô vuông sắc (LINE_8), dễ thấy hơn AA mờ
# blend   — vẽ lên layer riêng rồi pha sáng với ảnh gốc
# depth_blend — gần = to hơn + pha sáng (khuyên dùng)
DRAW_STYLES = ("circle", "square", "blend", "depth_blend")


def depth_to_radius(
    depths: np.ndarray,
    *,
    base_radius: int,
    ref_distance_m: float = 25.0,
    min_radius: int = 1,
    max_radius: int = 6,
) -> np.ndarray:
    """
    Đổi khoảng cách (m) → bán kính chấm (px) cho depth_blend.

    Công thức: radius = base_radius * ref_distance_m / depth_m
    Tại depth == ref_distance_m → radius == base_radius.
    """
    depths = np.maximum(depths.astype(np.float64), 0.5)
    radius = np.round(base_radius * ref_distance_m / depths).astype(np.int32)
    return np.clip(radius, min_radius, max_radius)


def _draw_markers(
    layer: np.ndarray,
    uv: np.ndarray,
    colors: np.ndarray | list[tuple[int, int, int]],
    radii: np.ndarray,
    *,
    style: str,
) -> int:
    h, w = layer.shape[:2]
    count = 0
    use_square = style == "square"
    # LINE_8: màu đặc, không bị anti-alias làm nhạt như LINE_AA
    line_type = cv2.LINE_8

    for i, ((u, v), r) in enumerate(zip(uv, radii)):
        u, v, r = int(u), int(v), int(r)
        if not (0 <= u < w and 0 <= v < h):
            continue
        if isinstance(colors, np.ndarray):
            color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
        else:
            color = colors[i]
        if use_square:
            cv2.rectangle(layer, (u - r, v - r), (u + r, v + r), color, -1, lineType=line_type)
        else:
            cv2.circle(layer, (u, v), r, color, -1, lineType=line_type)
        count += 1
    return count


def _blend_overlay(base: np.ndarray, layer: np.ndarray, alpha: float) -> np.ndarray:
    """Pha layer điểm lên ảnh — chỉ vùng có điểm."""
    mask = layer.max(axis=2) > 0
    if not np.any(mask):
        return base.copy()
    out = base.copy()
    blended = cv2.addWeighted(base, 1.0 - alpha, layer, alpha, 0)
    out[mask] = blended[mask]
    return out


def draw_lidar_overlay(
    base_img: np.ndarray,
    uv: np.ndarray,
    colors: np.ndarray | list[tuple[int, int, int]],
    depths: np.ndarray | None = None,
    *,
    style: str = "depth_blend",
    radius: int = 2,
    overlay_alpha: float = 0.7,
    depth_ref_m: float = 25.0,
    min_radius: int = 1,
    max_radius: int = 6,
) -> tuple[np.ndarray, int]:
    """
    Vẽ điểm LiDAR lên ảnh.

    Returns (image, num_points_drawn).
    """
    if style not in DRAW_STYLES:
        raise ValueError(f"style must be one of {DRAW_STYLES}, got {style!r}")

    if len(uv) == 0:
        return base_img.copy(), 0

    if style == "depth_blend":
        if depths is None:
            raise ValueError("depth_blend requires depths")
        radii = depth_to_radius(
            depths,
            base_radius=radius,
            ref_distance_m=depth_ref_m,
            min_radius=min_radius,
            max_radius=max_radius,
        )
        layer = np.zeros_like(base_img)
        count = _draw_markers(layer, uv, colors, radii, style="circle")
        return _blend_overlay(base_img, layer, overlay_alpha), count

    if style == "blend":
        radii = np.full(len(uv), radius, dtype=np.int32)
        layer = np.zeros_like(base_img)
        count = _draw_markers(layer, uv, colors, radii, style="circle")
        return _blend_overlay(base_img, layer, overlay_alpha), count

    radii = np.full(len(uv), radius, dtype=np.int32)
    out = base_img.copy()
    marker_style = "square" if style == "square" else "circle"
    count = _draw_markers(out, uv, colors, radii, style=marker_style)
    return out, count
