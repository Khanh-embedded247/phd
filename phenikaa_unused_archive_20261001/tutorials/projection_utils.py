"""Shared LiDAR projection + overlay helpers for camera tutorials."""

from __future__ import annotations

import cv2
import numpy as np

from lidar_color import build_intensity_stats, compute_point_colors, depths_from_camera_points
from lidar_draw import draw_lidar_overlay
from lidar_viz_config import (
    DEPTH_BINS,
    DEPTH_COLORS,
    DEPTH_JET_COLORMAP,
    DEPTH_JET_FAR_M,
    DEPTH_JET_NEAR_M,
    DEPTH_JET_NORM,
    DEPTH_JET_PERCENTILE_HIGH,
    DEPTH_JET_PERCENTILE_LOW,
    DEPTH_JET_METRIC,
    LIDAR_COLOR_MODE,
    LIDAR_DEPTH_REF_M,
    LIDAR_DEPTH_SIZE_MAX,
    LIDAR_DEPTH_SIZE_MIN,
    LIDAR_DRAW_STYLE,
    LIDAR_INTENSITY_COLORMAP,
    LIDAR_INTENSITY_NORM,
    LIDAR_INTENSITY_PERCENTILE_HIGH,
    LIDAR_INTENSITY_PERCENTILE_LOW,
    LIDAR_MONO_BGR,
    LIDAR_OVERLAY_ALPHA,
    LIDAR_POINT_RADIUS,
    UV_MARGIN_PX,
)


def project_uv_to_pixels(uv, *extras, w=None, h=None, margin=UV_MARGIN_PX):
    """projectPoints uv → int32; lọc NaN/inf và tọa độ overflow int32."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    valid = np.isfinite(uv).all(axis=1)
    if w is not None and h is not None:
        valid &= (
            (uv[:, 0] >= -margin)
            & (uv[:, 0] < w + margin)
            & (uv[:, 1] >= -margin)
            & (uv[:, 1] < h + margin)
        )
    else:
        valid &= (np.abs(uv) < 1e6).all(axis=1)

    uv = np.round(uv[valid])
    uv = np.clip(uv, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)
    if not extras:
        return uv
    if len(extras) == 1:
        return uv, np.asarray(extras[0])[valid]
    return (uv,) + tuple(np.asarray(e)[valid] for e in extras)


def intensity_stats_from_las(las) -> tuple[np.ndarray | None, dict | None]:
    """Đọc intensity + stats toàn frame từ laspy object."""
    if "intensity" not in las.point_format.dimension_names:
        return None, None
    intensity = np.array(las.intensity, dtype=np.float32)
    stats = build_intensity_stats(
        intensity,
        percentile_low=LIDAR_INTENSITY_PERCENTILE_LOW,
        percentile_high=LIDAR_INTENSITY_PERCENTILE_HIGH,
    )
    return intensity, stats


def paint_lidar_on_image(
    img: np.ndarray,
    points_xyz: np.ndarray,
    Rt: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    *,
    is_fisheye: bool,
    intensities: np.ndarray | None = None,
    intensity_stats: dict | None = None,
    uv_margin: int = UV_MARGIN_PX,
) -> tuple[np.ndarray, int]:
    """
    Project LiDAR XYZ (lidar frame) lên ảnh và vẽ theo lidar_viz_config.
    """
    h, w = img.shape[:2]
    if len(points_xyz) == 0:
        return img.copy(), 0

    points_hom = np.hstack((points_xyz, np.ones((len(points_xyz), 1))))
    points_cam = (Rt @ points_hom.T).T
    mask = points_cam[:, 2] > 0.1
    points_cam = points_cam[mask]
    if len(points_cam) == 0:
        return img.copy(), 0

    depths = depths_from_camera_points(points_cam, DEPTH_JET_METRIC)
    intensities_m = intensities[mask] if intensities is not None else None

    obj_pts = points_cam[:, np.newaxis, :].astype(np.float32)
    rvec = tvec = np.zeros(3, dtype=np.float32)
    if is_fisheye:
        uv, _ = cv2.fisheye.projectPoints(
            obj_pts, rvec, tvec, K.astype(np.float32), D.astype(np.float32)
        )
    else:
        uv, _ = cv2.projectPoints(obj_pts, rvec, tvec, K.astype(np.float32), D)

    if intensities_m is not None:
        uv, depths, intensities_m = project_uv_to_pixels(
            uv, depths, intensities_m, w=w, h=h, margin=uv_margin
        )
    else:
        uv, depths = project_uv_to_pixels(uv, depths, w=w, h=h, margin=uv_margin)

    colors = compute_point_colors(
        len(uv),
        LIDAR_COLOR_MODE,
        depths=depths,
        intensities=intensities_m,
        intensity_stats=intensity_stats,
        intensity_norm=LIDAR_INTENSITY_NORM,
        depth_bins=DEPTH_BINS,
        depth_colors=DEPTH_COLORS,
        mono_color=LIDAR_MONO_BGR,
        intensity_colormap=LIDAR_INTENSITY_COLORMAP,
        intensity_percentile_low=LIDAR_INTENSITY_PERCENTILE_LOW,
        intensity_percentile_high=LIDAR_INTENSITY_PERCENTILE_HIGH,
        depth_jet_near_m=DEPTH_JET_NEAR_M,
        depth_jet_far_m=DEPTH_JET_FAR_M,
        depth_jet_colormap=DEPTH_JET_COLORMAP,
        depth_jet_norm=DEPTH_JET_NORM,
        depth_jet_percentile_low=DEPTH_JET_PERCENTILE_LOW,
        depth_jet_percentile_high=DEPTH_JET_PERCENTILE_HIGH,
    )

    return draw_lidar_overlay(
        img,
        uv,
        colors,
        depths,
        style=LIDAR_DRAW_STYLE,
        radius=LIDAR_POINT_RADIUS,
        overlay_alpha=LIDAR_OVERLAY_ALPHA,
        depth_ref_m=LIDAR_DEPTH_REF_M,
        min_radius=LIDAR_DEPTH_SIZE_MIN,
        max_radius=LIDAR_DEPTH_SIZE_MAX,
    )
