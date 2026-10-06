"""LiDAR point color modes for camera projection."""

from __future__ import annotations

import cv2
import numpy as np

# depth          — màu rời theo bin 10 m (DEPTH_BINS / DEPTH_COLORS)
# depth_jet      — jet liên tục theo khoảng cách: đỏ/vàng gần → xanh xa (ảnh mẫu dataset)
# intensity      — colormap từ intensity (chuẩn hóa theo LIDAR_INTENSITY_NORM)
# intensity_gray — intensity → xám
# mono           — một màu BGR
COLOR_MODES = ("depth", "depth_jet", "intensity", "intensity_gray", "mono")

# raw            — I / 255 (giá trị gốc 8-bit trong .laz, khớp viewer chuẩn)
# global_max     — I / max(frame), giống tutorial 06 Open3D
# global_percentile — scale theo p1–p99 của **cả frame** (ổn định giữa các camera)
# local_percentile  — scale theo điểm nhìn thấy từng camera (dễ lệch màu, không khuyên dùng)
INTENSITY_NORM_MODES = ("raw", "global_max", "global_percentile", "local_percentile")

_OPENCV_COLORMAPS = {
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
}


def jet_colormap_bgr_t06(t: np.ndarray) -> np.ndarray:
    """Jet giống tutorial 06 (Open3D), trả về (N, 3) uint8 BGR."""
    t = np.clip(t.astype(np.float64), 0.0, 1.0)
    r = np.clip(np.minimum(4 * t - 1.5, -4 * t + 4.5), 0.0, 1.0)
    g = np.clip(np.minimum(4 * t - 0.5, -4 * t + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(4 * t + 0.5, -4 * t + 2.5), 0.0, 1.0)
    return (np.stack([b, g, r], axis=1) * 255.0).astype(np.uint8)


def build_intensity_stats(
    intensity: np.ndarray,
    *,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    raw_scale: float = 255.0,
) -> dict[str, float]:
    """Thống kê intensity trên **toàn bộ** point cloud của frame."""
    i = intensity.astype(np.float64)
    return {
        "min": float(i.min()),
        "max": float(i.max()),
        "mean": float(i.mean()),
        "p_lo": float(np.percentile(i, percentile_low)),
        "p_hi": float(np.percentile(i, percentile_high)),
        "raw_scale": float(raw_scale),
    }


def normalize_intensity_values(
    intensity: np.ndarray,
    mode: str,
    stats: dict[str, float] | None = None,
    *,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """Map intensity → [0, 1]."""
    if mode not in INTENSITY_NORM_MODES:
        raise ValueError(f"mode must be one of {INTENSITY_NORM_MODES}, got {mode!r}")

    i = intensity.astype(np.float64)

    if mode == "raw":
        scale = (stats or {}).get("raw_scale", 255.0)
        return np.clip(i / scale, 0.0, 1.0)

    if mode == "global_max":
        if stats is None:
            raise ValueError("global_max requires frame intensity stats")
        return np.clip(i / (stats["max"] + 1e-8), 0.0, 1.0)

    if mode == "global_percentile":
        if stats is None:
            raise ValueError("global_percentile requires frame intensity stats")
        lo, hi = stats["p_lo"], stats["p_hi"]
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((i - lo) / (hi - lo), 0.0, 1.0)

    # local_percentile — chỉ trên subset điểm đang vẽ
    lo = float(np.percentile(i, percentile_low))
    hi = float(np.percentile(i, percentile_high))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((i - lo) / (hi - lo), 0.0, 1.0)


def intensity_to_bgr(
    intensity: np.ndarray,
    *,
    norm_mode: str = "raw",
    stats: dict[str, float] | None = None,
    colormap: str = "jet_t06",
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    grayscale: bool = False,
) -> np.ndarray:
    """(N,) intensity → (N, 3) uint8 BGR."""
    norm = normalize_intensity_values(
        intensity,
        norm_mode,
        stats,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )

    if grayscale:
        gray = (norm * 255.0).astype(np.uint8)
        return np.repeat(gray.reshape(-1, 1), 3, axis=1)

    if colormap == "jet_t06":
        return jet_colormap_bgr_t06(norm)

    gray = (norm * 255.0).astype(np.uint8).reshape(-1, 1)
    cmap = _OPENCV_COLORMAPS.get(colormap, cv2.COLORMAP_JET)
    return cv2.applyColorMap(gray, cmap).reshape(-1, 3)


def colormap_bgr_from_t(t: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """t ∈ [0, 1] → (N, 3) BGR."""
    t = np.clip(t.astype(np.float64), 0.0, 1.0)
    if colormap == "jet_t06":
        return jet_colormap_bgr_t06(t)
    gray = (t * 255.0).astype(np.uint8).reshape(-1, 1)
    cmap = _OPENCV_COLORMAPS.get(colormap, cv2.COLORMAP_JET)
    return cv2.applyColorMap(gray, cmap).reshape(-1, 3)


def depths_from_camera_points(points_cam: np.ndarray, metric: str = "z") -> np.ndarray:
    """Độ sâu từ điểm trong hệ camera: z (trục nhìn) hoặc range (euclidean)."""
    if metric == "range":
        return np.linalg.norm(points_cam[:, :3], axis=1)
    return points_cam[:, 2].astype(np.float64)


def resolve_depth_jet_range(
    depths: np.ndarray,
    norm: str,
    near_m: float,
    far_m: float,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> tuple[float, float]:
    """
    fixed   — dùng near_m / far_m cố định
    visible — kéo giãn theo percentile điểm đang vẽ (mỗi camera / mỗi frame)
    """
    if norm == "visible" and len(depths) > 0:
        lo = float(np.percentile(depths, percentile_low))
        hi = float(np.percentile(depths, percentile_high))
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi
    return near_m, far_m


def depth_to_jet_bgr(
    depths: np.ndarray,
    *,
    near_m: float,
    far_m: float,
    colormap: str = "jet",
    norm: str = "fixed",
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> np.ndarray:
    """
    Màu theo khoảng cách tới camera (m).

    Gần → đỏ/vàng; xa → xanh lam — giống overlay LiDAR trên ảnh mẫu.
    Dùng norm='visible' để dải màu phủ đủ đỏ→xanh trên điểm trong khung hình.
    """
    d = depths.astype(np.float64)
    near_m, far_m = resolve_depth_jet_range(
        d, norm, near_m, far_m, percentile_low, percentile_high
    )
    if far_m <= near_m:
        far_m = near_m + 1.0
    t = np.clip((d - near_m) / (far_m - near_m), 0.0, 1.0)
    t = 1.0 - t  # gần = warm/red
    return colormap_bgr_from_t(t, colormap)


def depth_to_bgr(
    depths: np.ndarray,
    depth_bins: np.ndarray,
    depth_colors: list[tuple[int, int, int]],
) -> np.ndarray:
    bin_idx = np.clip(np.digitize(depths, depth_bins) - 1, 0, len(depth_colors) - 1)
    return np.array([depth_colors[i] for i in bin_idx], dtype=np.uint8)


def mono_bgr(count: int, color: tuple[int, int, int]) -> np.ndarray:
    return np.full((count, 3), color, dtype=np.uint8)


def compute_point_colors(
    count: int,
    mode: str,
    *,
    depths: np.ndarray | None = None,
    intensities: np.ndarray | None = None,
    intensity_stats: dict[str, float] | None = None,
    intensity_norm: str = "raw",
    depth_bins: np.ndarray | None = None,
    depth_colors: list[tuple[int, int, int]] | None = None,
    mono_color: tuple[int, int, int] = (0, 255, 255),
    intensity_colormap: str = "jet_t06",
    intensity_percentile_low: float = 1.0,
    intensity_percentile_high: float = 99.0,
    depth_jet_near_m: float = 5.0,
    depth_jet_far_m: float = 100.0,
    depth_jet_colormap: str = "jet",
    depth_jet_norm: str = "fixed",
    depth_jet_percentile_low: float = 2.0,
    depth_jet_percentile_high: float = 98.0,
) -> np.ndarray:
    if mode not in COLOR_MODES:
        raise ValueError(f"mode must be one of {COLOR_MODES}, got {mode!r}")

    if mode == "depth":
        if depths is None or depth_bins is None or depth_colors is None:
            raise ValueError("depth mode requires depths, depth_bins, depth_colors")
        return depth_to_bgr(depths, depth_bins, depth_colors)

    if mode == "depth_jet":
        if depths is None:
            raise ValueError("depth_jet requires depths")
        return depth_to_jet_bgr(
            depths,
            near_m=depth_jet_near_m,
            far_m=depth_jet_far_m,
            colormap=depth_jet_colormap,
            norm=depth_jet_norm,
            percentile_low=depth_jet_percentile_low,
            percentile_high=depth_jet_percentile_high,
        )

    if mode in ("intensity", "intensity_gray"):
        if intensities is None:
            raise ValueError(f"{mode} requires intensities from .laz")
        return intensity_to_bgr(
            intensities,
            norm_mode=intensity_norm,
            stats=intensity_stats,
            colormap=intensity_colormap,
            percentile_low=intensity_percentile_low,
            percentile_high=intensity_percentile_high,
            grayscale=(mode == "intensity_gray"),
        )

    return mono_bgr(count, mono_color)
