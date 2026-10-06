"""
Cấu hình chung cho projection LiDAR + box lên ảnh.
Chỉ sửa file này — tutorial 07, 08, 09, 10, 11 đều import từ đây.
"""

from __future__ import annotations

import numpy as np

# --- 3D box class colors (BGR) ---
CLASS_COLORS = {
    "Car": (0, 255, 255),
    "Rider": (0, 255, 0),
    "Pedestrian": (0, 0, 255),
}

# --- Depth colormap rời (LIDAR_COLOR_MODE = "depth") ---
DEPTH_BINS = np.arange(0, 300, 10)
DEPTH_COLORS = [(255, 255, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)] * 6
DEPTH_COLORS = DEPTH_COLORS[: len(DEPTH_BINS) - 1]

# --- Depth jet liên tục (LIDAR_COLOR_MODE = "depth_jet") ---
# Đỏ/vàng = gần, xanh lam = xa — giống ảnh overlay mẫu dataset
#
# DEPTH_JET_METRIC:
#   z     — độ sâu theo trục camera (khớp viewer mẫu hơn range)
#   range — khoảng cách euclidean √((X²+Y²+Z²))
#
# DEPTH_JET_NORM:
#   fixed   — dùng NEAR_M / FAR_M cố định (ảnh mẫu thường ~3–60 m)
#   visible — kéo giãn p2–p98 trên điểm trong khung hình (khuyên dùng urban)
#
DEPTH_JET_METRIC = "z"
DEPTH_JET_NORM = "fixed"
DEPTH_JET_NEAR_M = 2.0
DEPTH_JET_FAR_M = 50.0
DEPTH_JET_PERCENTILE_LOW = 2.0
DEPTH_JET_PERCENTILE_HIGH = 98.0
DEPTH_JET_COLORMAP = "turbo"

# --- Màu điểm LiDAR ---
# depth | depth_jet | intensity | intensity_gray | mono
LIDAR_COLOR_MODE = "depth_jet"
LIDAR_INTENSITY_NORM = "raw"  # raw | global_max | global_percentile | local_percentile
LIDAR_MONO_BGR = (0, 255, 0)
LIDAR_INTENSITY_COLORMAP = "jet_t06"  # jet_t06 (= tutorial 06) | jet | turbo | viridis
LIDAR_INTENSITY_PERCENTILE_LOW = 1.0  # percentile 0–100 (KHÔNG đặt 255)
LIDAR_INTENSITY_PERCENTILE_HIGH = 99.0

# --- LiDAR point overlay (kích thước chấm trên ảnh) ---
# LIDAR_POINT_RADIUS: bán kính px trên ảnh gốc (trước SAVE_SIZE).
#   circle | square | blend  → mọi điểm cùng radius = LIDAR_POINT_RADIUS
#   depth_blend → radius = round(RADIUS * DEPTH_REF_M / depth_m), clip [MIN, MAX]
#   VD RADIUS=2, REF_M=25: 25m→2px, 50m→1px, 5m→6px (MAX)
LIDAR_DRAW_STYLE = "circle"  # circle | square | blend | depth_blend
LIDAR_POINT_RADIUS = 2
LIDAR_OVERLAY_ALPHA = 0.45  # 0–1, chỉ blend / depth_blend
LIDAR_DEPTH_REF_M = 25.0
LIDAR_DEPTH_SIZE_MIN = 1
LIDAR_DEPTH_SIZE_MAX = 6

# Ảnh lưu từ tutorial 11
SAVE_SIZE = (1280, 1280)

# Lọc uv trước cast int32 (projection_utils)
UV_MARGIN_PX = 100
