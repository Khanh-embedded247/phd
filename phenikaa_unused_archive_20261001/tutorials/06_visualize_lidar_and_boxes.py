"""
Tutorial 06: Visualize LiDAR point cloud with 3D bounding boxes using Open3D.
Saves a screenshot to results/ and optionally opens an interactive window.

"""

import os

import laspy
import numpy as np
import open3d as o3d

from lidar_color import (
    build_intensity_stats,
    depth_to_jet_bgr,
    intensity_to_bgr,
)
from lidar_viz_config import (
    DEPTH_JET_COLORMAP,
    DEPTH_JET_FAR_M,
    DEPTH_JET_NEAR_M,
    DEPTH_JET_NORM,
    DEPTH_JET_PERCENTILE_HIGH,
    DEPTH_JET_PERCENTILE_LOW,
    LIDAR_INTENSITY_COLORMAP,
    LIDAR_INTENSITY_NORM,
    LIDAR_INTENSITY_PERCENTILE_HIGH,
    LIDAR_INTENSITY_PERCENTILE_LOW,
)
from phenikaa_paths import RESULT_ROOT, SEQUENCE_FOLDER, TIMESTAMP

# --- Open3D 3D view (tutorial 06) ---
# depth_jet — đỏ gần / xanh xa (dễ nhìn cấu trúc hơn intensity trên .laz)
# intensity — màu theo intensity gốc
O3D_LIDAR_COLOR_MODE = "depth_jet"
O3D_POINT_SIZE = 2.0
O3D_LINE_WIDTH = 4.0
O3D_BACKGROUND = np.array([0.12, 0.12, 0.14])  # nền tối → jet/turbo nổi hơn nền xám

CLASS_COLORS_RGB = {
    "Car": [1.0, 1.0, 0.0],
    "Rider": [0.0, 1.0, 0.0],
    "Pedestrian": [1.0, 0.0, 0.0],
}

OUT_DIR = RESULT_ROOT / SEQUENCE_FOLDER.name / TIMESTAMP / "open3d"
OUT_IMAGE = OUT_DIR / f"lidar_boxes_{TIMESTAMP}.png"

laz_path = SEQUENCE_FOLDER / "Lidar" / f"{TIMESTAMP}.laz"
ann_path = SEQUENCE_FOLDER / "Label" / f"{TIMESTAMP}.txt"

if not laz_path.exists():
    raise FileNotFoundError(f"LiDAR file not found: {laz_path}")

las = laspy.read(str(laz_path))
points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

if "intensity" in las.point_format.dimension_names:
    intensity = np.array(las.intensity, dtype=np.float32)
    intensity_stats = build_intensity_stats(
        intensity,
        percentile_low=LIDAR_INTENSITY_PERCENTILE_LOW,
        percentile_high=LIDAR_INTENSITY_PERCENTILE_HIGH,
    )
else:
    intensity = None
    intensity_stats = None

ranges = np.linalg.norm(points, axis=1)

if O3D_LIDAR_COLOR_MODE == "depth_jet":
    colors_bgr = depth_to_jet_bgr(
        ranges,
        near_m=DEPTH_JET_NEAR_M,
        far_m=DEPTH_JET_FAR_M,
        colormap=DEPTH_JET_COLORMAP,
        norm=DEPTH_JET_NORM,
        percentile_low=DEPTH_JET_PERCENTILE_LOW,
        percentile_high=DEPTH_JET_PERCENTILE_HIGH,
    )
elif O3D_LIDAR_COLOR_MODE == "intensity":
    if intensity is None:
        raise ValueError("No intensity in .laz — set O3D_LIDAR_COLOR_MODE='depth_jet'")
    colors_bgr = intensity_to_bgr(
        intensity,
        norm_mode=LIDAR_INTENSITY_NORM,
        stats=intensity_stats,
        colormap=LIDAR_INTENSITY_COLORMAP,
        percentile_low=LIDAR_INTENSITY_PERCENTILE_LOW,
        percentile_high=LIDAR_INTENSITY_PERCENTILE_HIGH,
    )
else:
    raise ValueError(f"Unknown O3D_LIDAR_COLOR_MODE: {O3D_LIDAR_COLOR_MODE}")

# Open3D dùng RGB [0, 1]
colors_rgb = colors_bgr[:, ::-1].astype(np.float64) / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

boxes = []
if ann_path.exists():
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                x, y, z, l, w, h, yaw = map(float, parts[:7])
                class_name = " ".join(parts[7:]).title()
                if class_name not in CLASS_COLORS_RGB:
                    continue

                corners = np.array([
                    [-l / 2, -w / 2, -h / 2], [l / 2, -w / 2, -h / 2],
                    [l / 2, w / 2, -h / 2], [-l / 2, w / 2, -h / 2],
                    [-l / 2, -w / 2, h / 2], [l / 2, -w / 2, h / 2],
                    [l / 2, w / 2, h / 2], [-l / 2, w / 2, h / 2],
                ])
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                rot = np.array([
                    [cos_y, -sin_y, 0.0],
                    [sin_y, cos_y, 0.0],
                    [0.0, 0.0, 1.0],
                ])
                corners = corners @ rot.T + np.array([x, y, z])
                lines = [
                    [0, 1], [1, 2], [2, 3], [3, 0],
                    [4, 5], [5, 6], [6, 7], [7, 4],
                    [0, 4], [1, 5], [2, 6], [3, 7],
                ]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                color = CLASS_COLORS_RGB[class_name]
                line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
                boxes.append(line_set)
            except ValueError:
                continue

geometries = [pcd, *boxes, o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)]

print(f"Sequence: {SEQUENCE_FOLDER.name} @ {TIMESTAMP}")
print(f"Points: {len(points):,} | Boxes: {len(boxes)} | Color: {O3D_LIDAR_COLOR_MODE}")

vis = o3d.visualization.Visualizer()
vis.create_window(
    window_name=f"PHENIKAA 3D — {SEQUENCE_FOLDER.name} — {TIMESTAMP}",
    width=1600,
    height=900,
    visible=bool(os.environ.get("DISPLAY")),
)

for geom in geometries:
    vis.add_geometry(geom)

opt = vis.get_render_option()
opt.point_size = O3D_POINT_SIZE
opt.background_color = O3D_BACKGROUND
opt.line_width = O3D_LINE_WIDTH

# Căn camera nhìn toàn scene
vis.reset_view_point(True)
for _ in range(5):
    vis.poll_events()
    vis.update_renderer()

OUT_DIR.mkdir(parents=True, exist_ok=True)
vis.capture_screen_image(str(OUT_IMAGE))
print(f"Saved: {OUT_IMAGE}")

if os.environ.get("DISPLAY"):
    print("Interactive view — close window to exit.")
    vis.run()

vis.destroy_window()
print("Done.")
