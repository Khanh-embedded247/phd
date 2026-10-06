"""
Tutorial 08: Project LiDAR points and 3D boxes onto all camera images in a 5x2 grid.
Displays a multi-view projection for a single timestamp.

"""

import cv2
import numpy as np
import json
import laspy
from scipy.spatial.transform import Rotation as R

from phenikaa_paths import (
    CAMERAS,
    EXTRINSIC_JSON,
    INTRINSIC_JSON,
    SEQUENCE_FOLDER,
    TIMESTAMP,
)
from lidar_viz_config import CLASS_COLORS
from projection_utils import intensity_stats_from_las, paint_lidar_on_image

with open(INTRINSIC_JSON) as f:
    INTRINSICS = json.load(f)
with open(EXTRINSIC_JSON) as f:
    EXTRINSICS = json.load(f)

# Grid
GRID_COLS = 5
GRID_ROWS = 2
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 720
CELL_WIDTH = OUTPUT_WIDTH // GRID_COLS
CELL_HEIGHT = OUTPUT_HEIGHT // GRID_ROWS

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_thickness = 1
text_color = (255, 255, 255)
padding_left = 12
padding_top = 40

ann_path = SEQUENCE_FOLDER / "Label" / f"{TIMESTAMP}.txt"
annotations = []
if ann_path.exists():
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            x, y, z, l, w, h, yaw = map(float, parts[:7])
            class_name = " ".join(parts[7:]).title()
            if class_name in CLASS_COLORS:
                annotations.append(
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "l": l,
                        "w": w,
                        "h": h,
                        "yaw": yaw,
                        "class_name": class_name,
                    }
                )

laz_path = SEQUENCE_FOLDER / "Lidar" / f"{TIMESTAMP}.laz"
points = np.empty((0, 3))
intensities = None
intensity_stats = None
if laz_path.exists():
    las = laspy.read(str(laz_path))
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    intensities, intensity_stats = intensity_stats_from_las(las)

canvas = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
canvas[:] = (30, 30, 30)

for idx, cam_name in enumerate(CAMERAS):
    jpg_path = SEQUENCE_FOLDER / "Image" / cam_name / f"{TIMESTAMP}.jpg"
    row = idx // GRID_COLS
    col = idx % GRID_COLS
    x_start = col * CELL_WIDTH
    y_start = row * CELL_HEIGHT

    if not jpg_path.exists():
        cv2.putText(
            canvas,
            cam_name,
            (x_start + padding_left, y_start + padding_top),
            font,
            font_scale,
            (0, 0, 255),
            font_thickness,
        )
        cv2.putText(
            canvas,
            "MISSING",
            (x_start + padding_left, y_start + padding_top + 40),
            font,
            font_scale,
            (0, 0, 255),
            font_thickness,
        )
        continue

    img = cv2.imread(str(jpg_path))
    h, w = img.shape[:2]
    is_fisheye = cam_name.startswith("CAM_F")

    K = np.array(INTRINSICS[cam_name]["camera_matrix"], dtype=np.float64).reshape(3, 3)
    D = np.array(INTRINSICS[cam_name]["distortion_coefficients"], dtype=np.float64)
    if is_fisheye and len(D) > 4:
        D = D[:4]

    Rt = np.array(EXTRINSICS[cam_name], dtype=np.float64)[:3, :]
    out_img = img.copy()

    if len(points) > 0:
        out_img, _ = paint_lidar_on_image(
            out_img,
            points,
            Rt,
            K,
            D,
            is_fisheye=is_fisheye,
            intensities=intensities,
            intensity_stats=intensity_stats,
        )

    rvec = tvec = np.zeros(3)
    for obj in annotations:
        corners = np.array(
            [
                [-obj["l"] / 2, -obj["w"] / 2, -obj["h"] / 2],
                [obj["l"] / 2, -obj["w"] / 2, -obj["h"] / 2],
                [obj["l"] / 2, obj["w"] / 2, -obj["h"] / 2],
                [-obj["l"] / 2, obj["w"] / 2, -obj["h"] / 2],
                [-obj["l"] / 2, -obj["w"] / 2, obj["h"] / 2],
                [obj["l"] / 2, -obj["w"] / 2, obj["h"] / 2],
                [obj["l"] / 2, obj["w"] / 2, obj["h"] / 2],
                [-obj["l"] / 2, obj["w"] / 2, obj["h"] / 2],
            ]
        )
        rot = R.from_euler("z", obj["yaw"]).as_matrix()
        corners = (rot @ corners.T).T + [obj["x"], obj["y"], obj["z"]]
        corners_hom = np.hstack((corners, np.ones((8, 1)))).T
        corners_cam = Rt @ corners_hom

        if np.any(corners_cam[2, :] <= 0.1):
            continue

        pts = corners_cam[:3, :].T[:, np.newaxis].astype(np.float32)
        if is_fisheye:
            uv, _ = cv2.fisheye.projectPoints(
                pts, rvec, tvec, K.astype(np.float32), D.astype(np.float32)
            )
        else:
            uv, _ = cv2.projectPoints(pts, rvec, tvec, K.astype(np.float32), D)

        uv_flat = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(uv_flat).all() or np.abs(uv_flat).max() >= 1e6:
            continue
        uv = np.round(uv_flat).astype(np.int32)

        color = CLASS_COLORS[obj["class_name"]]
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        for i, j in edges:
            cv2.line(out_img, tuple(uv[i]), tuple(uv[j]), color, 2)

    resized = cv2.resize(out_img, (CELL_WIDTH, CELL_HEIGHT), interpolation=cv2.INTER_AREA)
    canvas[y_start : y_start + CELL_HEIGHT, x_start : x_start + CELL_WIDTH] = resized
    cv2.putText(
        canvas,
        cam_name,
        (x_start + padding_left, y_start + padding_top),
        font,
        font_scale,
        text_color,
        font_thickness,
    )

cv2.putText(canvas, TIMESTAMP, (OUTPUT_WIDTH - 500, OUTPUT_HEIGHT - 30), font, 0.8, (255, 255, 255), 1)

window_name = f"Multi-View Projection | {SEQUENCE_FOLDER.name} | {TIMESTAMP}"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow(window_name, canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
