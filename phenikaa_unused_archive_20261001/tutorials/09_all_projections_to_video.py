"""
Tutorial 09: Create a video with 3D box projections on all cameras for the entire sequence.
Uses parallel processing for efficiency.

"""

import cv2
import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from phenikaa_paths import (
    CAMERAS,
    EXTRINSIC_JSON,
    INTRINSIC_JSON,
    RESULT_ROOT,
    SEQUENCE_FOLDER,
    TIMESTAMPS,
)
from lidar_viz_config import CLASS_COLORS

IMAGE_ROOT = SEQUENCE_FOLDER / "Image"
LABEL_ROOT = SEQUENCE_FOLDER / "Label"

RESULT_ROOT.mkdir(parents=True, exist_ok=True)

sequence_name = SEQUENCE_FOLDER.name.replace(" ", "_")
OUTPUT_VIDEO = RESULT_ROOT / f"{sequence_name}_projected_3d_box_5x2.mp4"
FPS = 10

# Grid
GRID_COLS = 5
GRID_ROWS = 2
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 720
CELL_WIDTH = OUTPUT_WIDTH // GRID_COLS  # 384
CELL_HEIGHT = OUTPUT_HEIGHT // GRID_ROWS  # 360

# Load calibrations
with open(INTRINSIC_JSON) as f:
    INTRINSICS = json.load(f)
with open(EXTRINSIC_JSON) as f:
    EXTRINSICS = json.load(f)

# Text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_thickness = 1
text_color = (255, 255, 255)
padding_left = 12
padding_top = 40
ts_font_scale = 0.8
ts_thickness = 1
ts_color = (255, 255, 255)

# Processing function
def process_camera(args):
    cam_idx, cam_name, ts, annotations = args
    jpg_path = IMAGE_ROOT / cam_name / f"{ts}.jpg"

    row = cam_idx // GRID_COLS
    col = cam_idx % GRID_COLS
    x_start = col * CELL_WIDTH
    y_start = row * CELL_HEIGHT

    if not jpg_path.exists():
        return (x_start, y_start, cam_name, None, True)

    img = cv2.imread(str(jpg_path))
    if img is None:
        return (x_start, y_start, cam_name, None, True)

    h, w = img.shape[:2]
    is_fisheye = cam_name.startswith('CAM_F')

    K = np.array(INTRINSICS[cam_name]["camera_matrix"], dtype=np.float64).reshape(3, 3)
    D = np.array(INTRINSICS[cam_name]["distortion_coefficients"], dtype=np.float64)
    if is_fisheye and len(D) > 4:
        D = D[:4]
    Rt = np.array(EXTRINSICS[cam_name], dtype=np.float64)[:3, :]

    out_img = img.copy()
    rvec = tvec = np.zeros(3)

    for obj in annotations:
        corners = np.array([
            [-obj['l']/2, -obj['w']/2, -obj['h']/2], [obj['l']/2, -obj['w']/2, -obj['h']/2],
            [obj['l']/2, obj['w']/2, -obj['h']/2], [-obj['l']/2, obj['w']/2, -obj['h']/2],
            [-obj['l']/2, -obj['w']/2, obj['h']/2], [obj['l']/2, -obj['w']/2, obj['h']/2],
            [obj['l']/2, obj['w']/2, obj['h']/2], [-obj['l']/2, obj['w']/2, obj['h']/2]
        ])

        rot = R.from_euler('z', obj['yaw']).as_matrix()
        corners = (rot @ corners.T).T + [obj['x'], obj['y'], obj['z']]
        corners_hom = np.hstack((corners, np.ones((8,1)))).T
        corners_cam = Rt @ corners_hom

        if np.any(corners_cam[2, :] <= 0.1):
            continue

        pts = corners_cam[:3, :].T[:, np.newaxis].astype(np.float32)

        if is_fisheye:
            uv, _ = cv2.fisheye.projectPoints(pts, rvec, tvec, K.astype(np.float32), D.astype(np.float32))
        else:
            uv, _ = cv2.projectPoints(pts, rvec, tvec, K.astype(np.float32), D)

        uv = np.round(uv.reshape(-1, 2)).astype(np.int32)

        color = CLASS_COLORS[obj['class_name']]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for i, j in edges:
            cv2.line(out_img, tuple(uv[i]), tuple(uv[j]), color, 2)

    resized = cv2.resize(out_img, (CELL_WIDTH, CELL_HEIGHT), interpolation=cv2.INTER_AREA)
    return (x_start, y_start, cam_name, resized, False)

timestamps = TIMESTAMPS
print(
    f"Sequence {SEQUENCE_FOLDER.name}: {len(timestamps)} frames "
    f"({timestamps[0]} → {timestamps[-1]})"
)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, FPS, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)

frame_count = 0
for ts in timestamps:
    frame_count += 1
    print(f"Processing frame {frame_count}/{len(timestamps)} → {ts}", end="\r")

    ann_path = LABEL_ROOT / f"{ts}.txt"
    annotations = []
    if ann_path.exists():
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                x, y, z, l, w, h, yaw = map(float, parts[:7])
                class_name = ' '.join(parts[7:]).title()
                if class_name in CLASS_COLORS:
                    annotations.append({'x': x, 'y': y, 'z': z, 'l': l, 'w': w, 'h': h, 'yaw': yaw, 'class_name': class_name})

    tasks = [(idx, cam, ts, annotations) for idx, cam in enumerate(CAMERAS)]

    canvas = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_camera, task) for task in tasks]
        for future in as_completed(futures):
            x_start, y_start, cam_name, resized, is_missing = future.result()
            if resized is not None:
                canvas[y_start:y_start + CELL_HEIGHT, x_start:x_start + CELL_WIDTH] = resized

            label_color = text_color if not is_missing else (0, 0, 255)
            cv2.putText(canvas, cam_name, (x_start + padding_left, y_start + padding_top),
                        font, font_scale, label_color, font_thickness)

            if is_missing:
                cv2.putText(canvas, "MISSING", (x_start + padding_left, y_start + padding_top + 40),
                            font, font_scale, (0, 0, 255), font_thickness)

    cv2.putText(canvas, ts, (OUTPUT_WIDTH - 500, OUTPUT_HEIGHT - 30),
                font, ts_font_scale, ts_color, ts_thickness)

    video_writer.write(canvas)

video_writer.release()
print("\nVideo created with 3D box projections.")
