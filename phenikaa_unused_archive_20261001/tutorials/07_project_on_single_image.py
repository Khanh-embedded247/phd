"""
Tutorial 07: Project LiDAR points and 3D bounding boxes onto a single camera image.
Displays side-by-side: original image (left) and projection overlay (right).

"""

import cv2
import numpy as np
import laspy
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from phenikaa_paths import (
    EXTRINSIC_JSON,
    INTRINSIC_JSON,
    SEQUENCE_FOLDER,
    TIMESTAMP,
)
from lidar_viz_config import CLASS_COLORS, SAVE_SIZE
from projection_utils import intensity_stats_from_las, paint_lidar_on_image

# ==================== CONFIGURATION ====================
CAMERA = "CAM_P_FL"  # Change to test other cameras

# ==================== LOAD CALIBRATION ====================
print("Loading calibration files...")
with open(INTRINSIC_JSON) as f:
    INTRINSICS = json.load(f)
with open(EXTRINSIC_JSON) as f:
    EXTRINSICS = json.load(f)

# ==================== DATA PATHS ====================
laz_path = SEQUENCE_FOLDER / "Lidar" / f"{TIMESTAMP}.laz"
jpg_path = SEQUENCE_FOLDER / "Image" / CAMERA / f"{TIMESTAMP}.jpg"
ann_path = SEQUENCE_FOLDER / "Label" / f"{TIMESTAMP}.txt"

if not jpg_path.exists():
    raise FileNotFoundError(f"Image not found: {jpg_path}")
if not laz_path.exists():
    raise FileNotFoundError(f"LiDAR not found: {laz_path}")

print(f"Processing {CAMERA} @ {TIMESTAMP}")

# Load image
img = cv2.imread(str(jpg_path))

# Load LiDAR
print("  Loading LiDAR point cloud...")
las = laspy.read(str(laz_path))
points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
intensities, intensity_stats = intensity_stats_from_las(las)
print(f"  → {len(points):,} points loaded")

# Load annotations
annotations = []
if ann_path.exists():
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                x, y, z, l, w, h, yaw = map(float, parts[:7])
                class_name = ' '.join(parts[7:]).title()
                if class_name in CLASS_COLORS:
                    annotations.append({
                        'x': x, 'y': y, 'z': z,
                        'l': l, 'w': w, 'h': h,
                        'yaw': yaw, 'class_name': class_name
                    })
            except:
                continue
    print(f"  → {len(annotations)} annotated objects loaded")

# ==================== CAMERA SETUP ====================
cam_data = INTRINSICS[CAMERA]
K = np.array(cam_data["camera_matrix"], dtype=np.float64).reshape(3, 3)
D = np.array(cam_data["distortion_coefficients"], dtype=np.float64)
if CAMERA.startswith('CAM_F') and len(D) > 4:
    D = D[:4]

Rt = np.array(EXTRINSICS[CAMERA], dtype=np.float64)[:3, :]  # Extrinsic: lidar → camera
is_fisheye = CAMERA.startswith('CAM_F')

h, w = img.shape[:2]

# Optional: undistort image (recommended for clean display)
if is_fisheye:
    out = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.0)
    K_new = out[0] if isinstance(out, (tuple, list)) else out
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, (w, h), cv2.CV_16SC2)
else:
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.0)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), K_new, (w, h), cv2.CV_16SC2)

img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

D_proj = D.astype(np.float32) if is_fisheye else np.zeros(4, dtype=np.float32)

# ==================== PROJECT LIDAR POINTS ====================
print("Projecting LiDAR points...")
out_points, n_pts = paint_lidar_on_image(
    img,
    points,
    Rt,
    K_new,
    D_proj,
    is_fisheye=is_fisheye,
    intensities=intensities,
    intensity_stats=intensity_stats,
)
print(f"  → {n_pts:,}/{len(points):,} points projected")

# ==================== PROJECT 3D BOXES ====================
print("Projecting 3D bounding boxes...")
out_boxes = img.copy()
visible_boxes = 0
rvec = tvec = np.zeros(3)

for obj in annotations:
    corners = np.array([
        [-obj['l']/2, -obj['w']/2, -obj['h']/2], [ obj['l']/2, -obj['w']/2, -obj['h']/2],
        [ obj['l']/2,  obj['w']/2, -obj['h']/2], [-obj['l']/2,  obj['w']/2, -obj['h']/2],
        [-obj['l']/2, -obj['w']/2,  obj['h']/2], [ obj['l']/2, -obj['w']/2,  obj['h']/2],
        [ obj['l']/2,  obj['w']/2,  obj['h']/2], [-obj['l']/2,  obj['w']/2,  obj['h']/2]
    ])

    rot = R.from_euler('z', obj['yaw']).as_matrix()
    corners = (rot @ corners.T).T + [obj['x'], obj['y'], obj['z']]
    corners_hom = np.hstack((corners, np.ones((8, 1)))).T
    corners_cam = Rt @ corners_hom

    if np.any(corners_cam[2, :] <= 0.1):
        continue

    pts = corners_cam[:3, :].T[:, np.newaxis].astype(np.float32)

    if is_fisheye:
        uv, _ = cv2.fisheye.projectPoints(pts, rvec, tvec, K_new.astype(np.float32), D.astype(np.float32))
    else:
        uv, _ = cv2.projectPoints(pts, rvec, tvec, K_new.astype(np.float32), np.zeros(4))

    uv = np.round(uv.reshape(-1, 2)).astype(int)

    # Check visibility
    if not np.any((uv[:,0] >= 0) & (uv[:,0] < w) & (uv[:,1] >= 0) & (uv[:,1] < h)):
        continue

    visible_boxes += 1
    color = CLASS_COLORS[obj['class_name']]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        cv2.line(out_boxes, tuple(uv[i]), tuple(uv[j]), color, 2)

print(f"  → {visible_boxes}/{len(annotations)} boxes visible")

# Combine points + boxes
combined = cv2.addWeighted(out_points, 0.6, out_boxes, 1.0, 0)

# ==================== DISPLAY RESULT ====================
fixed_size = SAVE_SIZE
left = cv2.resize(img, fixed_size)
right = cv2.resize(combined, fixed_size)

# Add camera name
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(left, CAMERA, (30, 60), font, 1.5, (0, 0, 0), 4, cv2.LINE_AA)
cv2.putText(left, CAMERA, (30, 60), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

preview = cv2.hconcat([left, right])

window_name = f"{CAMERA} | Left: Original | Right: LiDAR Points + 3D Boxes | {TIMESTAMP}"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow(window_name, preview)

print("\n" + "="*80)
print("Preview displayed — close window to exit")
print("="*80)

cv2.waitKey(0)
cv2.destroyAllWindows()
print("Done!")

