"""
Tutorial 11: All-in-one visualization for a single timestamp.
Generates:
- Multi-view grids (original image, projected points, projected boxes, projected points+boxes)
- Side-by-side before/after per camera
- Bird's Eye View (BEV) with boxes
Saves all results to a timestamped folder.

"""

import cv2
import numpy as np
import json
import laspy
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from phenikaa_paths import (
    CAMERAS,
    EXTRINSIC_JSON,
    INTRINSIC_JSON,
    RESULT_ROOT,
    SEQUENCE_FOLDER,
    TIMESTAMP,
)
from lidar_draw import draw_lidar_overlay
from lidar_color import compute_point_colors, depths_from_camera_points
from lidar_viz_config import (
    CLASS_COLORS,
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
    SAVE_SIZE,
)
from projection_utils import intensity_stats_from_las

RESULT_ROOT = RESULT_ROOT / SEQUENCE_FOLDER.name / TIMESTAMP
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

# Load calibrations
print("Loading calibration files...")
with open(INTRINSIC_JSON) as f:
    INTRINSICS = json.load(f)
with open(EXTRINSIC_JSON) as f:
    EXTRINSICS = json.load(f)
print("Calibrations loaded.")

# ==================== UTILS ====================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def draw_camera_name(img, cam_name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    x, y = 20, 60
    cv2.putText(img, cam_name, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, cam_name, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img

def resize_for_output(img, size, *, sharp_points=False):
    """INTER_LINEAR giữ điểm LiDAR sắc hơn khi downscale (tránh INTER_AREA làm mờ)."""
    interp = cv2.INTER_LINEAR if sharp_points else cv2.INTER_AREA
    return cv2.resize(img, size, interpolation=interp)

def create_multi_view_grid(images_dict, *, sharp_points=False):
    GRID_COLS = 5
    GRID_ROWS = 2
    OUTPUT_WIDTH = 1920
    OUTPUT_HEIGHT = 720
    CELL_WIDTH = OUTPUT_WIDTH // GRID_COLS   # 384
    CELL_HEIGHT = OUTPUT_HEIGHT // GRID_ROWS # 360

    canvas = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)  # Dark background

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 1
    text_color = (255, 255, 255)

    for idx, cam_name in enumerate(CAMERAS):
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        x_start = col * CELL_WIDTH
        y_start = row * CELL_HEIGHT

        img = images_dict.get(cam_name)
        if img is not None:
            resized = resize_for_output(img, (CELL_WIDTH, CELL_HEIGHT), sharp_points=sharp_points)
            canvas[y_start:y_start+CELL_HEIGHT, x_start:x_start+CELL_WIDTH] = resized

        # Camera label
        cv2.putText(canvas, cam_name, (x_start + 12, y_start + 40), font, font_scale, text_color, thickness)

    return canvas

# ==================== MAIN PROJECTOR CLASS ====================
class LidarCameraProjector:
    latest_points = None
    latest_intensity = None
    intensity_stats = None
    total_points = 0
    annotations = []

    def __init__(self, sequence_folder: Path, timestamp: str, cam_name: str):
        self.sequence_folder = sequence_folder
        self.timestamp = timestamp
        self.cam_name = cam_name
        self.is_fisheye = cam_name.startswith("CAM_F")
        self.mode = 'distorted' if self.is_fisheye else 'undistorted'

        self.jpg_path = sequence_folder / "Image" / cam_name / f"{timestamp}.jpg"
        self.laz_path = sequence_folder / "Lidar" / f"{timestamp}.laz"
        self.ann_path = sequence_folder / "Label" / f"{timestamp}.txt"

        self.load_image_and_setup()
        self.load_lidar_once()
        self.load_annotations_once()

    def load_image_and_setup(self):
        if not self.jpg_path.exists():
            raise FileNotFoundError(f"Image not found: {self.jpg_path}")

        raw_img = cv2.imread(str(self.jpg_path))
        h, w = raw_img.shape[:2]

        cam_calib = INTRINSICS[self.cam_name]
        K = np.array(cam_calib["camera_matrix"], dtype=np.float64).reshape(3, 3)
        D = np.array(cam_calib["distortion_coefficients"], dtype=np.float64)
        if self.is_fisheye and len(D) > 4:
            D = D[:4]

        Rt = np.array(EXTRINSICS[self.cam_name], dtype=np.float64)[:3, :]

        if self.is_fisheye and self.mode == 'distorted':
            self.img = raw_img.copy()
            self.K_proj = K.copy()
            self.D_proj = D.copy()
        else:
            if self.is_fisheye:
                K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.3)
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, (w, h), cv2.CV_32FC1)
                self.img = cv2.remap(raw_img, map1, map2, cv2.INTER_LINEAR)
                self.K_proj = K_new
                self.D_proj = np.zeros(4, dtype=np.float32)
            else:
                self.img = cv2.undistort(raw_img, K, D)
                self.K_proj = K.copy()
                self.D_proj = np.zeros_like(D)

        self.Rt = Rt
        self.h, self.w = self.img.shape[:2]

    def load_lidar_once(self):
        if LidarCameraProjector.latest_points is None:
            if not self.laz_path.exists():
                raise FileNotFoundError(f"LiDAR file not found: {self.laz_path}")
            print(f"Loading LiDAR points from {self.timestamp}.laz ...")
            las = laspy.read(str(self.laz_path))
            LidarCameraProjector.latest_points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
            if "intensity" in las.point_format.dimension_names:
                LidarCameraProjector.latest_intensity, LidarCameraProjector.intensity_stats = (
                    intensity_stats_from_las(las)
                )
                s = LidarCameraProjector.intensity_stats
                print(
                    f"  Intensity (frame): min={s['min']:.0f} max={s['max']:.0f} "
                    f"mean={s['mean']:.1f} p{LIDAR_INTENSITY_PERCENTILE_LOW:g}={s['p_lo']:.0f} "
                    f"p{LIDAR_INTENSITY_PERCENTILE_HIGH:g}={s['p_hi']:.0f}"
                )
            else:
                LidarCameraProjector.latest_intensity = None
                LidarCameraProjector.intensity_stats = None
                print("  Warning: no intensity channel in .laz — use LIDAR_COLOR_MODE='depth' or 'mono'")
            LidarCameraProjector.total_points = len(LidarCameraProjector.latest_points)
            print(f" → {LidarCameraProjector.total_points:,} points loaded.")
        self.points = LidarCameraProjector.latest_points
        self.intensities = LidarCameraProjector.latest_intensity

    def load_annotations_once(self):
        if not LidarCameraProjector.annotations and self.ann_path.exists():
            print(f"Loading annotations from {self.timestamp}.txt ...")
            with open(self.ann_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 8: continue
                    try:
                        x, y, z, l, w, h, yaw = map(float, parts[:7])
                        class_name = ' '.join(parts[7:]).title()
                        if class_name in CLASS_COLORS:
                            LidarCameraProjector.annotations.append({
                                'x': x, 'y': y, 'z': z,
                                'l': l, 'w': w, 'h': h,
                                'yaw': yaw, 'class_name': class_name
                            })
                    except:
                        continue
            print(f" → {len(LidarCameraProjector.annotations)} annotated objects.")

    def project_points(self):
        if len(self.points) == 0:
            return self.img.copy()

        points_hom = np.hstack((self.points, np.ones((len(self.points), 1))))
        points_cam = (self.Rt @ points_hom.T).T
        mask = points_cam[:, 2] > 0.1
        points_cam = points_cam[mask]
        intensities = (
            self.intensities[mask]
            if self.intensities is not None
            else None
        )

        depths = depths_from_camera_points(points_cam, DEPTH_JET_METRIC)

        pts_3d = points_cam[:, :3][:, np.newaxis, :].astype(np.float32)
        rvec = tvec = np.zeros(3, dtype=np.float32)

        if self.is_fisheye and self.mode == 'distorted':
            uv, _ = cv2.fisheye.projectPoints(pts_3d, rvec, tvec, self.K_proj.astype(np.float32), self.D_proj.astype(np.float32))
        else:
            uv, _ = cv2.projectPoints(pts_3d, rvec, tvec, self.K_proj.astype(np.float32), self.D_proj)

        uv = uv.reshape(-1, 2).round().astype(int)
        colors = compute_point_colors(
            len(uv),
            LIDAR_COLOR_MODE,
            depths=depths,
            intensities=intensities,
            intensity_stats=LidarCameraProjector.intensity_stats,
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

        out, valid_count = draw_lidar_overlay(
            self.img,
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

        print(
            f"  {self.cam_name}: {valid_count:,}/{LidarCameraProjector.total_points:,} "
            f"points projected ({LIDAR_DRAW_STYLE}, color={LIDAR_COLOR_MODE})"
        )
        return out

    def project_boxes(self, img):
        out = img.copy()
        visible = 0
        rvec = tvec = np.zeros(3, dtype=np.float32)

        for obj in LidarCameraProjector.annotations:
            corners = np.array([
                [-obj['l']/2, -obj['w']/2, -obj['h']/2], [obj['l']/2, -obj['w']/2, -obj['h']/2],
                [obj['l']/2, obj['w']/2, -obj['h']/2], [-obj['l']/2, obj['w']/2, -obj['h']/2],
                [-obj['l']/2, -obj['w']/2, obj['h']/2], [obj['l']/2, -obj['w']/2, obj['h']/2],
                [obj['l']/2, obj['w']/2, obj['h']/2], [-obj['l']/2, obj['w']/2, obj['h']/2]
            ])
            rot = R.from_euler('z', obj['yaw']).as_matrix()
            corners = (rot @ corners.T).T + [obj['x'], obj['y'], obj['z']]
            corners_hom = np.hstack((corners, np.ones((8, 1)))).T
            corners_cam = self.Rt @ corners_hom

            if np.any(corners_cam[2, :] <= 0.1):
                continue

            pts = corners_cam[:3, :].T[:, np.newaxis].astype(np.float32)
            if self.is_fisheye and self.mode == 'distorted':
                uv, _ = cv2.fisheye.projectPoints(pts, rvec, tvec, self.K_proj.astype(np.float32), self.D_proj.astype(np.float32))
            else:
                uv, _ = cv2.projectPoints(pts, rvec, tvec, self.K_proj.astype(np.float32), self.D_proj)

            uv = uv.reshape(-1, 2).round().astype(int)
            color = CLASS_COLORS[obj['class_name']]
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for i, j in edges:
                cv2.line(out, tuple(uv[i]), tuple(uv[j]), color, 2)
            visible += 1

        return out, visible

    def process(self):
        points_img = self.project_points()
        boxes_img, visible_boxes = self.project_boxes(self.img.copy())
        combined_img, _ = self.project_boxes(points_img.copy())

        print(f"  {self.cam_name}: {visible_boxes}/{len(LidarCameraProjector.annotations)} boxes visible")

        fixed_size = SAVE_SIZE

        # Individual saves
        for name, img in [
            ("original_image", self.img),
            ("projected_point", points_img),
            ("projected_box", boxes_img),
            ("projected_point&box", combined_img)
        ]:
            folder = RESULT_ROOT / name
            ensure_dir(folder)
            sharp = "point" in name
            cv2.imwrite(
                str(folder / f"{self.cam_name}.jpg"),
                resize_for_output(img, fixed_size, sharp_points=sharp),
            )

        # Before-after with camera name
        left = resize_for_output(self.img, fixed_size)
        left = draw_camera_name(left, self.cam_name)
        right = resize_for_output(combined_img, fixed_size, sharp_points=True)
        preview = np.hstack((left, right))

        preview_dir = RESULT_ROOT / "before_after"
        ensure_dir(preview_dir)
        cv2.imwrite(str(preview_dir / f"{self.cam_name}.jpg"), preview)

        return {
            "original": self.img,
            "points": points_img,
            "boxes": boxes_img,
            "combined": combined_img,
            "preview": preview
        }

# ==================== BEV ====================
def create_bev():
    points = LidarCameraProjector.latest_points
    annotations = LidarCameraProjector.annotations
    if points is None or len(points) == 0:
        print("No LiDAR points for BEV.")
        return

    grid_size = 0.05
    x_min, x_max = points[:, 0].min() - 10, points[:, 0].max() + 10
    y_min, y_max = points[:, 1].min() - 10, points[:, 1].max() + 10
    cols = int(np.ceil((x_max - x_min) / grid_size))
    rows = int(np.ceil((y_max - y_min) / grid_size))

    bev = np.zeros((rows, cols, 3), dtype=np.uint8)
    col_idx = ((points[:, 0] - x_min) / grid_size).astype(int)
    row_idx = ((y_max - points[:, 1]) / grid_size).astype(int)
    valid = (col_idx >= 0) & (col_idx < cols) & (row_idx >= 0) & (row_idx < rows)
    bev[row_idx[valid], col_idx[valid]] = [255, 255, 255]

    for obj in annotations:
        color = CLASS_COLORS[obj['class_name']]
        cx, cy = obj['x'], obj['y']
        l, w, yaw = obj['l'], obj['w'], obj['yaw']

        corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners = corners @ rot.T + [cx, cy]
        pts = np.array([[int((x - x_min)/grid_size), int((y_max - y)/grid_size)] for x, y in corners], np.int32)
        cv2.polylines(bev, [pts], True, color, thickness=3)

        # Direction arrow
        tip_x = cx + np.cos(yaw) * (l / 2 + 1.5)
        tip_y = cy + np.sin(yaw) * (l / 2 + 1.5)
        cv2.arrowedLine(bev,
                        (int((cx - x_min)/grid_size), int((y_max - cy)/grid_size)),
                        (int((tip_x - x_min)/grid_size), int((y_max - tip_y)/grid_size)),
                        color, thickness=2, tipLength=0.2)

    cv2.imwrite(str(RESULT_ROOT / f"lidar_bev_{grid_size:.3f}_horizontal.png"), bev)
    cv2.imwrite(str(RESULT_ROOT / f"lidar_bev_{grid_size:.3f}_vertical.png"), cv2.rotate(bev, cv2.ROTATE_90_COUNTERCLOCKWISE))
    print("BEV images saved.")

# ==================== MAIN ====================
def main():
    LidarCameraProjector.latest_points = None
    LidarCameraProjector.latest_intensity = None
    LidarCameraProjector.intensity_stats = None
    LidarCameraProjector.annotations = []

    print(f"\nProcessing timestamp: {TIMESTAMP}")
    print(
        f"LiDAR color: {LIDAR_COLOR_MODE} (norm={LIDAR_INTENSITY_NORM}, "
        f"cmap={LIDAR_INTENSITY_COLORMAP}) | draw: {LIDAR_DRAW_STYLE}"
    )
    print(f"Output folder: {RESULT_ROOT}")
    print("="*80)

    originals = {}
    points_dict = {}
    boxes_dict = {}
    combined_dict = {}
    previews = []

    for cam_name in CAMERAS:
        try:
            projector = LidarCameraProjector(SEQUENCE_FOLDER, TIMESTAMP, cam_name)
            result = projector.process()

            originals[cam_name] = result["original"]
            points_dict[cam_name] = result["points"]
            boxes_dict[cam_name] = result["boxes"]
            combined_dict[cam_name] = result["combined"]
            previews.append(result["preview"])
        except Exception as e:
            print(f"ERROR {cam_name}: {e}")
            blank = np.zeros((1080, 1920, 3), np.uint8)
            blank_preview = np.zeros((960, 1920, 3), np.uint8)
            for d in [originals, points_dict, boxes_dict, combined_dict]:
                d[cam_name] = blank
            previews.append(blank_preview)

    # Save multi-view grids
    cv2.imwrite(str(RESULT_ROOT / "original_image_grid.jpg"), create_multi_view_grid(originals))
    cv2.imwrite(str(RESULT_ROOT / "projected_point_grid.jpg"), create_multi_view_grid(points_dict, sharp_points=True))
    cv2.imwrite(str(RESULT_ROOT / "projected_box_grid.jpg"), create_multi_view_grid(boxes_dict))
    cv2.imwrite(str(RESULT_ROOT / "projected_point&box_grid.jpg"), create_multi_view_grid(combined_dict, sharp_points=True))

    # Before-after big grid (2 rows × 5 cols)
    before_after_big = np.vstack([np.hstack(previews[:5]), np.hstack(previews[5:])])
    cv2.imwrite(str(RESULT_ROOT / "before_and_after.jpg"), before_after_big)

    # BEV
    create_bev()

    print("\nAll visualizations completed!")
    print(f"Results saved to:\n{RESULT_ROOT}")

if __name__ == "__main__":
    main()