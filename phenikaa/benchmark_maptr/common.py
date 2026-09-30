#!/usr/bin/env python3
"""
Các hàm dùng chung cho benchmark HD map.

Quy ước chính của benchmark:
- Hệ cuối cùng để benchmark/vector map là hệ `map` trong `traj_lidar.txt`.
- `traj_lidar.txt` được hiểu là pose `T_map_lidar`.
- `lidar_at_t_cam` chỉ là hệ tạm để đồng bộ LiDAR về đúng thời điểm camera.
- Camera-LiDAR projection luôn dùng ảnh đã undistort, K mới, và D = 0.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import laspy
import numpy as np


PHENIKAA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = "Normal"
DEFAULT_CAMERA = "CAM_P_F"
DEFAULT_LIDAR = "LIDAR_TOP"

CALIB_ROOT = PHENIKAA_ROOT / "calib" / "vf_06_02"
INTRINSIC_JSON = CALIB_ROOT / "VF6_02_Intrinsics.json"
EXTRINSIC_JSON = CALIB_ROOT / "VF6_02_Extrinsics_By_Dates.json"
OUTPUT_ROOT = PHENIKAA_ROOT / "outputs" / "benchmark"

# Camera pinhole CAM_P_* giữ FOV rộng như trước vì ảnh sau undistort vẫn ổn.
PINHOLE_UNDISTORT_BALANCE = 1.0

# Camera fisheye CAM_F_* nếu để balance=1.0 sẽ giữ FOV quá rộng và kéo mép ảnh mạnh.
# balance=0.0 crop nhiều hơn nhưng ảnh phẳng, rõ, hợp hơn cho MapTR học từ ảnh.
FISHEYE_UNDISTORT_BALANCE = 0.0


@dataclass(frozen=True)
class Pose:
    """Một pose LiDAR/xe trong map tại một timestamp."""

    timestamp: float
    T_map_lidar: np.ndarray


@dataclass(frozen=True)
class Trajectory:
    """Trajectory đã cache timestamp để nội suy nhanh."""

    poses: list[Pose]
    timestamps: np.ndarray


def timestamp_from_name(path: Path | str) -> float:
    """Đọc timestamp từ tên file dạng sec-nsec.ext hoặc dạng số thực."""
    stem = Path(path).stem
    if "-" in stem:
        sec, nsec = stem.split("-", 1)
        return int(sec) + int(nsec) / (10 ** len(nsec))
    return float(stem)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    """Đổi quaternion [x, y, z, w] sang ma trận quay 3x3."""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        return np.eye(3)
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Đổi ma trận quay 3x3 sang quaternion [x, y, z, w]."""
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / np.linalg.norm(q)


def slerp_quat(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Nội suy quay bằng SLERP để pose ở giữa 2 timestamp không bị méo góc."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def make_transform(translation: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Tạo ma trận transform 4x4 từ translation và quaternion."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rot(quat_xyzw)
    T[:3, 3] = translation
    return T


def load_lidar_trajectory(path: Path) -> list[Pose]:
    """
    Đọc traj_lidar.txt.

    Format mỗi dòng:
        timestamp tx ty tz qx qy qz qw

    Theo bài toán hiện tại, pose này được hiểu là:
        T_map_lidar
    tức biến điểm từ hệ LiDAR tại timestamp đó sang hệ map.
    """
    poses: list[Pose] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            vals = [float(v) for v in text.split()]
            if len(vals) != 8:
                raise ValueError(f"Expected 8 columns in {path}, got: {text}")
            ts = vals[0]
            t = np.asarray(vals[1:4], dtype=np.float64)
            q = np.asarray(vals[4:8], dtype=np.float64)
            poses.append(Pose(ts, make_transform(t, q)))
    poses.sort(key=lambda p: p.timestamp)
    return poses


def load_trajectory(path: Path) -> Trajectory:
    """Đọc trajectory và cache mảng timestamp để lookup nhanh hơn."""
    poses = load_lidar_trajectory(path)
    return Trajectory(
        poses=poses,
        timestamps=np.asarray([p.timestamp for p in poses], dtype=np.float64),
    )


def interpolate_pose(poses: list[Pose] | Trajectory, timestamp: float) -> np.ndarray:
    """
    Nội suy T_map_lidar tại timestamp bất kỳ.

    Translation dùng nội suy tuyến tính.
    Rotation dùng SLERP quaternion.
    """
    if isinstance(poses, Trajectory):
        timestamps = poses.timestamps
        pose_list = poses.poses
    else:
        pose_list = poses
        timestamps = np.asarray([p.timestamp for p in pose_list], dtype=np.float64)

    if not pose_list:
        raise ValueError("Empty trajectory")
    idx = int(np.searchsorted(timestamps, timestamp, side="left"))
    if idx == 0:
        return pose_list[0].T_map_lidar.copy()
    if idx >= len(pose_list):
        return pose_list[-1].T_map_lidar.copy()
    p0 = pose_list[idx - 1]
    p1 = pose_list[idx]
    alpha = (timestamp - p0.timestamp) / (p1.timestamp - p0.timestamp)
    t = (1.0 - alpha) * p0.T_map_lidar[:3, 3] + alpha * p1.T_map_lidar[:3, 3]
    q0 = rot_to_quat_xyzw(p0.T_map_lidar[:3, :3])
    q1 = rot_to_quat_xyzw(p1.T_map_lidar[:3, :3])
    q = slerp_quat(q0, q1, alpha)
    return make_transform(t, q)


def transform_points(T: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    """
    Transform điểm 3D bằng ma trận 4x4.

    Nếu T = T_a_b thì:
        P_a = T_a_b @ P_b
    """
    return points_xyz @ T[:3, :3].T + T[:3, 3]


def load_laz_xyz_intensity(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Đọc LAZ thành XYZ và intensity nếu có."""
    las = laspy.read(str(path))
    points = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    intensity = None
    if hasattr(las, "intensity"):
        intensity = np.asarray(las.intensity, dtype=np.float64)
    return points, intensity


def nearest_path(timestamp: float, paths: list[Path]) -> tuple[Path, float]:
    """Tìm file có timestamp gần nhất với timestamp đầu vào."""
    if not paths:
        raise ValueError("No candidate paths")
    times = [timestamp_from_name(p) for p in paths]
    idx = bisect.bisect_left(times, timestamp)
    candidates = []
    if idx > 0:
        candidates.append((abs(times[idx - 1] - timestamp), paths[idx - 1]))
    if idx < len(paths):
        candidates.append((abs(times[idx] - timestamp), paths[idx]))
    dt, path = min(candidates, key=lambda x: x[0])
    return path, dt


def load_intrinsic(camera: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Đọc K, D và kích thước ảnh calib của camera từ VF6_02_Intrinsics.json."""
    data = load_json(INTRINSIC_JSON)
    calib = data[camera]
    K = np.asarray(calib["camera_matrix"], dtype=np.float64)
    D = np.asarray(calib["distortion_coefficients"], dtype=np.float64).reshape(-1)
    return K, D, int(calib["image_width"]), int(calib["image_height"])


def scale_camera_matrix(
    K: np.ndarray,
    calib_width: int,
    calib_height: int,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """
    Scale K khi ảnh thực tế khác kích thước ảnh dùng lúc calib.

    Ví dụ calib 1920x1536 nhưng ảnh input 1280x1024 thì phải scale fx/fy/cx/cy.
    D không scale.
    """
    K_scaled = K.copy()
    K_scaled[0, :] *= image_width / calib_width
    K_scaled[1, :] *= image_height / calib_height
    return K_scaled


def undistort_for_projection(
    image: np.ndarray,
    K_calib: np.ndarray,
    D: np.ndarray,
    calib_width: int,
    calib_height: int,
    is_fisheye: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Undistort ảnh và trả về ảnh đã nắn, K dùng để project, D = 0.

    Sau khi ảnh đã undistort, projection không được apply distortion lần nữa.
    """
    h, w = image.shape[:2]
    K_scaled = scale_camera_matrix(K_calib, calib_width, calib_height, w, h)
    if is_fisheye:
        D_use = D[:4]
        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_scaled, D_use, (w, h), np.eye(3), balance=FISHEYE_UNDISTORT_BALANCE
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K_scaled, D_use, np.eye(3), K_new, (w, h), cv2.CV_16SC2
        )
        image_new = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    else:
        K_new, _ = cv2.getOptimalNewCameraMatrix(
            K_scaled, D, (w, h), PINHOLE_UNDISTORT_BALANCE, (w, h)
        )
        image_new = cv2.undistort(image, K_scaled, D, None, K_new)
    return image_new, K_new, np.zeros(5, dtype=np.float64)


def load_lidar_to_camera(camera: str, lidar: str) -> np.ndarray:
    """
    Tạo transform LiDAR -> camera từ calib extrinsic.

    Với VF6_02 hiện tại, kết quả khớp thực nghiệm dùng:
        T_lidar_to_cam = T_base_to_cam @ T_lidar_to_base
    """
    extrinsics = load_json(EXTRINSIC_JSON)
    T_base_to_cam = np.asarray(extrinsics[camera], dtype=np.float64)
    T_lidar_to_base = np.asarray(extrinsics[lidar], dtype=np.float64)
    return T_base_to_cam @ T_lidar_to_base


def project_points_to_image(
    points_lidar: np.ndarray,
    T_lidar_to_cam: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Chiếu point trong hệ LiDAR tại thời điểm camera lên ảnh camera."""
    points_cam = transform_points(T_lidar_to_cam, points_lidar)
    valid = np.isfinite(points_cam).all(axis=1) & (points_cam[:, 2] > 0.1)
    points_cam = points_cam[valid]
    depths = points_cam[:, 2].copy()
    uv, _ = cv2.projectPoints(
        points_cam.reshape(-1, 1, 3).astype(np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        K.astype(np.float32),
        D.astype(np.float32),
    )
    return uv.reshape(-1, 2), depths


def draw_depth_overlay(image: np.ndarray, uv: np.ndarray, depths: np.ndarray) -> tuple[np.ndarray, int]:
    """Vẽ point lên ảnh, tô màu theo độ sâu để kiểm tra alignment bằng mắt."""
    h, w = image.shape[:2]
    valid = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    uv = uv[valid]
    depths = depths[valid]
    count = len(uv)
    if count == 0:
        return image.copy(), 0
    if len(uv) > 150_000:
        ids = np.linspace(0, len(uv) - 1, 150_000, dtype=np.int64)
        uv = uv[ids]
        depths = depths[ids]
    lo, hi = np.percentile(depths, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((depths - lo) / (hi - lo), 0.0, 1.0)
    colors = cv2.applyColorMap((255.0 * (1.0 - norm)).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
    out = image.copy()
    for (u, v), color in zip(np.round(uv).astype(np.int32), colors.reshape(-1, 3)):
        cv2.circle(out, (int(u), int(v)), 1, tuple(int(c) for c in color), -1)
    return out, count


def scenario_root(name: str) -> Path:
    """Trả về root của scenario, ví dụ data/Normal."""
    return PHENIKAA_ROOT / "data" / name


def ensure_output_dir(*parts: str) -> Path:
    """Tạo và trả về thư mục output trong outputs/benchmark."""
    out = OUTPUT_ROOT.joinpath(*parts)
    out.mkdir(parents=True, exist_ok=True)
    return out
