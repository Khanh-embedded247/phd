#!/usr/bin/env python3
import argparse
import csv
import glob
import heapq
import json
import math
import os
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple


def _ensure_ros_pythonpath() -> None:
    ros_python = '/opt/ros/noetic/lib/python3/dist-packages'
    if os.path.isdir(ros_python) and ros_python not in sys.path:
        sys.path.insert(0, ros_python)
    os.environ.setdefault('ROS_ROOT', '/opt/ros/noetic/share/ros')
    os.environ.setdefault('ROS_PACKAGE_PATH', '/opt/ros/noetic/share')
    os.environ.setdefault('ROS_MASTER_URI', 'http://localhost:11311')
    os.environ.setdefault('ROS_VERSION', '1')
    os.environ.setdefault('ROS_DISTRO', 'noetic')
    os.environ.setdefault('ROS_PYTHON_VERSION', '3')


_ensure_ros_pythonpath()

import laspy
import numpy as np
import rosbag
import rospy
from geometry_msgs.msg import Quaternion, TransformStamped, Twist, TwistStamped, Vector3
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage

try:
    from pnkx_vehicle_msgs.msg import SteerReport, WheelAndVehicleSpeedReport
    PNKX_IMPORT_ERROR = None
except ImportError as exc:
    SteerReport = None
    WheelAndVehicleSpeedReport = None
    PNKX_IMPORT_ERROR = exc


def parse_timestamp_text(value: str) -> Optional[Tuple[int, int]]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    if '-' in text:
        parts = text.split('-')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            sec = int(parts[0])
            nsec = int(parts[1])
            return normalize_sec_nsec(sec, nsec)

    try:
        sec_float = float(text)
        sec = int(sec_float)
        nsec = int(round((sec_float - sec) * 1_000_000_000))
        return normalize_sec_nsec(sec, nsec)
    except ValueError:
        return None


def parse_timestamp_from_filename(filepath: str) -> Optional[Tuple[int, int]]:
    base = os.path.basename(filepath)
    stem, _ = os.path.splitext(base)
    return parse_timestamp_text(stem)


def normalize_sec_nsec(sec: int, nsec: int) -> Tuple[int, int]:
    total_nsec = sec * 1_000_000_000 + nsec
    if total_nsec < 0:
        return 0, 0
    new_sec = total_nsec // 1_000_000_000
    new_nsec = total_nsec % 1_000_000_000
    return int(new_sec), int(new_nsec)


def to_ns(sec: int, nsec: int) -> int:
    return sec * 1_000_000_000 + nsec


def ns_to_sec_nsec(timestamp_ns: int) -> Tuple[int, int]:
    sec = timestamp_ns // 1_000_000_000
    nsec = timestamp_ns % 1_000_000_000
    return int(sec), int(nsec)


def safe_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None:
        return default
    text = str(value).strip()
    if text == '':
        return default
    try:
        return float(text)
    except ValueError:
        return default


def euler_deg_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Quaternion:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    return Quaternion(
        w=cr * cp * cy + sr * sp * sy,
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
    )


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> Quaternion:
    trace = float(np.trace(rotation))

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        return Quaternion(w=1.0)

    return Quaternion(
        x=qx / norm,
        y=qy / norm,
        z=qz / norm,
        w=qw / norm,
    )


def load_json_file(filepath: str) -> dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def sensor_name_to_frame(sensor_name: str) -> str:
    return sensor_name.lower()


def matrix_to_transform(
    matrix: np.ndarray,
    parent_frame: str,
    child_frame: str,
    stamp: rospy.Time,
) -> TransformStamped:
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent_frame
    msg.child_frame_id = child_frame
    msg.transform.translation.x = float(matrix[0, 3])
    msg.transform.translation.y = float(matrix[1, 3])
    msg.transform.translation.z = float(matrix[2, 3])
    msg.transform.rotation = rotation_matrix_to_quaternion(matrix[:3, :3])
    return msg


def make_static_tf_message(
    extrinsic_json: str,
    sensor_names: List[str],
    base_frame: str,
    extrinsic_direction: str,
) -> TFMessage:
    extrinsics = load_json_file(extrinsic_json)
    transforms = []
    stamp = rospy.Time(0, 0)

    for sensor_name in sensor_names:
        if sensor_name not in extrinsics:
            print(f"[WARN] Sensor {sensor_name} not found in calib: {extrinsic_json}")
            continue

        matrix = np.asarray(extrinsics[sensor_name], dtype=np.float64)
        if matrix.shape != (4, 4):
            print(f"[WARN] Sensor {sensor_name} has invalid matrix shape: {matrix.shape}")
            continue

        child_frame = sensor_name_to_frame(sensor_name)
        # ROS TF parent=base, child=sensor is the pose of the sensor in base_link,
        # which is the sensor-to-base point transform (p_base = T * p_sensor).
        if extrinsic_direction == 'sensor-to-base':
            transform_sensor_in_base = matrix
        elif extrinsic_direction == 'base-to-sensor':
            transform_sensor_in_base = np.linalg.inv(matrix)
        else:
            raise ValueError(f"Unknown extrinsic direction: {extrinsic_direction}")

        transforms.append(
            matrix_to_transform(
                transform_sensor_in_base,
                base_frame,
                child_frame,
                stamp,
            )
        )

    return TFMessage(transforms=transforms)


def sorted_files_by_timestamp(folder: str, pattern: str) -> List[str]:
    files = glob.glob(os.path.join(folder, pattern))
    valid: List[Tuple[Tuple[int, int], str]] = []
    for filepath in files:
        ts = parse_timestamp_from_filename(filepath)
        if ts is not None:
            valid.append((ts, filepath))
    valid.sort(key=lambda x: (x[0][0], x[0][1]))
    return [item[1] for item in valid]


# sensor_msgs/PointField datatype enum. There is no UINT64, so LAZ
# point_timestamp (ns) is stored as FLOAT64 seconds.
POINTFIELD_INT8 = 1
POINTFIELD_UINT8 = 2
POINTFIELD_UINT16 = 4
POINTFIELD_FLOAT32 = 7
POINTFIELD_FLOAT64 = 8

# Packed PointCloud2 layout matching LAZ dimensions. Offsets are explicit so
# numpy will not insert alignment padding that disagrees with point_step.
LIDAR_POINT_SPEC = [
    ('x', np.float32, POINTFIELD_FLOAT32, 0),
    ('y', np.float32, POINTFIELD_FLOAT32, 4),
    ('z', np.float32, POINTFIELD_FLOAT32, 8),
    ('intensity', np.uint16, POINTFIELD_UINT16, 12),
    ('return_number', np.uint8, POINTFIELD_UINT8, 14),
    ('number_of_returns', np.uint8, POINTFIELD_UINT8, 15),
    ('scan_direction_flag', np.uint8, POINTFIELD_UINT8, 16),
    ('edge_of_flight_line', np.uint8, POINTFIELD_UINT8, 17),
    ('classification', np.uint8, POINTFIELD_UINT8, 18),
    ('synthetic', np.uint8, POINTFIELD_UINT8, 19),
    ('key_point', np.uint8, POINTFIELD_UINT8, 20),
    ('withheld', np.uint8, POINTFIELD_UINT8, 21),
    ('scan_angle_rank', np.int8, POINTFIELD_INT8, 22),
    ('user_data', np.uint8, POINTFIELD_UINT8, 23),
    ('point_source_id', np.uint16, POINTFIELD_UINT16, 24),
    ('ring', np.uint16, POINTFIELD_UINT16, 26),
    # LIO-SAM / Velodyne: relative seconds from scan start (header stamp).
    ('time', np.float32, POINTFIELD_FLOAT32, 28),
    ('point_timestamp', np.float64, POINTFIELD_FLOAT64, 32),
]
LIDAR_POINT_STEP = 40


def lidar_cloud_dtype() -> np.dtype:
    return np.dtype(
        {
            'names': [name for name, _, _, _ in LIDAR_POINT_SPEC],
            'formats': [dtype for _, dtype, _, _ in LIDAR_POINT_SPEC],
            'offsets': [offset for _, _, _, offset in LIDAR_POINT_SPEC],
            'itemsize': LIDAR_POINT_STEP,
        }
    )


def lidar_point_fields() -> List[PointField]:
    return [
        PointField(name=name, offset=offset, datatype=pf_type, count=1)
        for name, _, pf_type, offset in LIDAR_POINT_SPEC
    ]


def las_dimension_array(las, name: str, dtype, count: int) -> np.ndarray:
    dimension_names = set(las.point_format.dimension_names)
    if name not in dimension_names:
        return np.zeros(count, dtype=dtype)
    return np.asarray(getattr(las, name), dtype=dtype)


def build_lidar_cloud(las) -> Tuple[bytes, int, Optional[Tuple[int, int]]]:
    count = int(las.header.point_count)
    points = np.zeros(count, dtype=lidar_cloud_dtype())
    header_ts: Optional[Tuple[int, int]] = None

    points['x'] = np.asarray(las.x, dtype=np.float32)
    points['y'] = np.asarray(las.y, dtype=np.float32)
    points['z'] = np.asarray(las.z, dtype=np.float32)
    points['intensity'] = las_dimension_array(las, 'intensity', np.uint16, count)
    points['return_number'] = las_dimension_array(las, 'return_number', np.uint8, count)
    points['number_of_returns'] = las_dimension_array(las, 'number_of_returns', np.uint8, count)
    points['scan_direction_flag'] = las_dimension_array(las, 'scan_direction_flag', np.uint8, count)
    points['edge_of_flight_line'] = las_dimension_array(las, 'edge_of_flight_line', np.uint8, count)
    points['classification'] = las_dimension_array(las, 'classification', np.uint8, count)
    points['synthetic'] = las_dimension_array(las, 'synthetic', np.uint8, count)
    points['key_point'] = las_dimension_array(las, 'key_point', np.uint8, count)
    points['withheld'] = las_dimension_array(las, 'withheld', np.uint8, count)
    points['scan_angle_rank'] = las_dimension_array(las, 'scan_angle_rank', np.int8, count)
    points['user_data'] = las_dimension_array(las, 'user_data', np.uint8, count)
    points['point_source_id'] = las_dimension_array(las, 'point_source_id', np.uint16, count)

    ring_name = None
    for candidate in ('ring', 'scan_id', 'laser_id', 'channel'):
        if hasattr(las, candidate):
            ring_name = candidate
            break
    if ring_name is not None:
        points['ring'] = np.asarray(getattr(las, ring_name), dtype=np.uint16)

    if hasattr(las, 'point_timestamp'):
        raw_time_ns = np.asarray(las.point_timestamp, dtype=np.uint64)
        points['point_timestamp'] = (raw_time_ns / np.float64(1e9)).astype(np.float64)
        points.sort(order='point_timestamp')
        t0 = float(points['point_timestamp'][0]) if count else 0.0
        points['time'] = (points['point_timestamp'] - t0).astype(np.float32)
        sec = int(t0)
        nsec = int(round((t0 - sec) * 1_000_000_000))
        header_ts = normalize_sec_nsec(sec, nsec)

    return points.tobytes(), count, header_ts


def iter_lidar_messages(
    lidar_folder: str,
    topic: str,
    frame_id: str,
    max_frames: Optional[int] = None,
)-> Iterator[Tuple[int, str, object]]:
    files = sorted_files_by_timestamp(lidar_folder, '*.laz')
    if max_frames is not None:
        files = files[: max(0, max_frames)]
    print(f'[INFO] Found {len(files)} LAZ files in {lidar_folder}', flush=True)
    if not files:
        return

    from laspy import LazBackend
    if not LazBackend.detect_available():
        raise RuntimeError(
            'No LAZ backend available (install lazrs in this Python env). '
            'Without it every .laz file is skipped and /lidar_top will be empty.'
        )

    fields = lidar_point_fields()
    seq = 0
    for filepath in files:
        parsed = parse_timestamp_from_filename(filepath)
        if parsed is None:
            continue
        sec, nsec = parsed

        try:
            las = laspy.read(filepath)
            cloud_data, width, point_header_ts = build_lidar_cloud(las)
        except Exception as exc:
            print(f"[WARN] Cannot read LAZ file {filepath}: {exc}")
            continue

        if point_header_ts is not None:
            sec, nsec = point_header_ts
        header = Header(seq=seq, stamp=rospy.Time(sec, nsec), frame_id=frame_id)
        seq += 1

        msg = PointCloud2(
            header=header,
            height=1,
            width=width,
            fields=fields,
            is_bigendian=False,
            point_step=LIDAR_POINT_STEP,
            row_step=LIDAR_POINT_STEP * width,
            data=cloud_data,
            is_dense=True,
        )

        yield to_ns(sec, nsec), topic, msg


def iter_imu_messages(
    imu_folder: str,
    topic: str,
    frame_id: str,
)-> Iterator[Tuple[int, str, object]]:
    files = sorted_files_by_timestamp(imu_folder, '*.csv')
    print(f'[INFO] Found {len(files)} IMU CSV files in {imu_folder}', flush=True)
    if not files:
        return

    seq = 0
    for filepath in files:
        file_ts = parse_timestamp_from_filename(filepath)
        if file_ts is None:
            file_ts = (0, 0)

        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_row = parse_timestamp_text(row.get('Timestamp', ''))
                sec, nsec = ts_row if ts_row is not None else file_ts

                header = Header(seq=seq, stamp=rospy.Time(sec, nsec), frame_id=frame_id)
                seq += 1

                msg = Imu(
                    header=header,
                    orientation=Quaternion(
                        x=safe_float(row, 'qx'),
                        y=safe_float(row, 'qy'),
                        z=safe_float(row, 'qz'),
                        w=safe_float(row, 'qw', 1.0),
                    ),
                    orientation_covariance=[0.0] * 9,
                    angular_velocity=Vector3(
                        x=safe_float(row, 'vx'),
                        y=safe_float(row, 'vy'),
                        z=safe_float(row, 'vz'),
                    ),
                    angular_velocity_covariance=[0.0] * 9,
                    linear_acceleration=Vector3(
                        x=safe_float(row, 'ax'),
                        y=safe_float(row, 'ay'),
                        z=safe_float(row, 'az'),
                    ),
                    linear_acceleration_covariance=[0.0] * 9,
                )

                yield to_ns(sec, nsec), topic, msg


def iter_gnss_messages(
    nav_folder: str,
    topic: str,
    frame_id: str,
)-> Iterator[Tuple[int, str, object]]:
    files = sorted_files_by_timestamp(nav_folder, '*.csv')
    if not files:
        return

    seq = 0
    for filepath in files:
        file_ts = parse_timestamp_from_filename(filepath)
        if file_ts is None:
            file_ts = (0, 0)

        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_row = parse_timestamp_text(row.get('Timestamp', ''))
                sec, nsec = ts_row if ts_row is not None else file_ts

                lat = safe_float(row, 'Latitude', float('nan'))
                lon = safe_float(row, 'Longitude', float('nan'))
                alt = safe_float(row, 'Altitude', float('nan'))

                lat_std = safe_float(row, 'Latitude_std', float('nan'))
                lon_std = safe_float(row, 'Longitude_std', float('nan'))
                alt_std = safe_float(row, 'Altitude_std', float('nan'))

                has_std = not (np.isnan(lat_std) or np.isnan(lon_std) or np.isnan(alt_std))
                if has_std:
                    covariance = [
                        lat_std * lat_std,
                        0.0,
                        0.0,
                        0.0,
                        lon_std * lon_std,
                        0.0,
                        0.0,
                        0.0,
                        alt_std * alt_std,
                    ]
                    covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
                else:
                    covariance = [0.0] * 9
                    covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

                has_fix = not (np.isnan(lat) or np.isnan(lon))
                status_value = NavSatStatus.STATUS_FIX if has_fix else NavSatStatus.STATUS_NO_FIX

                header = Header(seq=seq, stamp=rospy.Time(sec, nsec), frame_id=frame_id)
                seq += 1

                msg = NavSatFix(
                    header=header,
                    status=NavSatStatus(status=status_value, service=NavSatStatus.SERVICE_GPS),
                    latitude=lat,
                    longitude=lon,
                    altitude=alt,
                    position_covariance=covariance,
                    position_covariance_type=covariance_type,
                )

                yield to_ns(sec, nsec), topic, msg


def iter_gps_imu_messages(
    nav_folder: str,
    topic: str,
    frame_id: str,
) -> Iterator[Tuple[int, str, object]]:
    files = sorted_files_by_timestamp(nav_folder, '*.csv')
    if not files:
        return

    seq = 0
    for filepath in files:
        file_ts = parse_timestamp_from_filename(filepath)
        if file_ts is None:
            file_ts = (0, 0)

        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_row = parse_timestamp_text(row.get('Timestamp', ''))
                sec, nsec = ts_row if ts_row is not None else file_ts

                roll = safe_float(row, 'Roll')
                pitch = safe_float(row, 'Pitch')
                yaw = safe_float(row, 'Heading2')
                orientation = euler_deg_to_quaternion(roll, pitch, yaw)

                header = Header(seq=seq, stamp=rospy.Time(sec, nsec), frame_id=frame_id)
                seq += 1

                msg = Imu(
                    header=header,
                    orientation=orientation,
                    orientation_covariance=[0.0] * 9,
                    angular_velocity=Vector3(
                        x=safe_float(row, 'GyroX'),
                        y=safe_float(row, 'GyroY'),
                        z=safe_float(row, 'GyroZ'),
                    ),
                    angular_velocity_covariance=[0.0] * 9,
                    linear_acceleration=Vector3(
                        x=safe_float(row, 'AccX'),
                        y=safe_float(row, 'AccY'),
                        z=safe_float(row, 'AccZ'),
                    ),
                    linear_acceleration_covariance=[0.0] * 9,
                )

                yield to_ns(sec, nsec), topic, msg


def iter_wheel_speed_messages(
    vehicle_info_folder: str,
    topic: str,
    frame_id: str,
) -> Iterator[Tuple[int, str, object]]:
    files = sorted_files_by_timestamp(vehicle_info_folder, '*.csv')
    if not files:
        return

    seq = 0
    for filepath in files:
        file_ts = parse_timestamp_from_filename(filepath)
        if file_ts is None:
            file_ts = (0, 0)

        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_row = parse_timestamp_text(row.get('Timestamp', ''))
                sec, nsec = ts_row if ts_row is not None else file_ts

                header = Header(seq=seq, stamp=rospy.Time(sec, nsec), frame_id=frame_id)
                seq += 1

                msg = WheelAndVehicleSpeedReport(
                    header=header,
                    wheel_speed_rr=safe_float(row, 'wheel_rear_right_speed'),
                    wheel_speed_rl=safe_float(row, 'wheel_rear_left_speed'),
                    wheel_speed_fr=safe_float(row, 'wheel_front_right_speed'),
                    wheel_speed_fl=safe_float(row, 'wheel_front_left_speed'),
                    vehicle_speed=safe_float(row, 'vehicle_speed'),
                )

                yield to_ns(sec, nsec), topic, msg


def iter_twist_messages(
    vehicle_info_folder: str,
    topic: str,
    frame_id: str,
) -> Iterator[Tuple[int, str, object]]:
    files = sorted_files_by_timestamp(vehicle_info_folder, '*.csv')
    if not files:
        return

    seq = 0
    for filepath in files:
        file_ts = parse_timestamp_from_filename(filepath)
        if file_ts is None:
            file_ts = (0, 0)

        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_row = parse_timestamp_text(row.get('Timestamp', ''))
                sec, nsec = ts_row if ts_row is not None else file_ts

                vehicle_speed = safe_float(row, 'vehicle_speed') / 3.6

                header = Header(seq=seq, stamp=rospy.Time(sec, nsec), frame_id=frame_id)
                seq += 1

                msg = TwistStamped(
                    header=header,
                    twist=Twist(
                        linear=Vector3(x=vehicle_speed, y=0.0, z=0.0),
                        angular=Vector3(x=0.0, y=0.0, z=0.0),
                    ),
                )

                yield to_ns(sec, nsec), topic, msg


def iter_gnss_twist_messages(
    nav_folder: str,
    topic: str,
    frame_id: str,
) -> Iterator[Tuple[int, str, object]]:
    files = sorted_files_by_timestamp(nav_folder, '*.csv')
    if not files:
        return

    seq = 0
    for filepath in files:
        file_ts = parse_timestamp_from_filename(filepath)
        if file_ts is None:
            file_ts = (0, 0)

        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_row = parse_timestamp_text(row.get('Timestamp', ''))
                sec, nsec = ts_row if ts_row is not None else file_ts

                ve = safe_float(row, 'Ve')
                vn = safe_float(row, 'Vn')
                vu = safe_float(row, 'Vu')

                header = Header(seq=seq, stamp=rospy.Time(sec, nsec), frame_id=frame_id)
                seq += 1

                msg = TwistStamped(
                    header=header,
                    twist=Twist(
                        linear=Vector3(x=ve, y=vn, z=vu),
                        angular=Vector3(x=0.0, y=0.0, z=0.0),
                    ),
                )

                yield to_ns(sec, nsec), topic, msg


def iter_steer_messages(
    vehicle_steer_folder: str,
    topic: str,
    frame_id: str,
) -> Iterator[Tuple[int, str, object]]:
    files = sorted_files_by_timestamp(vehicle_steer_folder, '*.csv')
    if not files:
        return

    seq = 0
    for filepath in files:
        file_ts = parse_timestamp_from_filename(filepath)
        if file_ts is None:
            file_ts = (0, 0)

        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_row = parse_timestamp_text(row.get('Timestamp', ''))
                sec, nsec = ts_row if ts_row is not None else file_ts

                header = Header(seq=seq, stamp=rospy.Time(sec, nsec), frame_id=frame_id)
                seq += 1

                msg = SteerReport(
                    header=header,
                    steer_angle=safe_float(row, 'steer_angle'),
                    steer_speed=safe_float(row, 'steer_speed'),
                )

                yield to_ns(sec, nsec), topic, msg


def merge_streams(
    streams: List[Iterator[Tuple[int, str, object]]],
) -> Iterator[Tuple[int, str, object]]:
    heap: List[Tuple[int, int, str, object]] = []

    for i, stream in enumerate(streams):
        try:
            ts, topic, msg = next(stream)
            heapq.heappush(heap, (ts, i, topic, msg))
        except StopIteration:
            pass

    while heap:
        ts, i, topic, msg = heapq.heappop(heap)
        yield ts, topic, msg

        try:
            next_ts, next_topic, next_msg = next(streams[i])
            heapq.heappush(heap, (next_ts, i, next_topic, next_msg))
        except StopIteration:
            pass


def convert_to_bag(
    lidar_folder: str,
    imu_folder: str,
    nav_folder: str,
    vehicle_info_folder: str,
    vehicle_steer_folder: str,
    output_bag: str,
    lidar_topic: str,
    imu_topic: str,
    gnss_topic: str,
    gps_imu_topic: str,
    wheel_speed_topic: str,
    steer_topic: str,
    twist_topic: str,
    gnss_twist_topic: str,
    lidar_frame: str,
    imu_frame: str,
    gnss_frame: str,
    gps_imu_frame: str,
    wheel_speed_frame: str,
    steer_frame: str,
    twist_frame: str,
    gnss_twist_frame: str,
    max_lidar_frames: Optional[int] = None,
) -> None:
    lidar_iter = iter_lidar_messages(lidar_folder, lidar_topic, lidar_frame, max_lidar_frames)
    imu_iter = iter_imu_messages(imu_folder, imu_topic, imu_frame)
    gnss_iter = iter_gnss_messages(nav_folder, gnss_topic, gnss_frame)
    gps_imu_iter = iter_gps_imu_messages(nav_folder, gps_imu_topic, gps_imu_frame)
    wheel_speed_iter = iter_wheel_speed_messages(vehicle_info_folder, wheel_speed_topic, wheel_speed_frame)
    steer_iter = iter_steer_messages(vehicle_steer_folder, steer_topic, steer_frame)
    twist_iter = iter_twist_messages(vehicle_info_folder, twist_topic, twist_frame)
    gnss_twist_iter = iter_gnss_twist_messages(nav_folder, gnss_twist_topic, gnss_twist_frame)

    streams = [
        iter(lidar_iter),
        iter(imu_iter),
        iter(gnss_iter),
        iter(gps_imu_iter),
        iter(wheel_speed_iter),
        iter(steer_iter),
        iter(twist_iter),
        iter(gnss_twist_iter),
    ]

    message_counts = {
        lidar_topic: 0,
        imu_topic: 0,
        gnss_topic: 0,
        gps_imu_topic: 0,
        wheel_speed_topic: 0,
        steer_topic: 0,
        twist_topic: 0,
        gnss_twist_topic: 0,
    }

    total_count = 0
    progress_every = 5000
    start_time = time.time()

    with rosbag.Bag(output_bag, 'w') as bag:
        for ts_ns, topic, msg in merge_streams(streams):
            sec, nsec = ns_to_sec_nsec(ts_ns)
            bag.write(topic, msg, t=rospy.Time(sec, nsec))
            message_counts[topic] += 1
            total_count += 1

            if total_count % progress_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"Progress: {total_count} messages written "
                    f"(lidar={message_counts[lidar_topic]}, imu={message_counts[imu_topic]}, "
                    f"gnss={message_counts[gnss_topic]}, gps_imu={message_counts[gps_imu_topic]}, "
                    f"wheel={message_counts[wheel_speed_topic]}, steer={message_counts[steer_topic]}, "
                    f"twist={message_counts[twist_topic]}, gnss_twist={message_counts[gnss_twist_topic]}, "
                    f"elapsed={elapsed:.1f}s)"
                )

    print('Conversion completed.')
    for topic, count in message_counts.items():
        print(f'  {topic}: {count} messages')
    print(f'Output bag: {output_bag}')


def convert_normal_lidar_imu_to_bag(
    lidar_folder: str,
    imu_folder: str,
    output_bag: str,
    extrinsic_json: str,
    lidar_sensor: str,
    imu_sensor: str,
    lidar_topic: str,
    imu_topic: str,
    tf_static_topic: str,
    base_frame: str,
    lidar_frame: str,
    imu_frame: str,
    extrinsic_direction: str,
    max_lidar_frames: Optional[int] = None,
) -> None:
    lidar_iter = iter_lidar_messages(lidar_folder, lidar_topic, lidar_frame, max_lidar_frames)
    imu_iter = iter_imu_messages(imu_folder, imu_topic, imu_frame)
    streams = [iter(lidar_iter), iter(imu_iter)]

    message_counts = {
        tf_static_topic: 0,
        lidar_topic: 0,
        imu_topic: 0,
    }

    static_tf = make_static_tf_message(
        extrinsic_json,
        [lidar_sensor, imu_sensor],
        base_frame,
        extrinsic_direction,
    )
    for transform in static_tf.transforms:
        t = transform.transform.translation
        print(
            f'[INFO] TF {transform.header.frame_id} -> {transform.child_frame_id}: '
            f't=({t.x:.4f}, {t.y:.4f}, {t.z:.4f})',
            flush=True,
        )

    total_count = 0
    progress_every = 1000
    start_time = time.time()

    with rosbag.Bag(output_bag, 'w') as bag:
        if static_tf.transforms:
            bag.write(tf_static_topic, static_tf, t=rospy.Time(0, 0))
            message_counts[tf_static_topic] += 1
            total_count += 1

        for ts_ns, topic, msg in merge_streams(streams):
            sec, nsec = ns_to_sec_nsec(ts_ns)
            bag.write(topic, msg, t=rospy.Time(sec, nsec))
            message_counts[topic] += 1
            total_count += 1

            if total_count % progress_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"Progress: {total_count} messages written "
                    f"(tf_static={message_counts[tf_static_topic]}, "
                    f"lidar={message_counts[lidar_topic]}, "
                    f"imu={message_counts[imu_topic]}, "
                    f"elapsed={elapsed:.1f}s)"
                )

    print('Conversion completed.')
    for topic, count in message_counts.items():
        print(f'  {topic}: {count} messages')
    print(f'Output bag: {output_bag}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert LAZ + IMU/NAV/vehicle CSV data to a ROS1 bag with PointCloud2, Imu, NavSatFix, WheelAndVehicleSpeedReport, and SteerReport.',
    )
    default_project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
    default_normal_root = os.path.join(default_project_root, 'data', 'Normal')
    default_calib_root = os.path.join(default_project_root, 'calib', 'vf_06_02')

    parser.add_argument('--mode', choices=('normal', 'full'), default='normal', help='normal writes LiDAR_TOP + IMU + tf_static only')
    parser.add_argument('--normal-root', default=default_normal_root, help='Root folder of the Normal dataset')
    parser.add_argument('--extrinsic-json', default=os.path.join(default_calib_root, 'VF6_02_Extrinsics_By_Dates.json'), help='Sensor extrinsic JSON')
    parser.add_argument('--extrinsic-direction', choices=('sensor-to-base', 'base-to-sensor'), default='sensor-to-base', help='Direction of matrices stored in the extrinsic JSON')
    parser.add_argument('--base-frame', default='base_link', help='Base frame used for static TF')
    parser.add_argument('--lidar-sensor', default='LIDAR_TOP', help='LiDAR sensor name in Normal/LIDAR and calib JSON')
    parser.add_argument('--imu-sensor', default='IMU', help='IMU sensor name in calib JSON')

    parser.add_argument('--lidar-folder', default=None, help='Folder containing .laz files')
    parser.add_argument('--imu-folder', default=None, help='Folder containing IMU .csv files')
    parser.add_argument('--nav-folder', default=None, help='Folder containing NAV .csv files')
    parser.add_argument('--vehicle-info-folder', default='OTHERS/VEHICLE_INFO', help='Folder containing vehicle info .csv files')
    parser.add_argument('--vehicle-steer-folder', default='OTHERS/VEHICLE_STEER', help='Folder containing vehicle steer .csv files')
    parser.add_argument('--output-bag', default=os.path.join(default_project_root, 'outputs', 'normal_lidar_imu.bag'), help='Output .bag path')

    parser.add_argument('--lidar-topic', default='/lidar_top', help='PointCloud2 topic name')
    parser.add_argument('--imu-topic', default='/sensing/imu/imu_data', help='Imu topic name')
    parser.add_argument('--tf-static-topic', default='/tf_static', help='Static TF topic name')
    parser.add_argument('--gnss-topic', default='/sensing/gnss/nav_sat_fix', help='NavSatFix topic name')
    parser.add_argument('--gps-imu-topic', default='/sensing/gnss/imu_data', help='GPS-derived Imu topic name')
    parser.add_argument('--wheel-speed-topic', default='/vehicle/wheel_and_vehicle_speed_report', help='WheelAndVehicleSpeedReport topic name')
    parser.add_argument('--steer-topic', default='/vehicle/steer_report', help='SteerReport topic name')
    parser.add_argument('--twist-topic', default='/vehicle/twist', help='TwistStamped topic name')
    parser.add_argument('--gnss-twist-topic', default='/sensing/gnss/twist', help='GNSS TwistStamped topic name')

    parser.add_argument('--max-lidar-frames', type=int, default=None, help='Limit number of LAZ frames written (debug/smoke test)')
    parser.add_argument('--lidar-frame', default='lidar_link', help='PointCloud2 frame_id')
    parser.add_argument('--imu-frame', default='imu_link', help='Imu frame_id')
    parser.add_argument('--gnss-frame', default='gps_link', help='NavSatFix frame_id')
    parser.add_argument('--gps-imu-frame', default='gps_link', help='GPS-derived Imu frame_id')
    parser.add_argument('--wheel-speed-frame', default='base_link', help='WheelAndVehicleSpeedReport frame_id')
    parser.add_argument('--steer-frame', default='base_link', help='SteerReport frame_id')
    parser.add_argument('--twist-frame', default='base_link', help='TwistStamped frame_id')
    parser.add_argument('--gnss-twist-frame', default='base_link', help='GNSS TwistStamped frame_id')

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.lidar_folder is None:
        args.lidar_folder = os.path.join(args.normal_root, 'LIDAR', args.lidar_sensor)
    if args.imu_folder is None:
        args.imu_folder = os.path.join(args.normal_root, 'OTHERS', 'IMU')
    if args.nav_folder is None:
        args.nav_folder = os.path.join(args.normal_root, 'OTHERS', 'NAV')

    output_dir = os.path.dirname(os.path.abspath(args.output_bag))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.mode == 'normal':
        if not os.path.isdir(args.lidar_folder):
            print(f'[ERROR] Lidar folder not found: {args.lidar_folder}')
            return 1
        if not os.path.isdir(args.imu_folder):
            print(f'[ERROR] IMU folder not found: {args.imu_folder}')
            return 1
        if not os.path.isfile(args.extrinsic_json):
            print(f'[ERROR] Extrinsic JSON not found: {args.extrinsic_json}')
            return 1

        convert_normal_lidar_imu_to_bag(
            lidar_folder=args.lidar_folder,
            imu_folder=args.imu_folder,
            output_bag=args.output_bag,
            extrinsic_json=args.extrinsic_json,
            lidar_sensor=args.lidar_sensor,
            imu_sensor=args.imu_sensor,
            lidar_topic=args.lidar_topic,
            imu_topic=args.imu_topic,
            tf_static_topic=args.tf_static_topic,
            base_frame=args.base_frame,
            lidar_frame=args.lidar_frame,
            imu_frame=args.imu_frame,
            extrinsic_direction=args.extrinsic_direction,
            max_lidar_frames=args.max_lidar_frames,
        )
        return 0

    if PNKX_IMPORT_ERROR is not None:
        print('[ERROR] Cannot import pnkx_vehicle_msgs. Make sure the ROS workspace is sourced in the container.')
        print(f'        Import error: {PNKX_IMPORT_ERROR}')
        return 1

    if not os.path.isdir(args.lidar_folder):
        print(f'[ERROR] Lidar folder not found: {args.lidar_folder}')
        return 1
    if not os.path.isdir(args.imu_folder):
        print(f'[ERROR] IMU folder not found: {args.imu_folder}')
        return 1
    if not os.path.isdir(args.nav_folder):
        print(f'[ERROR] NAV folder not found: {args.nav_folder}')
        return 1
    if not os.path.isdir(args.vehicle_info_folder):
        print(f'[ERROR] Vehicle info folder not found: {args.vehicle_info_folder}')
        return 1
    if not os.path.isdir(args.vehicle_steer_folder):
        print(f'[ERROR] Vehicle steer folder not found: {args.vehicle_steer_folder}')
        return 1

    convert_to_bag(
        lidar_folder=args.lidar_folder,
        imu_folder=args.imu_folder,
        nav_folder=args.nav_folder,
        vehicle_info_folder=args.vehicle_info_folder,
        vehicle_steer_folder=args.vehicle_steer_folder,
        output_bag=args.output_bag,
        lidar_topic=args.lidar_topic,
        imu_topic=args.imu_topic,
        gnss_topic=args.gnss_topic,
        gps_imu_topic=args.gps_imu_topic,
        wheel_speed_topic=args.wheel_speed_topic,
        steer_topic=args.steer_topic,
        twist_topic=args.twist_topic,
        gnss_twist_topic=args.gnss_twist_topic,
        lidar_frame=args.lidar_frame,
        imu_frame=args.imu_frame,
        gnss_frame=args.gnss_frame,
        gps_imu_frame=args.gps_imu_frame,
        wheel_speed_frame=args.wheel_speed_frame,
        steer_frame=args.steer_frame,
        twist_frame=args.twist_frame,
        gnss_twist_frame=args.gnss_twist_frame,
        max_lidar_frames=args.max_lidar_frames,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
