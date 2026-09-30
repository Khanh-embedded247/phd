#!/usr/bin/env python3
"""Tao file infos .pkl Phenikaa theo schema MapTR/nuScenes.

Muc tieu cua buoc nay:
  - Tao duoc file .pkl co cau truc giong MapTR de dataset/model doc duoc.
  - Moi sample gom 12 camera, pose lidar trong map, lidar_path va calib camera.
  - Mac dinh dung anh da undistort vi MapTR chi nhan camera pinhole K, khong co D.
  - Luon tao LiDAR da dong bo ve dung timestamp camera t_cam.
  - Chua dua GT vector map vao. Khi chua co lanelet/osm GT thi de USE_GT=False.

Ghi chu quan trong:
  - Trong .pkl MapTR, cam_intrinsic la ma tran 3x3.
  - sensor2lidar_* la transform camera -> lidar.
  - File traj_lidar.txt cua Phenikaa dang duoc hieu la T_map_lidar.
  - ego frame duoc dat la base_link thuc te cua xe.
  - `traj_lidar.txt` la T_map_lidar_top, nen can doi sang T_map_base_link.

Chay thu nhanh:
  python3 phd/phenikaa/benchmark_maptr/03_build_phenikaa_infos.py --max-samples 5

Kiem tra output:
  python3 phd/phenikaa/benchmark_maptr/02_check_infos.py \
      --pkl phd/phenikaa/outputs/benchmark_maptr/Normal/phenikaa_maptr_infos_val.pkl
"""

from __future__ import annotations

import argparse
import bisect
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import laspy
import numpy as np


PHENIKAA_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    DEFAULT_SCENARIO,
    INTRINSIC_JSON,
    EXTRINSIC_JSON,
    interpolate_pose,
    load_intrinsic,
    load_json,
    load_lidar_trajectory,
    load_lidar_to_camera,
    rot_to_quat_xyzw,
    scenario_root,
    timestamp_from_name,
    transform_points,
    undistort_for_projection,
)
from pipeline_config import load_pipeline_config  # noqa: E402


# ============================================================
# USER CONFIG
# ============================================================
# Chinh truc tiep o day neu muon, CLI args van co the override.

CONFIG_SCENARIO = DEFAULT_SCENARIO
CONFIG_SPLIT = "val"
CONFIG_REFERENCE_CAMERA = "CAM_P_F"
CONFIG_LIDAR = "LIDAR_TOP"

# Thu tu camera nen giu co dinh, vi model se nhin input theo thu tu nay.
CONFIG_CAMERAS = [
    "CAM_P_F",
    "CAM_P_FL",
    "CAM_P_FR",
    "CAM_P_B",
    "CAM_P_L",
    "CAM_P_R",
    "CAM_P_LB",
    "CAM_P_RB",
    "CAM_F_F",
    "CAM_F_L",
    "CAM_F_R",
    "CAM_F_B",
]

# Gioi han sample de test nhanh. None = toan bo frame cua reference camera.
CONFIG_MAX_SAMPLES = 200

# Neu dt anh camera khac qua nguong nay thi bo sample, tranh gom anh lech qua xa.
CONFIG_MAX_CAMERA_DT_SEC = 0.001

# Neu LiDAR nguon gan nhat lech qua nguong nay thi van ghi sample, nhung canh bao trong report.
CONFIG_WARN_LIDAR_DT_SEC = 0.05

# Luon tao file LiDAR moi da warp ve dung timestamp camera.
# Khong nen tat buoc nay vi benchmark can point cloud tai dung t_cam.
CONFIG_SYNC_METHOD = "ego_warp_lidar_to_camera_time"

# Ghi de file LAZ synced neu da ton tai.
CONFIG_OVERWRITE_SYNCED_LIDAR = False

# "undistorted": tao anh da undistort va luu K_new vao pkl.
# "raw": dung anh goc va K calib goc, chi nen dung de test schema nhanh.
CONFIG_IMAGE_MODE = "undistorted"
CONFIG_PROGRESS_EVERY = 10


def parse_args() -> argparse.Namespace:
    pipe_cfg = load_pipeline_config()
    vars_cfg = pipe_cfg["_vars"]
    build_cfg = pipe_cfg.get("build_infos", {})
    parser = argparse.ArgumentParser(description="Tao Phenikaa infos .pkl theo format MapTR.")
    parser.add_argument("--scenario", default=vars_cfg.get("SCENARIO", CONFIG_SCENARIO))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(vars_cfg["DATA_ROOT"]) if vars_cfg.get("DATA_ROOT") else None,
        help="Thu muc data scenario. Mac dinh: <PHENIKAA_ROOT>/data/<scenario>.",
    )
    parser.add_argument("--split", default=build_cfg.get("split", CONFIG_SPLIT))
    parser.add_argument("--reference-camera", default=CONFIG_REFERENCE_CAMERA)
    parser.add_argument("--lidar", default=CONFIG_LIDAR)
    parser.add_argument("--max-samples", type=int, default=int(build_cfg.get("max_samples", CONFIG_MAX_SAMPLES)))
    parser.add_argument("--start-index", type=int, default=int(build_cfg.get("start_index", 0)))
    parser.add_argument("--max-camera-dt-sec", type=float, default=float(build_cfg.get("max_camera_dt_sec", CONFIG_MAX_CAMERA_DT_SEC)))
    parser.add_argument("--warn-lidar-dt-sec", type=float, default=float(build_cfg.get("warn_lidar_dt_sec", CONFIG_WARN_LIDAR_DT_SEC)))
    parser.add_argument(
        "--progress-every",
        type=int,
        default=int(build_cfg.get("progress_every", CONFIG_PROGRESS_EVERY)),
        help="In tien trinh moi N sample hop le. 0 de tat log lap.",
    )
    parser.add_argument(
        "--overwrite-synced-lidar",
        action="store_true",
        default=bool(build_cfg.get("overwrite_synced_lidar", CONFIG_OVERWRITE_SYNCED_LIDAR)),
        help="Ghi de file LAZ synced neu da ton tai.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("undistorted", "raw"),
        default=build_cfg.get("image_mode", CONFIG_IMAGE_MODE),
        help="undistorted dung cho MapTR chuan hon; raw chi de test schema nhanh.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PHENIKAA_ROOT / "outputs" / "benchmark_maptr",
    )
    parser.add_argument(
        "--out-pkl",
        type=Path,
        default=Path(vars_cfg["INFOS"]) if vars_cfg.get("INFOS") else None,
        help="Duong dan pkl output. Mac dinh nam trong outputs/benchmark_maptr/<scenario>.",
    )
    return parser.parse_args()


def timestamp_to_us(timestamp: float) -> int:
    """MapTR/nuScenes luu timestamp dang microsecond int."""
    return int(round(float(timestamp) * 1_000_000.0))


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> list[float]:
    """pyquaternion/nuScenes dung [w, x, y, z], helper Phenikaa dang la [x, y, z, w]."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def transform_to_translation_quat_wxyz(T: np.ndarray) -> tuple[list[float], list[float]]:
    """Doi ma tran 4x4 thanh translation va quaternion [w,x,y,z]."""
    translation = [float(v) for v in T[:3, 3]]
    quat_wxyz = quat_xyzw_to_wxyz(rot_to_quat_xyzw(T[:3, :3]))
    return translation, quat_wxyz


def list_timestamped_files(directory: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """Lay danh sach file co timestamp tren ten file va sort theo timestamp."""
    if not directory.exists():
        raise FileNotFoundError(directory)
    paths = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes]
    paths.sort(key=timestamp_from_name)
    return paths


def nearest_file(timestamp: float, paths: list[Path], times: list[float]) -> tuple[Path, float]:
    """Tim file gan timestamp nhat."""
    if not paths:
        raise ValueError("Danh sach file rong")
    idx = bisect.bisect_left(times, timestamp)
    candidates: list[tuple[float, int]] = []
    if idx > 0:
        candidates.append((abs(times[idx - 1] - timestamp), idx - 1))
    if idx < len(paths):
        candidates.append((abs(times[idx] - timestamp), idx))
    dt, best_idx = min(candidates, key=lambda item: item[0])
    return paths[best_idx], float(dt)


def make_can_bus(T_map_lidar: np.ndarray) -> np.ndarray:
    """Tao can_bus 18 phan tu de MapTR khong loi shape.

    MapTR sau do se ghi de can_bus[:3] bang ego2global_translation,
    va ghi them yaw vao cac vi tri cuoi. Ta dien du lieu co ban la du.
    """
    can_bus = np.zeros(18, dtype=np.float64)
    can_bus[:3] = T_map_lidar[:3, 3]
    yaw = float(np.arctan2(T_map_lidar[1, 0], T_map_lidar[0, 0]))
    can_bus[-2] = yaw
    can_bus[-1] = np.degrees(yaw)
    return can_bus


def load_lidar_to_ego(lidar: str) -> np.ndarray:
    """Doc T_ego_lidar tu calib.

    Trong calib VF6_02, ma tran cua LIDAR_TOP dang duoc dung nhu:
        P_base = T_base_lidar @ P_lidar

    base o day chinh la ego/base_link cua xe.
    """
    extrinsics = load_json(EXTRINSIC_JSON)
    return np.asarray(extrinsics[lidar], dtype=np.float64)


def timestamp_to_name(timestamp: float) -> str:
    """Doi timestamp giay sang ten file sec-nsec on dinh."""
    sec = int(np.floor(timestamp))
    nsec = int(round((timestamp - sec) * 1_000_000_000.0))
    if nsec >= 1_000_000_000:
        sec += 1
        nsec -= 1_000_000_000
    return f"{sec}-{nsec:09d}"


def make_synced_lidar_path(output_lidar_dir: Path, timestamp: float, reference_camera: str) -> Path:
    """Ten file LAZ moi tuong ung voi timestamp camera."""
    return output_lidar_dir / f"{timestamp_to_name(timestamp)}_{reference_camera}_synced.laz"


def write_lidar_synced_to_camera_time(
    source_lidar_path: Path,
    out_lidar_path: Path,
    T_map_lidar_frame: np.ndarray,
    T_map_lidar_cam: np.ndarray,
    overwrite: bool,
) -> Path:
    """Warp LAZ tu timestamp LiDAR nguon sang he LiDAR tai timestamp camera.

    LAZ nguon da deskew trong he LiDAR tai t_lidar:
        P_lidar(t_lidar)

    Can tao point cloud trong he LiDAR tai t_cam:
        P_lidar(t_cam) =
            inv(T_map_lidar(t_cam))
            @ T_map_lidar(t_lidar)
            @ P_lidar(t_lidar)

    File moi giu nguyen cac dimension khac nhu intensity, ring, point_timestamp.
    Chi cap nhat X/Y/Z.
    """
    if out_lidar_path.exists() and not overwrite:
        return out_lidar_path

    out_lidar_path.parent.mkdir(parents=True, exist_ok=True)
    las = laspy.read(str(source_lidar_path))
    points = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    T_lidar_frame_to_lidar_cam = np.linalg.inv(T_map_lidar_cam) @ T_map_lidar_frame
    points_synced = transform_points(T_lidar_frame_to_lidar_cam, points)
    las.x = points_synced[:, 0]
    las.y = points_synced[:, 1]
    las.z = points_synced[:, 2]
    las.write(str(out_lidar_path))
    return out_lidar_path


def prepare_camera_image(
    image_path: Path,
    camera: str,
    image_mode: str,
    output_image_dir: Path,
) -> tuple[Path, np.ndarray]:
    """Tra ve duong dan anh MapTR se doc va K 3x3 tuong ung."""
    K_calib, D, calib_w, calib_h = load_intrinsic(camera)

    if image_mode == "raw":
        return image_path, K_calib.astype(np.float64)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")

    image_out, K_new, _ = undistort_for_projection(
        image,
        K_calib,
        D,
        calib_w,
        calib_h,
        is_fisheye=camera.startswith("CAM_F"),
    )
    cam_dir = output_image_dir / camera
    cam_dir.mkdir(parents=True, exist_ok=True)
    out_path = cam_dir / image_path.name
    ok = cv2.imwrite(str(out_path), image_out)
    if not ok:
        raise IOError(f"Khong ghi duoc anh undistort: {out_path}")
    return out_path, K_new.astype(np.float64)


def build_camera_entry(
    camera: str,
    lidar: str,
    image_path: Path,
    image_timestamp: float,
    T_map_ego: np.ndarray,
    T_lidar_to_ego: np.ndarray,
    image_mode: str,
    output_image_dir: Path,
) -> dict[str, Any]:
    """Tao dict cho mot camera theo schema MapTR."""
    data_path, K = prepare_camera_image(image_path, camera, image_mode, output_image_dir)

    # Transform da dung khi projection:
    #   P_cam = T_lidar_to_cam @ P_lidar
    # MapTR can sensor2lidar, tuc camera -> lidar.
    T_lidar_to_cam = load_lidar_to_camera(camera, lidar)
    T_cam_to_lidar = np.linalg.inv(T_lidar_to_cam)
    T_cam_to_ego = T_lidar_to_ego @ T_cam_to_lidar

    ego2global_translation, ego2global_rotation = transform_to_translation_quat_wxyz(T_map_ego)
    sensor2ego_translation, sensor2ego_rotation = transform_to_translation_quat_wxyz(T_cam_to_ego)

    return {
        "data_path": str(data_path),
        "type": camera,
        "sample_data_token": f"{camera}_{timestamp_to_us(image_timestamp)}",
        "timestamp": timestamp_to_us(image_timestamp),
        "cam_intrinsic": K,
        "sensor2lidar_rotation": T_cam_to_lidar[:3, :3].astype(np.float64),
        "sensor2lidar_translation": T_cam_to_lidar[:3, 3].astype(np.float64),
        "sensor2ego_rotation": sensor2ego_rotation,
        "sensor2ego_translation": sensor2ego_translation,
        "ego2global_rotation": ego2global_rotation,
        "ego2global_translation": ego2global_translation,
    }


def build_infos(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, int]]:
    """Tao toan bo infos list cho scenario."""
    start_clock = time.time()
    root = args.data_root.expanduser().resolve() if args.data_root is not None else scenario_root(args.scenario)
    camera_root = root / "CAMERA"
    lidar_dir = root / "dump" / "frames" / "laz"
    traj_path = root / "dump" / "traj_lidar.txt"

    print("[BUILD INFOS]")
    print(f"Scenario            : {args.scenario}", flush=True)
    print(f"Data root           : {root}", flush=True)
    print(f"Camera root         : {camera_root}", flush=True)
    print(f"LiDAR dir           : {lidar_dir}", flush=True)
    print(f"Trajectory          : {traj_path}", flush=True)
    print(f"Image mode          : {args.image_mode}", flush=True)
    print(f"Max samples         : {'full' if args.max_samples is None else args.max_samples}", flush=True)
    print(f"Start index         : {args.start_index}", flush=True)
    print(f"Max camera dt       : {args.max_camera_dt_sec:.6f}s", flush=True)

    cameras = CONFIG_CAMERAS
    missing_cameras = [cam for cam in cameras if not (camera_root / cam).exists()]
    if missing_cameras:
        raise FileNotFoundError(f"Thieu folder camera: {missing_cameras}")

    if args.reference_camera not in cameras:
        raise ValueError(f"reference-camera {args.reference_camera} khong nam trong CONFIG_CAMERAS")

    image_paths_by_cam: dict[str, list[Path]] = {}
    image_times_by_cam: dict[str, list[float]] = {}
    for camera in cameras:
        paths = list_timestamped_files(camera_root / camera, (".jpg", ".jpeg", ".png"))
        if not paths:
            raise ValueError(f"Khong co anh trong {camera_root / camera}")
        image_paths_by_cam[camera] = paths
        image_times_by_cam[camera] = [timestamp_from_name(p) for p in paths]
        print(f"Camera {camera:8s}      : {len(paths)} images", flush=True)

    lidar_paths = list_timestamped_files(lidar_dir, (".laz", ".las"))
    lidar_times = [timestamp_from_name(p) for p in lidar_paths]
    poses = load_lidar_trajectory(traj_path)
    T_lidar_to_ego = load_lidar_to_ego(args.lidar)
    T_ego_to_lidar = np.linalg.inv(T_lidar_to_ego)
    print(f"LiDAR frames        : {len(lidar_paths)}", flush=True)
    print(f"Trajectory poses    : {len(poses)}", flush=True)

    output_scenario_dir = args.output_root / args.scenario
    output_image_dir = output_scenario_dir / f"images_{args.image_mode}"
    output_lidar_dir = output_scenario_dir / "lidar_synced_to_camera"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_lidar_dir.mkdir(parents=True, exist_ok=True)

    ref_paths = image_paths_by_cam[args.reference_camera]
    ref_times = image_times_by_cam[args.reference_camera]
    end_index = len(ref_paths)
    valid_time_start = max(poses[0].timestamp, lidar_times[0])
    valid_time_end = min(poses[-1].timestamp, lidar_times[-1])
    print(f"Valid time start    : {valid_time_start:.9f}", flush=True)
    print(f"Valid time end      : {valid_time_end:.9f}", flush=True)
    print(f"Output image dir    : {output_image_dir}", flush=True)
    print(f"Output lidar dir    : {output_lidar_dir}", flush=True)
    print("[PROCESS]", flush=True)

    infos: list[dict[str, Any]] = []
    skipped_camera_dt = 0
    skipped_outside_time = 0
    warned_lidar_dt = 0

    for ref_index in range(args.start_index, end_index):
        if args.max_samples is not None and len(infos) >= args.max_samples:
            break

        t_sample = ref_times[ref_index]
        if t_sample < valid_time_start or t_sample > valid_time_end:
            skipped_outside_time += 1
            continue

        T_map_lidar = interpolate_pose(poses, t_sample)
        T_map_ego = T_map_lidar @ T_ego_to_lidar
        ego2global_translation, ego2global_rotation = transform_to_translation_quat_wxyz(T_map_ego)

        lidar_path, lidar_dt = nearest_file(t_sample, lidar_paths, lidar_times)
        t_lidar = timestamp_from_name(lidar_path)
        if lidar_dt > args.warn_lidar_dt_sec:
            warned_lidar_dt += 1

        cams: dict[str, dict[str, Any]] = {}
        camera_ok = True
        max_camera_dt = 0.0

        for camera in cameras:
            image_path, image_dt = nearest_file(t_sample, image_paths_by_cam[camera], image_times_by_cam[camera])
            max_camera_dt = max(max_camera_dt, image_dt)
            if image_dt > args.max_camera_dt_sec:
                camera_ok = False
                break
            cams[camera] = build_camera_entry(
                camera=camera,
                lidar=args.lidar,
                image_path=image_path,
                image_timestamp=timestamp_from_name(image_path),
                T_map_ego=T_map_ego,
                T_lidar_to_ego=T_lidar_to_ego,
                image_mode=args.image_mode,
                output_image_dir=output_image_dir,
            )

        if not camera_ok:
            skipped_camera_dt += 1
            continue

        T_map_lidar_frame = interpolate_pose(poses, t_lidar)
        output_lidar_path = make_synced_lidar_path(
            output_lidar_dir,
            t_sample,
            args.reference_camera,
        )
        write_lidar_synced_to_camera_time(
            source_lidar_path=lidar_path,
            out_lidar_path=output_lidar_path,
            T_map_lidar_frame=T_map_lidar_frame,
            T_map_lidar_cam=T_map_lidar,
            overwrite=args.overwrite_synced_lidar,
        )

        token = f"{args.scenario}_{args.split}_{len(infos):06d}_{timestamp_to_us(t_sample)}"
        info = {
            "token": token,
            "timestamp": timestamp_to_us(t_sample),
            "lidar_path": str(output_lidar_path),
            "sweeps": [],
            "lidar2ego_translation": transform_to_translation_quat_wxyz(T_lidar_to_ego)[0],
            "lidar2ego_rotation": transform_to_translation_quat_wxyz(T_lidar_to_ego)[1],
            "ego2global_translation": ego2global_translation,
            "ego2global_rotation": ego2global_rotation,
            "prev": "",
            "next": "",
            "scene_token": f"{args.scenario}_{args.split}",
            "can_bus": make_can_bus(T_map_lidar),
            "frame_idx": len(infos),
            "map_location": args.scenario,
            "cams": cams,
            # Khong co GT vector map o buoc nay.
            "annotation": {
                "divider": [],
                "ped_crossing": [],
                "boundary": [],
                "centerline": [],
            },
            # Cac field phu de debug Phenikaa, MapTR se bo qua neu khong dung.
            "phenikaa": {
                "reference_camera": args.reference_camera,
                "reference_image_path": str(ref_paths[ref_index]),
                "reference_timestamp_sec": t_sample,
                "source_lidar_path": str(lidar_path),
                "source_lidar_timestamp_sec": t_lidar,
                "synced_lidar_path": str(output_lidar_path),
                "synced_lidar_timestamp_sec": t_sample,
                "lidar_dt_sec": lidar_dt,
                "sync_method": CONFIG_SYNC_METHOD,
                "max_camera_dt_sec": max_camera_dt,
                "image_mode": args.image_mode,
                "ego_frame": "base_link",
                "source_pose_frame": "lidar_top",
                "T_lidar_to_base_link": T_lidar_to_ego.tolist(),
                "traj_path": str(traj_path),
                "intrinsic_json": str(INTRINSIC_JSON),
                "extrinsic_json": str(EXTRINSIC_JSON),
            },
        }
        infos.append(info)
        if args.progress_every > 0 and (
            len(infos) == 1
            or len(infos) % args.progress_every == 0
            or (args.max_samples is not None and len(infos) >= args.max_samples)
        ):
            elapsed = max(time.time() - start_clock, 1e-6)
            rate = len(infos) / elapsed
            target = args.max_samples if args.max_samples is not None else "full"
            print(
                f"[PROGRESS] infos={len(infos)}/{target} "
                f"ref_index={ref_index}/{end_index - 1} "
                f"t_cam={t_sample:.9f} "
                f"lidar_dt={lidar_dt * 1000.0:.1f}ms "
                f"max_cam_dt={max_camera_dt * 1000.0:.1f}ms "
                f"skipped_time={skipped_outside_time} "
                f"skipped_cam={skipped_camera_dt} "
                f"rate={rate:.2f} sample/s",
                flush=True,
            )

    for i, info in enumerate(infos):
        info["frame_idx"] = i
        info["prev"] = infos[i - 1]["token"] if i > 0 else ""
        info["next"] = infos[i + 1]["token"] if i + 1 < len(infos) else ""

    data = {
        "infos": infos,
        "metadata": {
            "version": "phenikaa-vf06-02",
            "scenario": args.scenario,
            "split": args.split,
            "num_cams": len(cameras),
            "camera_order": cameras,
            "image_mode": args.image_mode,
            "lidar_sync": CONFIG_SYNC_METHOD,
            "use_gt": False,
            "ego_frame": "base_link",
            "source_pose_frame": "lidar_top",
        },
    }
    stats = {
        "infos": len(infos),
        "skipped_camera_dt": skipped_camera_dt,
        "skipped_outside_time": skipped_outside_time,
        "warned_lidar_dt": warned_lidar_dt,
        "available_reference_frames": len(ref_paths),
        "valid_time_start": valid_time_start,
        "valid_time_end": valid_time_end,
        "elapsed_sec": time.time() - start_clock,
    }
    return data, stats


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        args.max_samples = None
    if args.out_pkl is None:
        out_dir = args.output_root / args.scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pkl = out_dir / f"phenikaa_maptr_infos_{args.split}.pkl"
    else:
        out_pkl = args.out_pkl.expanduser().resolve()
        out_pkl.parent.mkdir(parents=True, exist_ok=True)

    data, stats = build_infos(args)
    if not data["infos"]:
        raise RuntimeError("Khong tao duoc sample nao. Hay tang max-camera-dt-sec hoac kiem tra timestamp camera.")

    with out_pkl.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("[DONE]")
    print(f"Output pkl              : {out_pkl}")
    print(f"Scenario                : {args.scenario}")
    print(f"Split                   : {args.split}")
    print(f"Image mode              : {args.image_mode}")
    print(f"Camera count            : {len(CONFIG_CAMERAS)}")
    print(f"Camera order            : {CONFIG_CAMERAS}")
    print(f"Infos written           : {stats['infos']}")
    print(f"Reference frames found  : {stats['available_reference_frames']}")
    print(f"Valid time start        : {stats['valid_time_start']:.9f}")
    print(f"Valid time end          : {stats['valid_time_end']:.9f}")
    print(f"Skipped outside time    : {stats['skipped_outside_time']}")
    print(f"Skipped by camera dt    : {stats['skipped_camera_dt']}")
    print(f"LiDAR dt warnings       : {stats['warned_lidar_dt']}")
    print(f"Elapsed                 : {stats['elapsed_sec']:.1f} sec")
    print("\nKiem tra schema bang:")
    print(f"python3 {Path(__file__).with_name('02_check_infos.py')} --pkl {out_pkl}")


if __name__ == "__main__":
    main()
