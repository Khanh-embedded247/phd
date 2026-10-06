"""
B3 Step 1 - forward MapTR on Phenikaa frames and write MapTR-style results.

Output:
  outputs/phenikaa_vis/phenikaa_b3/maptr_results/nuscmap_results.json

The JSON is then consumed by infer_phenikaa.py to save per-frame local vectors.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from phenikaa_hdmap.paths import EXTRINSIC_JSON, INTRINSIC_JSON, PHENIKAA_DATA, PROJECT_ROOT, THIRD_PARTY_MAPTR

MAPTR_ROOT = THIRD_PARTY_MAPTR
if str(MAPTR_ROOT) not in sys.path:
    sys.path.insert(0, str(MAPTR_ROOT))

MAPTR_CLASSES = {0: "divider", 1: "ped_crossing", 2: "boundary"}
MAPTRV2_CONFIG = MAPTR_ROOT / "projects/configs/maptrv2/maptrv2_nusc_r50_24ep_eval_8g.py"
MAPTRV2_CHECKPOINT = MAPTR_ROOT / "ckpts/maptrv2_nusc_r50_24ep.pth"
MAPTR_FUSION_CONFIG = MAPTR_ROOT / "projects/configs/maptr/maptr_tiny_fusion_eval_8g.py"
MAPTR_FUSION_CHECKPOINT = MAPTR_ROOT / "ckpts/maptr_tiny_fusion_24e.pth"
IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
TARGET_W = 832
TARGET_H = 480
PHENIKAA6_CAMERAS = ("CAM_P_F", "CAM_P_FL", "CAM_P_FR", "CAM_P_B", "CAM_P_L", "CAM_P_R")
PHENIKAA10_CAMERAS = (
    "CAM_P_L",
    "CAM_P_FL",
    "CAM_P_F",
    "CAM_P_FR",
    "CAM_P_R",
    "CAM_P_B",
    "CAM_F_L",
    "CAM_F_F",
    "CAM_F_R",
    "CAM_F_B",
)

BASELINE_DEFAULTS = {
    "b1_camera": {
        "config": MAPTRV2_CONFIG,
        "checkpoint": MAPTRV2_CHECKPOINT,
        "camera_set": "all10",
    },
    "b2_fusion": {
        "config": MAPTR_FUSION_CONFIG,
        "checkpoint": MAPTR_FUSION_CHECKPOINT,
        "camera_set": "phenikaa6",
    },
}


def _load_infos(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "infos" in payload:
        return list(payload["infos"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported infos format: {path}")


def _load_calib() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        json.loads(INTRINSIC_JSON.read_text(encoding="utf-8")),
        json.loads(EXTRINSIC_JSON.read_text(encoding="utf-8")),
    )


def _load_lidar_points(path: str, device: Any) -> Any:
    import torch

    if not path:
        return torch.zeros((0, 5), dtype=torch.float32, device=device)

    lidar_path = Path(path)
    candidate_paths = [lidar_path]
    if lidar_path.suffix.lower() == ".laz":
        candidate_paths.extend(
            [
                lidar_path.with_suffix(".bin"),
                lidar_path.parent.parent / "LidarBin" / f"{lidar_path.stem}.bin",
                lidar_path.parent.parent / "Lidar_bin" / f"{lidar_path.stem}.bin",
            ]
        )

    for candidate in candidate_paths[1:]:
        if candidate.is_file():
            lidar_path = candidate
            break

    if not lidar_path.is_file():
        return torch.zeros((0, 5), dtype=torch.float32, device=device)

    if lidar_path.suffix.lower() == ".bin":
        arr = np.fromfile(str(lidar_path), dtype=np.float32)
        if arr.size % 5 == 0:
            points = arr.reshape(-1, 5)
        elif arr.size % 4 == 0:
            points4 = arr.reshape(-1, 4)
            points = np.zeros((points4.shape[0], 5), dtype=np.float32)
            points[:, :4] = points4
        else:
            points = np.zeros((0, 5), dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(points.astype(np.float32))).to(device).contiguous()

    try:
        import laspy

        las = laspy.read(str(lidar_path))
        xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
        intensity = (
            np.asarray(las.intensity, dtype=np.float32).reshape(-1, 1)
            if "intensity" in las.point_format.dimension_names
            else np.zeros((xyz.shape[0], 1), dtype=np.float32)
        )
        ring = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        points = np.concatenate((xyz, intensity, ring), axis=1)
    except Exception as exc:
        print(
            f"  WARNING: could not load LiDAR points from {lidar_path}: {exc}\n"
            f"           Install lazrs/laszip or convert LAZ to BIN beside the .laz file."
        )
        points = np.zeros((0, 5), dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(points.astype(np.float32))).to(device).contiguous()


def _pad_to_divisor(img: np.ndarray, divisor: int = 32) -> np.ndarray:
    h, w = img.shape[:2]
    pad_h = int(np.ceil(h / divisor) * divisor)
    pad_w = int(np.ceil(w / divisor) * divisor)
    padded = np.zeros((pad_h, pad_w, img.shape[2]), dtype=img.dtype)
    padded[:h, :w] = img
    return padded


def _camera_K_D(intrinsics: dict[str, Any], cam_name: str) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(intrinsics[cam_name]["camera_matrix"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(intrinsics[cam_name].get("distortion_coefficients", []), dtype=np.float64).reshape(-1)
    if cam_name.startswith("CAM_F") and D.size > 4:
        D = D[:4]
    return K, D


def _undistort_image_and_K(
    img_rgb: np.ndarray,
    intrinsics: dict[str, Any],
    cam_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    K, D = _camera_K_D(intrinsics, cam_name)
    if D.size == 0:
        return img_rgb, K

    h, w = img_rgb.shape[:2]
    if cam_name.startswith("CAM_F"):
        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            D.astype(np.float64),
            (w, h),
            np.eye(3),
            balance=0.3,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K,
            D.astype(np.float64),
            np.eye(3),
            K_new,
            (w, h),
            cv2.CV_16SC2,
        )
        img_rgb = cv2.remap(img_rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return img_rgb, K_new

    K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.0)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), K_new, (w, h), cv2.CV_16SC2)
    img_rgb = cv2.remap(img_rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return img_rgb, K_new


def _load_and_preprocess_image(
    path: str,
    intrinsics: dict[str, Any],
    cam_name: str,
    undistort: bool,
) -> tuple[np.ndarray, tuple[int, int, int], tuple[int, int, int], np.ndarray, float, float]:
    import mmcv

    try:
        img_rgb = mmcv.imread(path, channel_order="rgb")
    except Exception:
        img_rgb = None
    if img_rgb is None:
        src_h, src_w = TARGET_H, TARGET_W
        img_rgb = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.float32)
        K_used = np.eye(3, dtype=np.float64)
    else:
        src_h, src_w = img_rgb.shape[:2]
        img_rgb = img_rgb.astype(np.float32)
        K_used, _ = _camera_K_D(intrinsics, cam_name)
        if undistort:
            img_rgb, K_used = _undistort_image_and_K(img_rgb, intrinsics, cam_name)
            img_rgb = img_rgb.astype(np.float32)
        img_rgb = mmcv.imresize(img_rgb, (TARGET_W, TARGET_H)).astype(np.float32)
    img_shape = img_rgb.shape
    img_rgb = (img_rgb - IMG_MEAN) / IMG_STD
    img_rgb = _pad_to_divisor(img_rgb, 32)
    pad_shape = img_rgb.shape
    sx = TARGET_W / float(src_w)
    sy = TARGET_H / float(src_h)
    return img_rgb.transpose(2, 0, 1), img_shape, pad_shape, K_used, sx, sy


def _camera_intrinsic_4x4(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    K = np.asarray(K, dtype=np.float32).reshape(3, 3).copy()
    K[0, :] *= sx
    K[1, :] *= sy
    K4 = np.eye(4, dtype=np.float32)
    K4[:3, :3] = K
    return K4


def _lidar_to_camera_4x4(extrinsics: dict[str, Any], cam_name: str) -> np.ndarray:
    mat = np.asarray(extrinsics[cam_name], dtype=np.float32)
    if mat.shape == (4, 4):
        return mat
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :] = mat
        return out
    if mat.size == 12:
        out = np.eye(4, dtype=np.float32)
        out[:3, :] = mat.reshape(3, 4)
        return out
    raise ValueError(f"Unsupported extrinsic shape for {cam_name}: {mat.shape}")


def _make_img_aug_matrix(sx: float, sy: float) -> np.ndarray:
    aug = np.eye(4, dtype=np.float32)
    aug[0, 0] = sx
    aug[1, 1] = sy
    return aug


def _select_cameras(cams: dict[str, str], camera_set: str) -> list[str]:
    if camera_set == "phenikaa6":
        return [cam for cam in PHENIKAA6_CAMERAS if cam in cams]
    if camera_set == "all10":
        return [cam for cam in PHENIKAA10_CAMERAS if cam in cams]
    raise ValueError(f"Unknown camera set: {camera_set}")


def _make_can_bus(info: dict[str, Any]) -> np.ndarray:
    can_bus = np.zeros(18, dtype=np.float32)
    can_bus[:3] = np.asarray(info.get("ego_translation", [0.0, 0.0, 0.0]), dtype=np.float32)
    return can_bus


def _prepare_frame(
    info: dict[str, Any],
    index: int,
    device: Any,
    camera_set: str,
    undistort: bool,
) -> tuple[Any, Any, list[list[dict[str, Any]]]]:
    import torch

    intrinsics, extrinsics = _load_calib()
    cams = info.get("cams", {})
    cam_names = _select_cameras(cams, camera_set)
    if not cam_names:
        raise ValueError(f"Frame has no cameras: {info.get('token', index)}")

    imgs = []
    filenames = []
    img_shapes = []
    pad_shapes = []
    lidar2img = []
    lidar2cam = []
    cam2lidar = []
    camera2ego = []
    camera_intrinsics = []
    img_aug_matrix = []
    for cam_name in cam_names:
        if cam_name not in intrinsics:
            raise KeyError(f"Missing intrinsic for camera {cam_name} in {INTRINSIC_JSON}")
        if cam_name not in extrinsics:
            raise KeyError(f"Missing extrinsic for camera {cam_name} in {EXTRINSIC_JSON}")

        filename = str(cams[cam_name])
        img_chw, this_img_shape, this_pad_shape, K_used, sx, sy = _load_and_preprocess_image(
            filename,
            intrinsics,
            cam_name,
            undistort,
        )
        imgs.append(img_chw)
        filenames.append(filename)
        img_shapes.append(this_img_shape)
        pad_shapes.append(this_pad_shape)

        K4 = _camera_intrinsic_4x4(K_used, sx, sy)
        l2c = _lidar_to_camera_4x4(extrinsics, cam_name)
        c2l = np.linalg.inv(l2c).astype(np.float32)
        aug = _make_img_aug_matrix(1.0, 1.0)
        lidar2cam.append(l2c)
        cam2lidar.append(c2l)
        camera2ego.append(c2l)  # Treat Phenikaa LiDAR frame as ego frame for E1.
        camera_intrinsics.append(K4)
        img_aug_matrix.append(aug)
        lidar2img.append((K4 @ l2c).astype(np.float32))

    img_tensor = torch.from_numpy(np.stack(imgs, axis=0)).unsqueeze(0).to(device)
    points = _load_lidar_points(str(info.get("lidar_path", "")), device)
    lidar2ego = np.eye(4, dtype=np.float32)
    meta = {
        "filename": filenames,
        "ori_shape": img_shapes,
        "img_shape": img_shapes,
        "pad_shape": pad_shapes,
        "scale_factor": [0.5 for _ in cam_names],
        "flip": False,
        "box_mode_3d": None,
        "box_type_3d": None,
        "img_norm_cfg": {"mean": IMG_MEAN, "std": IMG_STD, "to_rgb": True},
        "sample_idx": str(info.get("token", index)),
        "scene_token": str(info.get("sequence", "RESIDENTIAL_AREA")),
        "pts_filename": str(info.get("lidar_path", "")),
        "lidar2img": lidar2img,
        "lidar2cam": lidar2cam,
        "cam2lidar": cam2lidar,
        "camera2ego": camera2ego,
        "camera_intrinsics": camera_intrinsics,
        "cam_intrinsic": camera_intrinsics,
        "img_aug_matrix": img_aug_matrix,
        "lidar_aug_matrix": np.eye(4, dtype=np.float32),
        "lidar2ego": lidar2ego,
        "lidar2global": lidar2ego,
        "can_bus": _make_can_bus(info),
        "timestamp": info.get("timestamp", index),
        "phenikaa_camera_set": camera_set,
        "phenikaa_undistort": undistort,
        "phenikaa_lidar_points": int(points.shape[0]),
    }
    return img_tensor, [[points]], [[meta]]


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _prediction_to_vectors(pred: dict[str, Any], score_thresh: float) -> list[dict[str, Any]]:
    det = pred.get("pts_bbox", pred)
    required = {"scores_3d", "labels_3d", "pts_3d"}
    if not required.issubset(det):
        return []

    scores = _tensor_to_numpy(det["scores_3d"]).reshape(-1)
    labels = _tensor_to_numpy(det["labels_3d"]).reshape(-1)
    pts = _tensor_to_numpy(det["pts_3d"])
    vectors = []
    for score, label, line in zip(scores, labels, pts):
        if float(score) < score_thresh:
            continue
        class_id = int(label)
        line = np.asarray(line, dtype=np.float64)
        if line.ndim != 2 or line.shape[0] < 2:
            continue
        vectors.append(
            {
                "pts": line[:, :2].tolist(),
                "pts_num": int(line.shape[0]),
                "cls_name": MAPTR_CLASSES.get(class_id, f"class_{class_id}"),
                "type": class_id,
                "confidence_level": float(score),
            }
        )
    return vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infos", type=Path, default=PHENIKAA_DATA / "infos_residential.pkl")
    parser.add_argument(
        "--baseline",
        choices=("b1_camera", "b2_fusion"),
        default="b2_fusion",
        help="b1_camera = MapTRv2 camera-only; b2_fusion = MapTR camera+LiDAR fusion from B2.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs/phenikaa_vis/phenikaa_b3/maptr_results/nuscmap_results.json",
    )
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--score-thresh", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--camera-set",
        choices=("phenikaa6", "all10"),
        default=None,
        help="Default is phenikaa6 for b2_fusion and all10 for b1_camera.",
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="Disable image undistortion from Camera_Intrinsics.json distortion coefficients.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    defaults = BASELINE_DEFAULTS[args.baseline]
    if args.config is None:
        args.config = defaults["config"]
    if args.checkpoint is None:
        args.checkpoint = defaults["checkpoint"]
    if args.camera_set is None:
        args.camera_set = defaults["camera_set"]

    for path, name in ((args.infos, "infos"), (args.config, "config"), (args.checkpoint, "checkpoint")):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

    from mmcv import Config
    from mmcv.runner import load_checkpoint
    import torch
    # RTX 40xx + torch1.9 can fail on CUDA inverse; MapTR's own test.py uses this patch.
    import tools.cuda_inverse_patch  # noqa: F401
    # Register MapTR custom modules before build_model.
    import projects.mmdet3d_plugin  # noqa: F401
    from mmdet3d.models import build_model

    infos = _load_infos(args.infos)
    cfg = Config.fromfile(str(args.config))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model_modality = str(cfg.model.get("modality", cfg.model.get("input_modality", "")))
    use_lidar_points = model_modality == "fusion" or cfg.model.get("lidar_encoder", None) is not None
    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "MapTR inference needs CUDA in this repo, but torch.cuda.is_available() is False. "
            "B2/B3 fusion needs CUDA for the sparse LiDAR backbone, and the camera baseline also uses CUDA-only BEV ops. "
            "Run on the machine/env where your GPU is visible."
        )
    if device_name == "cpu":
        raise RuntimeError(
            "CPU inference is not supported for this MapTR setup because it uses CUDA-only BEV/sparse ops. "
            "Run with --device cuda on an environment where torch.cuda.is_available() is True."
        )
    device = torch.device(device_name)

    if use_lidar_points and args.camera_set == "all10":
        print(
            "WARNING: b2_fusion checkpoint was trained with nuScenes 6-camera fusion. "
            "Use --camera-set phenikaa6 for zero-shot; all10 is for a 10-camera fine-tuned checkpoint."
        )
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(args.checkpoint), map_location="cpu")
    model.to(device)
    model.eval()

    limit = min(len(infos), args.max_frames)
    results = []
    failed = 0
    print(f"Step 1 MapTR forward: {limit} frames on {device}")
    with torch.no_grad():
        for index, info in enumerate(infos[:limit]):
            token = str(info.get("token", index))
            if index % 20 == 0:
                print(f"  [{index}/{limit}] {token}")
            try:
                img, points, img_metas = _prepare_frame(
                    info,
                    index,
                    device,
                    args.camera_set,
                    not args.no_undistort,
                )
                model_kwargs = dict(return_loss=False, rescale=True, img=[img], img_metas=img_metas)
                if use_lidar_points:
                    if not points or points[0][0].shape[0] == 0:
                        raise RuntimeError(
                            "Fusion baseline needs non-empty LiDAR points, but this frame has 0 points. "
                            "Install lazrs/laszip or convert Phenikaa .laz files to .bin."
                        )
                    model_kwargs["points"] = points
                output = model(**model_kwargs)
                pred = output[0] if isinstance(output, list) and output else output
                vectors = _prediction_to_vectors(pred, args.score_thresh)
            except Exception as exc:
                failed += 1
                print(f"  ERROR frame {index} token={token}: {exc}")
                traceback.print_exc()
                vectors = []
            results.append({"sample_token": token, "vectors": vectors})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "source": "Phenikaa",
            "model_config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "baseline": args.baseline,
            "score_thresh": args.score_thresh,
            "camera_set": args.camera_set,
            "undistort": not args.no_undistort,
            "uses_lidar_points": use_lidar_points,
            "uses_calib": {
                "intrinsics": str(INTRINSIC_JSON),
                "extrinsics": str(EXTRINSIC_JSON),
            },
        },
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    vector_count = sum(len(item["vectors"]) for item in results)
    print(f"Step 1 done: frames={len(results)} failed={failed} vectors={vector_count}")
    print(f"MapTR results: {args.out}")
    if failed == len(results):
        args.out.unlink(missing_ok=True)
        raise RuntimeError("MapTR forward failed for every frame; inspect errors above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
