#!/usr/bin/env python3
"""Tao anh visualize kieu MapTR paper cho Phenikaa.

Input la thu muc inference co cac sample */prediction.json.
Output la anh montage: cac camera xung quanh + Prediction BEV + GT BEV neu co.

Ghi chu:
  - Duong prediction/GT duoc chieu len anh voi gia dinh vector nam tren mat phang z=0
    trong he lidar/base local cua frame.
  - Khi chay --no-gt o inference, cot GT se rong; day la che do dung cho doan chua label.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np


PHENIKAA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRED_DIR = PHENIKAA_ROOT / "outputs" / "benchmark_maptr" / "Normal" / "inference_unlabeled_val"
DEFAULT_INFOS = PHENIKAA_ROOT / "outputs" / "benchmark_maptr" / "Normal" / "phenikaa_maptr_infos_val.pkl"
DEFAULT_OUT_DIR = PHENIKAA_ROOT / "outputs" / "benchmark_maptr" / "Normal" / "paper_style_unlabeled_val"

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from pipeline_config import expand_config_value, load_pipeline_config  # noqa: E402

CAMERA_ORDER = [
    "CAM_P_FL", "CAM_P_F", "CAM_P_FR",
    "CAM_P_L", "CAM_P_B", "CAM_P_R",
    "CAM_P_LB", "CAM_P_RB", "CAM_F_F",
    "CAM_F_L", "CAM_F_R", "CAM_F_B",
]
COLORS = {
    "divider": (0, 165, 255),
    "ped_crossing": (255, 80, 30),
    "boundary": (0, 0, 255),
    "speed_bump": (180, 60, 180),
}


def parse_args() -> argparse.Namespace:
    pipe_cfg = load_pipeline_config()
    vars_cfg = pipe_cfg["_vars"]
    vis_cfg = pipe_cfg.get("visualize", {})
    parser = argparse.ArgumentParser(description="Visualize MapTR results in paper style.")
    parser.add_argument("--pred-dir", type=Path, default=Path(vars_cfg.get("INTERMEDIATE_DIR", DEFAULT_PRED_DIR)))
    parser.add_argument("--infos", type=Path, default=Path(vars_cfg.get("INFOS", DEFAULT_INFOS)))
    parser.add_argument("--out-dir", type=Path, default=Path(expand_config_value(vis_cfg.get("paper_out_dir", DEFAULT_OUT_DIR), pipe_cfg)))
    parser.add_argument("--style", choices=("paper", "map"), default="paper",
                        help="paper = 12 camera + BEV, map = clean BEV map giong MapTR vis_pred.")
    parser.add_argument("--max-samples", type=int, default=int(vis_cfg.get("max_samples", 8)))
    parser.add_argument("--start", type=int, default=int(vis_cfg.get("start", 0)))
    parser.add_argument("--score-thresh", type=float, default=float(vis_cfg.get("score_thresh", 0.30)))
    parser.add_argument("--cam-width", type=int, default=int(vis_cfg.get("cam_width", 320)))
    parser.add_argument("--map-width", type=int, default=int(vis_cfg.get("map_width", 900)))
    parser.add_argument("--map-height", type=int, default=int(vis_cfg.get("map_height", 1600)))
    return parser.parse_args()


def load_infos(path: Path) -> dict[str, dict]:
    with path.open("rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) else data
    return {info["token"]: info for info in infos}


def prediction_files(pred_dir: Path) -> list[Path]:
    files = sorted(pred_dir.glob("*/prediction.json"))
    if files:
        return files
    return sorted(pred_dir.glob("start_*_n_*/*/prediction.json"))


def lidar2img_matrix(cam: dict) -> np.ndarray:
    R_cam_lidar = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float64)
    t_cam_lidar = np.asarray(cam["sensor2lidar_translation"], dtype=np.float64).reshape(3, 1)
    R_lidar_cam = R_cam_lidar.T
    t_lidar_cam = -R_lidar_cam @ t_cam_lidar
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_lidar_cam
    T[:3, 3] = t_lidar_cam[:, 0]
    K = np.eye(4, dtype=np.float64)
    K[:3, :3] = np.asarray(cam["cam_intrinsic"], dtype=np.float64)
    return K @ T


def quat_wxyz_to_rot(q: list[float]) -> np.ndarray:
    w, x, y, z = map(float, q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def lidar2ego_matrix(info: dict) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_rot(info.get("lidar2ego_rotation", [1.0, 0.0, 0.0, 0.0]))
    T[:3, 3] = np.asarray(info.get("lidar2ego_translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    return T


def project_polyline(points: list[list[float]], lidar2img: np.ndarray, ego2lidar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) == 0:
        return np.empty((0, 2)), np.zeros(0, dtype=bool)
    pts_ego_h = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(len(pts)), np.ones(len(pts))])
    pts_lidar_h = (ego2lidar @ pts_ego_h.T).T
    pts_h = np.column_stack([pts_lidar_h[:, 0], pts_lidar_h[:, 1], pts_lidar_h[:, 2], np.ones(len(pts))])
    proj = (lidar2img @ pts_h.T).T
    z = proj[:, 2]
    valid = z > 1e-6
    uv = np.zeros((len(pts), 2), dtype=np.float64)
    uv[valid, 0] = proj[valid, 0] / z[valid]
    uv[valid, 1] = proj[valid, 1] / z[valid]
    return uv, valid


def draw_records(img: np.ndarray, records: list[dict], cam: dict, info: dict, score_thresh: float, dashed: bool) -> np.ndarray:
    h, w = img.shape[:2]
    mat = lidar2img_matrix(cam)
    ego2lidar = np.linalg.inv(lidar2ego_matrix(info))
    out = img.copy()
    for rec in records:
        if "score" in rec and float(rec["score"]) < score_thresh:
            continue
        uv, valid_z = project_polyline(rec.get("points", []), mat, ego2lidar)
        valid = valid_z & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        pts = uv.astype(np.int32)
        color = COLORS.get(rec.get("class_name", ""), (0, 255, 0))
        for i in range(len(pts) - 1):
            if valid[i] and valid[i + 1] and (not dashed or i % 2 == 0):
                cv2.line(out, tuple(pts[i]), tuple(pts[i + 1]), color, 2, cv2.LINE_AA)
    return out


def resize_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = max(1, int(round(h * width / w)))
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)


def put_title(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (255, 255, 255), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def make_camera_grid(sample: dict, info: dict, width: int, score_thresh: float) -> np.ndarray:
    tiles = []
    gt = sample.get("gt", []) or []
    pred = sample.get("pred", []) or []
    for name in CAMERA_ORDER:
        cam = info["cams"].get(name)
        if cam is None:
            continue
        img = cv2.imread(cam["data_path"])
        if img is None:
            continue
        if gt:
            img = draw_records(img, gt, cam, info, score_thresh, dashed=True)
        img = draw_records(img, pred, cam, info, score_thresh, dashed=False)
        tiles.append(put_title(resize_width(img, width), name))

    rows = []
    for i in range(0, len(tiles), 3):
        row = tiles[i:i + 3]
        max_h = max(t.shape[0] for t in row)
        padded = []
        for tile in row:
            if tile.shape[0] < max_h:
                pad = np.full((max_h - tile.shape[0], tile.shape[1], 3), 255, dtype=np.uint8)
                tile = np.vstack([tile, pad])
            padded.append(tile)
        rows.append(np.hstack(padded))
    return np.vstack(rows) if rows else np.full((480, width * 3, 3), 255, dtype=np.uint8)


def read_panel(path: Path, height: int, title: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        img = np.full((height, max(1, height // 2), 3), 255, dtype=np.uint8)
    else:
        scale = height / img.shape[0]
        img = cv2.resize(img, (max(1, int(round(img.shape[1] * scale))), height), interpolation=cv2.INTER_AREA)
    return put_title(img, title)


def legend(width: int) -> np.ndarray:
    out = np.full((36, width, 3), 255, dtype=np.uint8)
    x = 10
    for name, color in COLORS.items():
        cv2.circle(out, (x, 18), 6, color, -1)
        cv2.putText(out, name, (x + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        x += 145
    cv2.putText(out, "GT dashed / Prediction solid", (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def point_to_map_pixel(point_xy: np.ndarray, width: int, height: int) -> tuple[int, int]:
    """Doi toa do local lidar/base sang pixel BEV giong MapTR: y ngang, x doc."""
    x_forward = float(point_xy[0])
    y_left = float(point_xy[1])
    x_min, x_max = -15.0, 15.0
    y_min, y_max = -30.0, 30.0
    u = int(round((y_left - y_min) / (y_max - y_min) * (width - 1)))
    v = int(round((x_max - x_forward) / (x_max - x_min) * (height - 1)))
    return u, v


def draw_vehicle_icon(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    car_w, car_h = max(24, w // 18), max(44, h // 20)
    cv2.rectangle(img, (cx - car_w // 2, cy - car_h // 2),
                  (cx + car_w // 2, cy + car_h // 2), (255, 70, 140), -1, cv2.LINE_AA)
    cv2.rectangle(img, (cx - car_w // 4, cy - car_h // 5),
                  (cx + car_w // 4, cy + car_h // 5), (80, 20, 60), 2, cv2.LINE_AA)
    cv2.line(img, (cx - car_w // 2, cy - car_h // 3),
             (cx + car_w // 2, cy - car_h // 3), (80, 20, 60), 2, cv2.LINE_AA)
    cv2.line(img, (cx - car_w // 2, cy + car_h // 3),
             (cx + car_w // 2, cy + car_h // 3), (80, 20, 60), 2, cv2.LINE_AA)


def draw_map_records(
    records: list[dict],
    width: int,
    height: int,
    score_thresh: float,
    title: str,
    dashed: bool = False,
) -> np.ndarray:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    draw_vehicle_icon(img)
    for rec in records:
        if "score" in rec and float(rec["score"]) < score_thresh:
            continue
        pts = np.asarray(rec.get("points", []), dtype=np.float64)
        if pts.ndim != 2 or len(pts) < 2:
            continue
        color = COLORS.get(rec.get("class_name", ""), (0, 180, 0))
        pixels = [point_to_map_pixel(pt, width, height) for pt in pts]
        for i in range(len(pixels) - 1):
            if dashed and i % 2 == 1:
                continue
            cv2.line(img, pixels[i], pixels[i + 1], color, 5, cv2.LINE_AA)
        for p in pixels:
            cv2.circle(img, p, 7, color, -1, cv2.LINE_AA)
    cv2.putText(img, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    return img


def draw_overlay_map(sample: dict, width: int, height: int, score_thresh: float) -> np.ndarray:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    draw_vehicle_icon(img)
    for rec in sample.get("gt", []) or []:
        pts = np.asarray(rec.get("points", []), dtype=np.float64)
        if pts.ndim != 2 or len(pts) < 2:
            continue
        color = COLORS.get(rec.get("class_name", ""), (0, 180, 0))
        pixels = [point_to_map_pixel(pt, width, height) for pt in pts]
        for i in range(len(pixels) - 1):
            if i % 2 == 0:
                cv2.line(img, pixels[i], pixels[i + 1], color, 3, cv2.LINE_AA)
    for rec in sample.get("pred", []) or []:
        if float(rec.get("score", 0.0)) < score_thresh:
            continue
        pts = np.asarray(rec.get("points", []), dtype=np.float64)
        if pts.ndim != 2 or len(pts) < 2:
            continue
        color = COLORS.get(rec.get("class_name", ""), (0, 180, 0))
        pixels = [point_to_map_pixel(pt, width, height) for pt in pts]
        for i in range(len(pixels) - 1):
            cv2.line(img, pixels[i], pixels[i + 1], color, 6, cv2.LINE_AA)
    cv2.putText(img, "OVERLAY: GT dashed / PRED solid", (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
    return img


def visualize_map_style(pred_file: Path, out_dir: Path, args: argparse.Namespace) -> list[Path]:
    with pred_file.open("r", encoding="utf-8") as f:
        sample = json.load(f)
    sample_dir = out_dir / f"{sample['index']:06d}_{sample['token']}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    written = []
    pred_img = draw_map_records(
        sample.get("pred", []) or [],
        args.map_width,
        args.map_height,
        args.score_thresh,
        "PRED_MAP",
        dashed=False,
    )
    pred_path = sample_dir / "PRED_MAP.png"
    cv2.imwrite(str(pred_path), pred_img)
    written.append(pred_path)

    if sample.get("gt"):
        gt_img = draw_map_records(
            sample.get("gt", []) or [],
            args.map_width,
            args.map_height,
            args.score_thresh,
            "GT_MAP",
            dashed=False,
        )
        gt_path = sample_dir / "GT_MAP.png"
        cv2.imwrite(str(gt_path), gt_img)
        written.append(gt_path)

        overlay = draw_overlay_map(sample, args.map_width, args.map_height, args.score_thresh)
        overlay_path = sample_dir / "OVERLAY_MAP.png"
        cv2.imwrite(str(overlay_path), overlay)
        written.append(overlay_path)
    return written


def visualize(pred_file: Path, infos: dict[str, dict], out_dir: Path, args: argparse.Namespace) -> Path | None:
    with pred_file.open("r", encoding="utf-8") as f:
        sample = json.load(f)
    info = infos.get(sample["token"])
    if info is None:
        return None
    cam_grid = make_camera_grid(sample, info, args.cam_width, args.score_thresh)
    panel_h = cam_grid.shape[0] // 2
    pred_panel = read_panel(pred_file.parent / "pred_bev.png", panel_h, "Prediction")
    gt_panel = read_panel(pred_file.parent / "gt_bev.png", panel_h, "GT")
    right = np.vstack([pred_panel, gt_panel])

    if right.shape[0] < cam_grid.shape[0]:
        right = np.vstack([right, np.full((cam_grid.shape[0] - right.shape[0], right.shape[1], 3), 255, dtype=np.uint8)])
    if cam_grid.shape[0] < right.shape[0]:
        cam_grid = np.vstack([cam_grid, np.full((right.shape[0] - cam_grid.shape[0], cam_grid.shape[1], 3), 255, dtype=np.uint8)])

    gap = np.full((cam_grid.shape[0], 16, 3), 255, dtype=np.uint8)
    montage = np.hstack([cam_grid, gap, right])
    montage = np.vstack([montage, legend(montage.shape[1])])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample['index']:06d}_{sample['token']}_paper_style.jpg"
    cv2.imwrite(str(out_path), montage)
    return out_path


def main() -> None:
    args = parse_args()
    files = prediction_files(args.pred_dir.expanduser().resolve())
    files = files[args.start:] if args.max_samples <= 0 else files[args.start:args.start + args.max_samples]
    out_dir = args.out_dir.expanduser().resolve()
    written = []
    if args.style == "map":
        for pred_file in files:
            outs = visualize_map_style(pred_file, out_dir, args)
            written.extend(outs)
            for out in outs:
                print(f"[OK] {out}")
    else:
        infos = load_infos(args.infos.expanduser().resolve())
        for pred_file in files:
            out = visualize(pred_file, infos, out_dir, args)
            if out is not None:
                written.append(out)
                print(f"[OK] {out}")
    print("[DONE]")
    print(f"Input predictions: {args.pred_dir}")
    print(f"Written          : {len(written)}")
    print(f"Output dir       : {out_dir}")


if __name__ == "__main__":
    main()
