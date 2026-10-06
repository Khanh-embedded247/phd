#!/usr/bin/env python3
"""Professional visualization for Phenikaa MapTR outputs.

Creates presentation-ready figures from existing pipeline artifacts:
  - global_map_overview.png: route + graph OSM + thin OSM in map coordinates.
  - frame_dashboard/*.png: selected camera context + local raw prediction + local thin map.

This script does not run inference. It only visualizes existing outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


PHENIKAA_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from pipeline_config import expand_config_value, load_pipeline_config  # noqa: E402

CLASS_COLORS = {
    "lane_divider": (30, 195, 255),
    "road_edge_marking": (0, 220, 220),
    "stop_line": (70, 90, 255),
    "ped_crossing": (255, 140, 40),
    "boundary": (60, 220, 80),
    "road_boundary": (60, 220, 80),
    "speed_bump": (200, 80, 220),
}
CAMERA_PANEL_ORDER = ["CAM_P_F", "CAM_P_FL", "CAM_P_FR", "CAM_P_B"]


def parse_args() -> argparse.Namespace:
    cfg = load_pipeline_config()
    vars_cfg = cfg["_vars"]
    vis_cfg = cfg.get("visualize", {})
    final_dir = Path(vars_cfg["FINAL_DIR"])
    parser = argparse.ArgumentParser(description="Create professional MapTR visualization figures.")
    parser.add_argument("--infos", type=Path, default=Path(vars_cfg["INFOS"]))
    parser.add_argument("--predictions", type=Path, default=final_dir / "predictions.json")
    parser.add_argument("--intermediate-dir", type=Path, default=Path(vars_cfg["INTERMEDIATE_DIR"]))
    parser.add_argument("--graph-osm", type=Path, default=final_dir / "predicted_vector_map_graph.osm")
    parser.add_argument("--thin-osm", type=Path, default=final_dir / "predicted_vector_map_thin.osm")
    parser.add_argument("--out-dir", type=Path, default=Path(expand_config_value(vis_cfg.get("professional_out_dir", "${FINAL_DIR}/visualizations/professional"), cfg)))
    parser.add_argument("--start", type=int, default=int(vis_cfg.get("start", 0)))
    parser.add_argument("--max-samples", type=int, default=int(vis_cfg.get("professional_max_samples", 6)))
    parser.add_argument("--step", type=int, default=int(vis_cfg.get("professional_step", 250)))
    parser.add_argument("--score-thresh", type=float, default=float(vis_cfg.get("score_thresh", 0.30)))
    parser.add_argument("--global-width", type=int, default=int(vis_cfg.get("professional_global_width", 2200)))
    parser.add_argument("--global-height", type=int, default=int(vis_cfg.get("professional_global_height", 1500)))
    parser.add_argument("--local-size", type=int, default=int(vis_cfg.get("professional_local_size", 900)))
    return parser.parse_args()


def quat_wxyz_to_rot(q: list[float]) -> np.ndarray:
    w, x, y, z = map(float, q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def transform_from_info(info: dict) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_rot(info["ego2global_rotation"])
    T[:3, 3] = np.asarray(info["ego2global_translation"], dtype=np.float64)
    return T


def load_infos(path: Path) -> tuple[list[dict], dict[str, dict]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) else data
    for info in infos:
        info["_T_map_ego"] = transform_from_info(info)
    return infos, {info["token"]: info for info in infos}


def semantic_to_class(tags: dict[str, str]) -> str:
    semantic = tags.get("semantic_class", "")
    if semantic == "road_boundary":
        return "boundary"
    if semantic:
        return semantic
    typ = tags.get("type", "")
    if typ == "road_border":
        return "boundary"
    if typ == "stop_line":
        return "stop_line"
    if typ == "line_thin":
        return "lane_divider"
    return typ or "unknown"


def load_osm_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    root = ET.parse(path).getroot()
    nodes: dict[str, np.ndarray] = {}
    for node in root.findall("node"):
        tags = {tag.attrib.get("k", ""): tag.attrib.get("v", "") for tag in node.findall("tag")}
        if "local_x" not in tags or "local_y" not in tags:
            continue
        nodes[node.attrib["id"]] = np.asarray([float(tags["local_x"]), float(tags["local_y"])], dtype=np.float64)
    lines = []
    for way in root.findall("way"):
        tags = {tag.attrib.get("k", ""): tag.attrib.get("v", "") for tag in way.findall("tag")}
        pts = [nodes[nd.attrib["ref"]] for nd in way.findall("nd") if nd.attrib.get("ref") in nodes]
        if len(pts) < 2:
            continue
        cls = semantic_to_class(tags)
        lines.append({"class_name": cls, "points": np.asarray(pts, dtype=np.float64), "tags": tags})
    return lines


def line_bounds(lines: list[dict], trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chunks = []
    if len(trajectory):
        chunks.append(trajectory[:, :2])
    chunks.extend(line["points"] for line in lines if len(line["points"]))
    all_pts = np.vstack(chunks) if chunks else np.zeros((1, 2), dtype=np.float64)
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    pad = max(float(np.max(hi - lo)) * 0.06, 10.0)
    return lo - pad, hi + pad


def make_mapper(lo: np.ndarray, hi: np.ndarray, width: int, height: int):
    span = hi - lo
    scale = min((width - 140) / max(float(span[0]), 1e-6), (height - 170) / max(float(span[1]), 1e-6))
    cx, cy = (lo + hi) * 0.5
    ox, oy = width * 0.5, height * 0.52

    def to_px(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        u = (pts[:, 0] - cx) * scale + ox
        v = -(pts[:, 1] - cy) * scale + oy
        return np.round(np.column_stack([u, v])).astype(np.int32)

    return to_px, scale


def draw_polyline(img: np.ndarray, pts_px: np.ndarray, color: tuple[int, int, int], thickness: int, alpha: float = 1.0, dashed: bool = False) -> None:
    if len(pts_px) < 2:
        return
    overlay = img.copy()
    for i in range(len(pts_px) - 1):
        if dashed and i % 2 == 1:
            continue
        cv2.line(overlay, tuple(pts_px[i]), tuple(pts_px[i + 1]), color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def draw_header(img: np.ndarray, title: str, subtitle: str) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 86), (18, 22, 30), -1)
    cv2.putText(img, title, (34, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(img, subtitle, (34, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (185, 195, 210), 1, cv2.LINE_AA)


def draw_legend(img: np.ndarray, x: int, y: int) -> None:
    cv2.rectangle(img, (x - 18, y - 30), (x + 430, y + 185), (28, 32, 42), -1)
    cv2.putText(img, "Legend", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    yy = y + 30
    for cls in ["boundary", "lane_divider", "road_edge_marking", "stop_line", "ped_crossing", "speed_bump"]:
        color = CLASS_COLORS.get(cls, (220, 220, 220))
        cv2.line(img, (x, yy), (x + 38, yy), color, 4, cv2.LINE_AA)
        cv2.putText(img, cls, (x + 52, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 225, 230), 1, cv2.LINE_AA)
        yy += 24
    cv2.putText(img, "thin: bold / graph: faint", (x, yy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 170, 185), 1, cv2.LINE_AA)


def draw_scale_bar(img: np.ndarray, scale: float, x: int, y: int, meters: int = 20) -> None:
    px = int(round(meters * scale))
    cv2.line(img, (x, y), (x + px, y), (235, 235, 235), 4, cv2.LINE_AA)
    cv2.line(img, (x, y - 7), (x, y + 7), (235, 235, 235), 2, cv2.LINE_AA)
    cv2.line(img, (x + px, y - 7), (x + px, y + 7), (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(img, f"{meters} m", (x, y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)


def create_global_overview(infos: list[dict], graph_lines: list[dict], thin_lines: list[dict], out_path: Path, width: int, height: int) -> None:
    traj = np.asarray([info["_T_map_ego"][:2, 3] for info in infos], dtype=np.float64)
    lo, hi = line_bounds(graph_lines + thin_lines, traj)
    to_px, scale = make_mapper(lo, hi, width, height)
    img = np.full((height, width, 3), (11, 14, 19), dtype=np.uint8)

    for line in graph_lines:
        color = CLASS_COLORS.get(line["class_name"], (150, 150, 150))
        draw_polyline(img, to_px(line["points"]), color, 1, alpha=0.20)
    for line in thin_lines:
        color = CLASS_COLORS.get(line["class_name"], (240, 240, 240))
        thickness = 5 if line["class_name"] in ("stop_line", "ped_crossing", "speed_bump") else 3
        draw_polyline(img, to_px(line["points"]), color, thickness, alpha=0.96)
    if len(traj) >= 2:
        draw_polyline(img, to_px(traj), (235, 235, 235), 2, alpha=0.75, dashed=False)
        for idx in np.linspace(0, len(traj) - 1, min(16, len(traj))).astype(int):
            p = to_px(traj[idx:idx + 1])[0]
            cv2.circle(img, tuple(p), 4, (255, 255, 255), -1, cv2.LINE_AA)

    counts = {}
    for line in thin_lines:
        counts[line["class_name"]] = counts.get(line["class_name"], 0) + 1
    subtitle = "route frames={} | graph vectors={} | thin vectors={} | {}".format(
        len(infos), len(graph_lines), len(thin_lines), ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    )
    draw_header(img, "Phenikaa MapTR HD Vector Map Overview", subtitle)
    draw_legend(img, width - 470, 130)
    draw_scale_bar(img, scale, 52, height - 58)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def local_to_pixel(points: np.ndarray, size: int, x_range=(-30.0, 30.0), y_range=(-30.0, 30.0)) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    u = (pts[:, 1] - y_range[0]) / (y_range[1] - y_range[0]) * (size - 1)
    v = (x_range[1] - pts[:, 0]) / (x_range[1] - x_range[0]) * (size - 1)
    return np.round(np.column_stack([u, v])).astype(np.int32)


def draw_local_bev(records: list[dict], title: str, size: int, score_thresh: float = 0.0) -> np.ndarray:
    img = np.full((size, size, 3), (18, 20, 25), dtype=np.uint8)
    for k in range(-30, 31, 10):
        p1 = local_to_pixel(np.asarray([[-30, k], [30, k]], dtype=np.float64), size)
        p2 = local_to_pixel(np.asarray([[k, -30], [k, 30]], dtype=np.float64), size)
        cv2.line(img, tuple(p1[0]), tuple(p1[1]), (42, 48, 58), 1, cv2.LINE_AA)
        cv2.line(img, tuple(p2[0]), tuple(p2[1]), (42, 48, 58), 1, cv2.LINE_AA)
    car = local_to_pixel(np.asarray([[-1.8, -0.9], [1.8, 0.9]], dtype=np.float64), size)
    cv2.rectangle(img, tuple(car[0]), tuple(car[1]), (245, 245, 245), 2, cv2.LINE_AA)
    nose = local_to_pixel(np.asarray([[2.2, 0.0]], dtype=np.float64), size)[0]
    cv2.circle(img, tuple(nose), 5, (245, 245, 245), -1, cv2.LINE_AA)
    for rec in records:
        if float(rec.get("score", 1.0)) < score_thresh:
            continue
        pts = np.asarray(rec.get("points", []), dtype=np.float64)
        if pts.ndim != 2 or len(pts) < 2:
            continue
        cls = rec.get("class_name", rec.get("semantic_class", "unknown"))
        color = CLASS_COLORS.get(cls, (190, 190, 190))
        thickness = 4 if cls in ("stop_line", "ped_crossing", "speed_bump") else 2
        draw_polyline(img, local_to_pixel(pts[:, :2], size), color, thickness, alpha=0.95)
    cv2.rectangle(img, (0, 0), (size, 42), (25, 29, 38), -1)
    cv2.putText(img, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (238, 238, 238), 2, cv2.LINE_AA)
    return img


def crop_thin_to_local(thin_lines: list[dict], info: dict) -> list[dict]:
    T = info["_T_map_ego"]
    inv = np.linalg.inv(T)
    out = []
    for line in thin_lines:
        pts = line["points"]
        pts_h = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(len(pts)), np.ones(len(pts))])
        local = (inv @ pts_h.T).T[:, :2]
        mask = (local[:, 0] >= -35) & (local[:, 0] <= 35) & (local[:, 1] >= -35) & (local[:, 1] <= 35)
        if mask.sum() < 2:
            continue
        out.append({"class_name": line["class_name"], "points": local})
    return out


def resize_to(img: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), (18, 20, 25), dtype=np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def camera_panel(info: dict, names: list[str], width: int, tile_h: int) -> np.ndarray:
    tiles = []
    for name in names:
        cam = info.get("cams", {}).get(name)
        img = cv2.imread(cam["data_path"]) if cam else None
        if img is None:
            img = np.full((tile_h, width, 3), (35, 38, 45), dtype=np.uint8)
        tile = resize_to(img, width, tile_h)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 30), (25, 29, 38), -1)
        cv2.putText(tile, name, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(tile)
    return np.hstack(tiles)


def prediction_files(intermediate_dir: Path) -> list[Path]:
    return sorted(intermediate_dir.glob("*/prediction.json"))


def create_frame_dashboards(files: list[Path], infos_by_token: dict[str, dict], thin_lines: list[dict], out_dir: Path, args: argparse.Namespace) -> None:
    dash_dir = out_dir / "frame_dashboard"
    dash_dir.mkdir(parents=True, exist_ok=True)
    selected = files[args.start::max(args.step, 1)]
    if args.max_samples > 0:
        selected = selected[:args.max_samples]
    for pred_file in selected:
        sample = json.loads(pred_file.read_text(encoding="utf-8"))
        token = sample.get("token")
        info = infos_by_token.get(token)
        if not info:
            continue
        cam = camera_panel(info, CAMERA_PANEL_ORDER, 420, 250)
        raw = draw_local_bev(sample.get("pred", []) or [], "Raw model prediction", args.local_size, args.score_thresh)
        thin = draw_local_bev(crop_thin_to_local(thin_lines, info), "Postprocessed thin HD map", args.local_size, 0.0)
        bottom = np.hstack([resize_to(raw, cam.shape[1] // 2, 620), resize_to(thin, cam.shape[1] // 2, 620)])
        canvas = np.vstack([cam, bottom])
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 52), (12, 15, 22), -1)
        title = f"Frame {sample.get('index', '?')} | {token}"
        cv2.putText(canvas, title, (22, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 2, cv2.LINE_AA)
        out = dash_dir / f"{int(sample.get('index', 0)):06d}_{token}_dashboard.png"
        cv2.imwrite(str(out), canvas)
        print(f"[OK] {out}")


def main() -> None:
    args = parse_args()
    infos, infos_by_token = load_infos(args.infos.expanduser().resolve())
    graph_lines = load_osm_lines(args.graph_osm.expanduser().resolve())
    thin_lines = load_osm_lines(args.thin_osm.expanduser().resolve())
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    overview = out_dir / "global_map_overview.png"
    create_global_overview(infos, graph_lines, thin_lines, overview, args.global_width, args.global_height)
    print(f"[OK] {overview}")

    files = prediction_files(args.intermediate_dir.expanduser().resolve())
    create_frame_dashboards(files, infos_by_token, thin_lines, out_dir, args)
    print("[DONE]")
    print(f"Output dir : {out_dir}")
    print(f"Graph lines: {len(graph_lines)}")
    print(f"Thin lines : {len(thin_lines)}")


if __name__ == "__main__":
    main()
