#!/usr/bin/env python3
"""Xuat prediction MapTR thanh file OSM trong he toa do map.

MapTR du doan vector trong he local cua tung frame, tuc he lidar/base tai thoi diem anh.
Script nay:
  1. Doc prediction.json cua cac frame da inference.
  2. Doc infos.pkl de lay pose T_map_lidar cua tung frame.
  3. Doi cac polyline prediction tu local sang map.
  4. Bo bot cac line trung nhau bang Chamfer distance don gian.
  5. Ghi ra .osm voi node local_x/local_y va way mang tag loai vector.

Ghi chu:
  - Day la "predicted vector map" dang raw/merged don gian.
  - Chua phai Lanelet2 day du voi relation lanelet left/right/centerline.
"""

from __future__ import annotations

import argparse
import json
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


PHENIKAA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRED_DIR = PHENIKAA_ROOT / "outputs" / "benchmark_maptr" / "Normal" / "inference_scan"
DEFAULT_INFOS = PHENIKAA_ROOT / "outputs" / "benchmark_maptr" / "Normal" / "phenikaa_maptr_infos_train_osm.pkl"
DEFAULT_OUT = PHENIKAA_ROOT / "outputs" / "benchmark_maptr" / "Normal" / "predicted_vector_map.osm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MapTR predictions to OSM map coordinates.")
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--infos", type=Path, default=DEFAULT_INFOS)
    parser.add_argument("--out-osm", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--score-thresh", type=float, default=0.30)
    parser.add_argument("--dedup-thresh-m", type=float, default=0.50)
    return parser.parse_args()


def quat_wxyz_to_rot(q: list[float]) -> np.ndarray:
    w, x, y, z = map(float, q)
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n == 0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def make_transform(rotation_wxyz: list[float], translation: list[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_rot(rotation_wxyz)
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def transform_polyline(points_xy: list[list[float]], T_map_lidar: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    pts3 = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(len(pts)), np.ones(len(pts))])
    out = (T_map_lidar @ pts3.T).T
    return out[:, :2]


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float((d.min(axis=1).mean() + d.min(axis=0).mean()) * 0.5)


def read_infos(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) else data
    by_token = {}
    for info in infos:
        by_token[info["token"]] = make_transform(
            info["ego2global_rotation"],
            info["ego2global_translation"],
        )
    return by_token


def read_prediction_samples(pred_dir: Path) -> list[dict]:
    if (pred_dir / "predictions.json").exists():
        with (pred_dir / "predictions.json").open("r", encoding="utf-8") as f:
            return json.load(f)

    files = sorted(pred_dir.glob("*/prediction.json"))
    if not files:
        files = sorted(pred_dir.glob("start_*_n_*/*/prediction.json"))
    samples = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            samples.append(json.load(f))
    return samples


def collect_lines(pred_dir: Path, infos_by_token: dict[str, np.ndarray], score_thresh: float) -> list[dict]:
    lines = []
    for sample in read_prediction_samples(pred_dir):
        token = sample["token"]
        if token not in infos_by_token:
            continue
        T_map_lidar = infos_by_token[token]
        for pred in sample.get("pred", []) or []:
            if float(pred.get("score", 0.0)) < score_thresh:
                continue
            points_map = transform_polyline(pred.get("points", []), T_map_lidar)
            if len(points_map) < 2:
                continue
            lines.append({
                "token": token,
                "label": int(pred["label"]),
                "class_name": pred["class_name"],
                "score": float(pred.get("score", 0.0)),
                "points": points_map,
            })
    return lines


def deduplicate_lines(lines: list[dict], threshold: float) -> list[dict]:
    """Bo line trung nhau bang luoi khong gian de tranh O(N^2) qua cham."""
    kept: list[dict] = []
    grid: dict[tuple[str, int, int], list[int]] = {}
    cell_size = max(float(threshold), 1e-3)
    for line in sorted(lines, key=lambda item: item["score"], reverse=True):
        center = np.asarray(line["points"], dtype=np.float64).mean(axis=0)
        cx = int(np.floor(center[0] / cell_size))
        cy = int(np.floor(center[1] / cell_size))
        duplicate = False
        candidate_ids = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidate_ids.extend(grid.get((line["class_name"], cx + dx, cy + dy), []))
        for old_idx in candidate_ids:
            old = kept[old_idx]
            if chamfer(line["points"], old["points"]) <= threshold:
                duplicate = True
                break
        if not duplicate:
            kept_idx = len(kept)
            kept.append(line)
            grid.setdefault((line["class_name"], cx, cy), []).append(kept_idx)
    return kept


def add_tag(parent: ET.Element, key: str, value: str) -> None:
    ET.SubElement(parent, "tag", {"k": key, "v": value})


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Thut le XML tuong thich Python 3.8."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def write_osm(lines: list[dict], out_path: Path) -> None:
    osm = ET.Element("osm", {"version": "0.6", "generator": "phenikaa_maptr_prediction"})
    next_id = -1

    for line_idx, line in enumerate(lines):
        node_ids = []
        for x, y in line["points"]:
            node_id = str(next_id)
            next_id -= 1
            node = ET.SubElement(osm, "node", {
                "id": node_id,
                "visible": "true",
                "lat": "0.0",
                "lon": "0.0",
            })
            add_tag(node, "local_x", f"{float(x):.6f}")
            add_tag(node, "local_y", f"{float(y):.6f}")
            node_ids.append(node_id)

        way_id = str(next_id)
        next_id -= 1
        way = ET.SubElement(osm, "way", {"id": way_id, "visible": "true"})
        for node_id in node_ids:
            ET.SubElement(way, "nd", {"ref": node_id})
        add_tag(way, "source", "MapTR_Phenikaa_prediction")
        add_tag(way, "maptr_class", line["class_name"])
        add_tag(way, "maptr_label", str(line["label"]))
        add_tag(way, "score", f"{line['score']:.6f}")
        add_tag(way, "sample_token", line["token"])
        if line["class_name"] == "divider":
            add_tag(way, "type", "line_thin")
            add_tag(way, "subtype", "dashed")
        elif line["class_name"] == "boundary":
            add_tag(way, "type", "road_border")
        elif line["class_name"] == "ped_crossing":
            add_tag(way, "type", "pedestrian_marking")
        add_tag(way, "phenikaa_pred_id", str(line_idx))

    tree = ET.ElementTree(osm)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    else:
        indent_xml(osm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    args = parse_args()
    infos_by_token = read_infos(args.infos.expanduser().resolve())
    lines = collect_lines(args.pred_dir.expanduser().resolve(), infos_by_token, args.score_thresh)
    kept = deduplicate_lines(lines, args.dedup_thresh_m)
    write_osm(kept, args.out_osm.expanduser().resolve())

    print("[DONE]")
    print(f"Input predictions : {args.pred_dir}")
    print(f"Raw lines         : {len(lines)}")
    print(f"Kept lines        : {len(kept)}")
    print(f"Output OSM        : {args.out_osm}")


if __name__ == "__main__":
    main()
