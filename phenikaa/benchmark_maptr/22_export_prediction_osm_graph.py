#!/usr/bin/env python3
"""Xuat prediction MapTR thanh OSM dang graph, giu cong/nhanh/vach ngan.

Khac voi `21_postprocess_predictions_to_thin_osm.py`:
  - File 21 ep prediction ve vai duong dai song song, hop de xem lanelet cuc gon.
  - File nay giu tung doan vector du doan sau khi bo trung lap vua phai.

Muc tieu:
  - Nhin thay doan cong, nhanh re, vach ke duong.
  - Giu duoc ped_crossing va speed_bump neu model da duoc train class do.
  - Xuat OSM local_x/local_y, lat/lon de rong giong file VMB goc.
"""

from __future__ import annotations

import argparse
import json
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import sys


PHENIKAA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRED = PHENIKAA_ROOT / "outputs" / "benchmark_maptr" / "Normal" / "final_run" / "no_metric_full_normal" / "predictions.json"
DEFAULT_INFOS = PHENIKAA_ROOT / "outputs" / "benchmark_maptr" / "Normal" / "phenikaa_maptr_infos_full_normal.pkl"
DEFAULT_OUT = PHENIKAA_ROOT / "outputs" / "benchmark_maptr" / "Normal" / "final_run" / "no_metric_full_normal" / "predicted_vector_map_graph.osm"

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from pipeline_config import expand_config_value, load_pipeline_config  # noqa: E402

CLASS_TO_OSM_TAGS = {
    "lane_divider": {"type": "line_thin", "subtype": "dashed", "colour": "white", "semantic_class": "lane_divider"},
    "road_edge_marking": {"type": "line_thin", "subtype": "solid", "colour": "white", "semantic_class": "road_edge_marking"},
    "stop_line": {"type": "stop_line", "subtype": "solid", "colour": "white", "semantic_class": "stop_line"},
    "divider": {"type": "line_thin", "subtype": "dashed", "colour": "white", "semantic_class": "lane_divider"},
    "boundary": {"type": "road_border", "subtype": "solid", "semantic_class": "road_boundary"},
    "ped_crossing": {"type": "pedestrian_marking", "subtype": "crosswalk", "colour": "white"},
    "speed_bump": {"type": "speed_bump", "subtype": "solid", "colour": "yellow", "semantic_class": "speed_bump"},
}


def parse_args() -> argparse.Namespace:
    pipe_cfg = load_pipeline_config()
    vars_cfg = pipe_cfg["_vars"]
    export_cfg = pipe_cfg.get("export_osm", {})
    graph_cfg = export_cfg.get("graph", {})
    parser = argparse.ArgumentParser(description="Export MapTR predictions to graph-style local OSM.")
    parser.add_argument("--predictions", type=Path, default=Path(vars_cfg["INTERMEDIATE_DIR"]) / "predictions.json")
    parser.add_argument("--infos", type=Path, default=Path(vars_cfg.get("INFOS", DEFAULT_INFOS)))
    parser.add_argument("--out-osm", type=Path, default=Path(expand_config_value(graph_cfg.get("out_osm", DEFAULT_OUT), pipe_cfg)))
    parser.add_argument("--score-thresh", type=float, default=float(export_cfg.get("score_thresh", 0.45)))
    parser.add_argument("--min-length-m", type=float, default=float(graph_cfg.get("min_length_m", 0.40)))
    parser.add_argument("--dedup-chamfer-m", type=float, default=float(graph_cfg.get("dedup_chamfer_m", 0.65)))
    parser.add_argument("--dedup-cell-m", type=float, default=float(graph_cfg.get("dedup_cell_m", 2.0)))
    parser.add_argument("--sample-step", type=int, default=int(graph_cfg.get("sample_step", 1)),
                        help="Lay moi N frame. =1 la dung tat ca frame.")
    parser.add_argument("--max-lines", type=int, default=int(graph_cfg.get("max_lines", 0)),
                        help="Gioi han so line sau dedup. 0 la khong gioi han.")
    return parser.parse_args()


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


def transform_from_info(info: dict) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_rot(info["ego2global_rotation"])
    T[:3, 3] = np.asarray(info["ego2global_translation"], dtype=np.float64)
    return T


def load_info_transforms(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) else data
    return {info["token"]: transform_from_info(info) for info in infos}


def line_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float((d.min(axis=1).mean() + d.min(axis=0).mean()) * 0.5)


def transform_points_to_map(points_xy: np.ndarray, T_map_lidar: np.ndarray) -> np.ndarray:
    pts_h = np.column_stack([
        points_xy[:, 0],
        points_xy[:, 1],
        np.zeros(len(points_xy), dtype=np.float64),
        np.ones(len(points_xy), dtype=np.float64),
    ])
    return (T_map_lidar @ pts_h.T).T[:, :2]


def collect_lines(predictions: list[dict], infos: dict[str, np.ndarray], args: argparse.Namespace) -> list[dict]:
    lines = []
    for sample_idx, sample in enumerate(predictions):
        if args.sample_step > 1 and sample_idx % args.sample_step != 0:
            continue
        token = sample.get("token")
        if token not in infos:
            continue
        T_map_lidar = infos[token]
        for pred_idx, pred in enumerate(sample.get("pred", []) or []):
            score = float(pred.get("score", 0.0))
            if score < args.score_thresh:
                continue
            pts = np.asarray(pred.get("points", []), dtype=np.float64)
            if pts.ndim != 2 or len(pts) < 2:
                continue
            pts_map = transform_points_to_map(pts, T_map_lidar)
            length_m = line_length(pts_map)
            if length_m < args.min_length_m:
                continue
            lines.append({
                "class_name": pred.get("class_name", str(pred.get("label", ""))),
                "label": int(pred.get("label", -1)),
                "score": score,
                "token": token,
                "sample_index": int(sample.get("index", sample_idx)),
                "pred_index": pred_idx,
                "points": pts_map,
                "length_m": length_m,
            })
    return lines


def deduplicate_lines(lines: list[dict], args: argparse.Namespace) -> list[dict]:
    """Bo bot line trung, nhung khong trung binh hoa nen van giu duong cong."""
    kept: list[dict] = []
    grid: dict[tuple[str, int, int], list[int]] = {}
    cell = max(float(args.dedup_cell_m), 1e-6)

    for line in sorted(lines, key=lambda item: item["score"], reverse=True):
        center = np.asarray(line["points"], dtype=np.float64).mean(axis=0)
        cx = int(np.floor(center[0] / cell))
        cy = int(np.floor(center[1] / cell))
        candidate_ids = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidate_ids.extend(grid.get((line["class_name"], cx + dx, cy + dy), []))

        duplicate = False
        for old_id in candidate_ids:
            old = kept[old_id]
            if chamfer(line["points"], old["points"]) <= args.dedup_chamfer_m:
                duplicate = True
                break
        if duplicate:
            continue

        kept_id = len(kept)
        kept.append(line)
        grid.setdefault((line["class_name"], cx, cy), []).append(kept_id)
        if args.max_lines > 0 and len(kept) >= args.max_lines:
            break

    return sorted(kept, key=lambda item: (item["class_name"], item["sample_index"], item["pred_index"]))


def add_tag(parent: ET.Element, key: str, value: str) -> None:
    ET.SubElement(parent, "tag", {"k": key, "v": value})


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Thut le XML cho Python 3.8."""
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
    osm = ET.Element("osm", {"version": "0.6", "generator": "JOSM"})
    next_node_id = 1
    next_way_id = 1

    for line in lines:
        node_ids = []
        for x, y in line["points"]:
            node_id = str(next_node_id)
            next_node_id += 1
            node = ET.SubElement(osm, "node", {
                "id": node_id,
                "visible": "true",
                "version": "1",
                "lat": "",
                "lon": "",
            })
            add_tag(node, "local_x", f"{float(x):.6f}")
            add_tag(node, "local_y", f"{float(y):.6f}")
            add_tag(node, "ele", "0.000000")
            node_ids.append(node_id)

        way = ET.SubElement(osm, "way", {
            "id": str(next_way_id),
            "visible": "true",
            "version": "1",
        })
        next_way_id += 1
        for node_id in node_ids:
            ET.SubElement(way, "nd", {"ref": node_id})

        for key, value in CLASS_TO_OSM_TAGS.get(line["class_name"], {"type": line["class_name"]}).items():
            add_tag(way, key, value)
        add_tag(way, "maptr_class", line["class_name"])
        add_tag(way, "score", f"{line['score']:.6f}")
        add_tag(way, "sample_index", str(line["sample_index"]))
        add_tag(way, "sample_token", line["token"])

    tree = ET.ElementTree(osm)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    else:
        indent_xml(osm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="UTF-8", xml_declaration=True)


def main() -> None:
    args = parse_args()
    with args.predictions.expanduser().resolve().open("r", encoding="utf-8") as f:
        predictions = json.load(f)
    infos = load_info_transforms(args.infos.expanduser().resolve())
    raw_lines = collect_lines(predictions, infos, args)
    kept_lines = deduplicate_lines(raw_lines, args)
    write_osm(kept_lines, args.out_osm.expanduser().resolve())

    counts = {}
    for line in kept_lines:
        counts[line["class_name"]] = counts.get(line["class_name"], 0) + 1

    print("[DONE]")
    print(f"Predictions : {args.predictions}")
    print(f"Infos       : {args.infos}")
    print(f"Output OSM  : {args.out_osm}")
    print(f"Raw lines   : {len(raw_lines)}")
    print(f"Kept lines  : {len(kept_lines)}")
    print("Classes     : " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
