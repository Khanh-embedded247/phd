#!/usr/bin/env python3
"""Gom prediction MapTR bi trung lap thanh map mong hon.

Van de cua raw prediction:
  - Moi frame MapTR du doan mot patch local quanh xe.
  - Cung mot vach/road boundary se xuat hien lai o rat nhieu frame lien tiep.
  - Neu xuat thang tat ca ra OSM thi map bi day dac, moi vach thanh hang tram line chong len nhau.

Y tuong script nay:
  1. Doc predictions.json va infos.pkl.
  2. Dung quang duong xe chay lam truc s.
  3. Doi moi diem prediction thanh (s_global, lateral_y).
  4. Gom theo class + cum lateral_y.
  5. Moi 1m tren truc s chi giu mot diem trung binh.
  6. Lam muot polyline va xuat OSM mong: moi vach/lane boundary chi con mot duong.

Output la OSM kieu JOSM/Lanelet2 de de mo bang Tier4 Vector Map Builder.
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
DEFAULT_OUT = PHENIKAA_ROOT / "outputs" / "benchmark_maptr" / "Normal" / "final_run" / "no_metric_full_normal" / "predicted_vector_map_thin.osm"

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from pipeline_config import expand_config_value, load_pipeline_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    pipe_cfg = load_pipeline_config()
    vars_cfg = pipe_cfg["_vars"]
    export_cfg = pipe_cfg.get("export_osm", {})
    thin_cfg = export_cfg.get("thin", {})
    parser = argparse.ArgumentParser(description="Postprocess MapTR predictions into thin OSM.")
    parser.add_argument("--predictions", type=Path, default=Path(vars_cfg["INTERMEDIATE_DIR"]) / "predictions.json")
    parser.add_argument("--infos", type=Path, default=Path(vars_cfg.get("INFOS", DEFAULT_INFOS)))
    parser.add_argument("--out-osm", type=Path, default=Path(expand_config_value(thin_cfg.get("out_osm", DEFAULT_OUT), pipe_cfg)))
    parser.add_argument("--score-thresh", type=float, default=float(export_cfg.get("score_thresh", 0.30)))
    parser.add_argument("--lateral-bin-m", type=float, default=float(thin_cfg.get("lateral_bin_m", 0.25)))
    parser.add_argument("--min-lateral-bin-ratio", type=float, default=float(thin_cfg.get("min_lateral_bin_ratio", 0.02)))
    parser.add_argument("--min-lateral-cluster-points", type=int, default=int(thin_cfg.get("min_lateral_cluster_points", 500)))
    parser.add_argument("--s-bin-m", type=float, default=float(thin_cfg.get("s_bin_m", 1.0)))
    parser.add_argument("--min-bin-observations", type=int, default=int(thin_cfg.get("min_bin_observations", 3)))
    parser.add_argument("--min-line-points", type=int, default=int(thin_cfg.get("min_line_points", 8)))
    parser.add_argument("--smooth-window", type=int, default=int(thin_cfg.get("smooth_window", 7)))
    parser.add_argument(
        "--create-lanelet-relations",
        action="store_true",
        default=bool(thin_cfg.get("create_lanelet_relations", False)),
        help="Chi bat khi cac boundary da sach. Mac dinh tat de tranh noi lanelet lung tung.",
    )
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


def load_infos(path: Path) -> tuple[dict[str, dict], dict[str, float]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) else data
    by_token = {}
    stations = {}
    prev_xy = None
    station = 0.0
    for info in infos:
        T = transform_from_info(info)
        xy = T[:2, 3]
        if prev_xy is not None:
            station += float(np.linalg.norm(xy - prev_xy))
        prev_xy = xy
        item = dict(info)
        item["_T_map_lidar"] = T
        by_token[info["token"]] = item
        stations[info["token"]] = station
    return by_token, stations


def moving_average(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(points) < window:
        return points
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    out = np.column_stack([
        np.convolve(padded[:, 0], kernel, mode="valid"),
        np.convolve(padded[:, 1], kernel, mode="valid"),
    ])
    return out


def collect_points(predictions: list[dict], infos: dict[str, dict], stations: dict[str, float], score_thresh: float) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for sample in predictions:
        token = sample.get("token")
        if token not in infos:
            continue
        T = infos[token]["_T_map_lidar"]
        station0 = stations[token]
        for rec in sample.get("pred", []) or []:
            if float(rec.get("score", 0.0)) < score_thresh:
                continue
            cls = rec.get("class_name", str(rec.get("label", "")))
            pts = np.asarray(rec.get("points", []), dtype=np.float64)
            if pts.ndim != 2 or len(pts) < 2:
                continue
            pts_h = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(len(pts)), np.ones(len(pts))])
            pts_map = (T @ pts_h.T).T[:, :2]
            score = float(rec.get("score", 1.0))
            for local, map_xy in zip(pts, pts_map):
                groups.setdefault(cls, []).append({
                    "s": station0 + float(local[0]),
                    "lat": float(local[1]),
                    "x": float(map_xy[0]),
                    "y": float(map_xy[1]),
                    "score": score,
                })
    return groups


def cluster_lateral(values: np.ndarray, bin_size: float, min_bin_ratio: float, min_cluster_points: int) -> list[tuple[float, float]]:
    """Tim cac cum lateral bang histogram de bo nhiem thua giua cac vach."""
    if len(values) == 0:
        return []
    lo = float(np.floor(values.min() / bin_size) * bin_size)
    hi = float(np.ceil(values.max() / bin_size) * bin_size + bin_size)
    hist, edges = np.histogram(values, bins=np.arange(lo, hi + bin_size, bin_size))
    if len(hist) == 0 or hist.max() == 0:
        return []
    active = hist >= max(1, int(hist.max() * min_bin_ratio))
    clusters = []
    start_idx = None
    for idx, is_active in enumerate(active.tolist() + [False]):
        if is_active and start_idx is None:
            start_idx = idx
        elif not is_active and start_idx is not None:
            end_idx = idx - 1
            count = int(hist[start_idx:end_idx + 1].sum())
            if count >= min_cluster_points:
                clusters.append((float(edges[start_idx]), float(edges[end_idx + 1])))
            start_idx = None
    return clusters


def build_thin_lines(groups: dict[str, list[dict]], args: argparse.Namespace) -> list[dict]:
    lines = []
    for cls, rows in groups.items():
        lat_values = np.asarray([r["lat"] for r in rows], dtype=np.float64)
        lat_clusters = cluster_lateral(
            lat_values,
            args.lateral_bin_m,
            args.min_lateral_bin_ratio,
            args.min_lateral_cluster_points,
        )
        for lat_min, lat_max in lat_clusters:
            selected = [r for r in rows if lat_min <= r["lat"] <= lat_max]
            if not selected:
                continue
            bins: dict[int, list[dict]] = {}
            for r in selected:
                s_bin = int(np.floor(r["s"] / args.s_bin_m))
                bins.setdefault(s_bin, []).append(r)
            line_pts = []
            line_scores = []
            for s_bin in sorted(bins):
                vals = bins[s_bin]
                if len(vals) < args.min_bin_observations:
                    continue
                weights = np.asarray([v["score"] for v in vals], dtype=np.float64)
                xy = np.asarray([[v["x"], v["y"]] for v in vals], dtype=np.float64)
                mean_xy = (xy * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-9)
                line_pts.append(mean_xy)
                line_scores.append(float(weights.mean()))
            if len(line_pts) < args.min_line_points:
                continue
            pts = moving_average(np.asarray(line_pts, dtype=np.float64), args.smooth_window)
            lines.append({
                "class_name": cls,
                "score": float(np.mean(line_scores)) if line_scores else 1.0,
                "lat_range": [lat_min, lat_max],
                "points": pts,
            })
    return lines


def add_tag(parent: ET.Element, key: str, value: str) -> None:
    ET.SubElement(parent, "tag", {"k": key, "v": value})


def indent_xml(elem: ET.Element, level: int = 0) -> None:
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


def write_osm(lines: list[dict], out_path: Path, create_lanelet_relations: bool) -> None:
    # Tier4/JOSM de doc hon voi root OSM 0.6 va moi object co version.
    osm = ET.Element("osm", {"version": "0.6", "generator": "JOSM"})

    next_node_id = 1
    next_way_id = 1
    next_relation_id = 1
    way_infos = []

    for line_idx, line in enumerate(lines):
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

        way_id = str(next_way_id)
        next_way_id += 1
        way = ET.SubElement(osm, "way", {
            "id": way_id,
            "visible": "true",
            "version": "1",
        })
        for node_id in node_ids:
            ET.SubElement(way, "nd", {"ref": node_id})
        cls = line["class_name"]
        if cls in ("divider", "lane_divider"):
            add_tag(way, "colour", "white")
            add_tag(way, "type", "line_thin")
            add_tag(way, "subtype", "dashed")
            add_tag(way, "semantic_class", "lane_divider")
        elif cls == "road_edge_marking":
            add_tag(way, "colour", "white")
            add_tag(way, "type", "line_thin")
            add_tag(way, "subtype", "solid")
            add_tag(way, "semantic_class", "road_edge_marking")
        elif cls == "stop_line":
            add_tag(way, "colour", "white")
            add_tag(way, "type", "stop_line")
            add_tag(way, "subtype", "solid")
            add_tag(way, "semantic_class", "stop_line")
        elif cls == "boundary":
            add_tag(way, "type", "road_border")
            add_tag(way, "subtype", "solid")
            add_tag(way, "semantic_class", "road_boundary")
        elif cls == "ped_crossing":
            add_tag(way, "type", "pedestrian_marking")
            add_tag(way, "subtype", "crosswalk")
        elif cls == "speed_bump":
            add_tag(way, "type", "speed_bump")
            add_tag(way, "subtype", "solid")
            add_tag(way, "semantic_class", "speed_bump")

        way_infos.append({
            "id": way_id,
            "class_name": cls,
            "lat_center": float(sum(line["lat_range"]) * 0.5),
            "points": np.asarray(line["points"], dtype=np.float64),
        })

    if not create_lanelet_relations:
        tree = ET.ElementTree(osm)
        if hasattr(ET, "indent"):
            ET.indent(tree, space="  ")
        else:
            indent_xml(osm)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out_path, encoding="UTF-8", xml_declaration=True)
        return

    # Tao centerline + relation lanelet giua cac road_border lien ke.
    # Chi ghep boundary, khong ghep divider/crosswalk/speed_bump.
    way_infos = [item for item in way_infos if item["class_name"] == "boundary"]
    way_infos.sort(key=lambda item: item["lat_center"])
    for right_line, left_line in zip(way_infos[:-1], way_infos[1:]):
        right_pts = right_line["points"]
        left_pts = left_line["points"]
        n = min(len(right_pts), len(left_pts))
        if n < 2:
            continue
        center_pts = (right_pts[:n] + left_pts[:n]) * 0.5
        center_node_ids = []
        for x, y in center_pts:
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
            center_node_ids.append(node_id)

        center_way_id = str(next_way_id)
        next_way_id += 1
        center_way = ET.SubElement(osm, "way", {
            "id": center_way_id,
            "visible": "true",
            "version": "1",
        })
        for node_id in center_node_ids:
            ET.SubElement(center_way, "nd", {"ref": node_id})
        add_tag(center_way, "colour", "white")
        add_tag(center_way, "type", "line_thin")
        add_tag(center_way, "subtype", "solid")

        relation_id = str(next_relation_id)
        next_relation_id += 1
        relation = ET.SubElement(osm, "relation", {
            "id": relation_id,
            "visible": "true",
            "version": "1",
        })
        ET.SubElement(relation, "member", {"type": "way", "role": "left", "ref": left_line["id"]})
        ET.SubElement(relation, "member", {"type": "way", "role": "right", "ref": right_line["id"]})
        ET.SubElement(relation, "member", {"type": "way", "role": "centerline", "ref": center_way_id})
        add_tag(relation, "type", "lanelet")
        add_tag(relation, "subtype", "road")
        add_tag(relation, "location", "urban")
        add_tag(relation, "one_way", "yes")

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
    infos, stations = load_infos(args.infos.expanduser().resolve())
    groups = collect_points(predictions, infos, stations, args.score_thresh)
    thin_lines = build_thin_lines(groups, args)
    write_osm(thin_lines, args.out_osm.expanduser().resolve(), args.create_lanelet_relations)
    print("[DONE]")
    print(f"Predictions : {args.predictions}")
    print(f"Infos       : {args.infos}")
    print(f"Output OSM  : {args.out_osm}")
    print(f"Classes     : {', '.join(f'{k}={len(v)}pts' for k, v in groups.items())}")
    print(f"Thin lines  : {len(thin_lines)}")
    print(f"Lanelet rel : {'on' if args.create_lanelet_relations else 'off'}")
    for line in thin_lines:
        print(f"  {line['class_name']:12s} points={len(line['points']):5d} lat={line['lat_range'][0]:.2f}..{line['lat_range'][1]:.2f}")


if __name__ == "__main__":
    main()
