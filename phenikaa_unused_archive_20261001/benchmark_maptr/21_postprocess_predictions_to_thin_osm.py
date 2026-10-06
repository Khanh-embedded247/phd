#!/usr/bin/env python3
"""Gom prediction MapTR bi trung lap thanh map mong hon.

Van de cua raw prediction:
  - Moi frame MapTR du doan mot patch local quanh xe.
  - Cung mot vach/road boundary se xuat hien lai o rat nhieu frame lien tiep.
  - Neu xuat thang tat ca ra OSM thi map bi day dac, moi vach thanh hang tram line chong len nhau.

Mac dinh moi:
  - Dung graph/geometry merge: chi gop cac vector cung class khi thuc su chong len nhau.
  - Khong tu noi endpoint, vi giao lo/duong cua rat de bi noi cheo.
  - Giu hinh dang vector tot nhat thay vi trung binh theo truc xe chay.

Che do cu `station_lateral` van giu lai de so sanh/thuc nghiem, nhung khong nen dung cho giao lo.

Output la OSM kieu JOSM/Lanelet2 de de mo bang Tier4 Vector Map Builder.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import sys


PHENIKAA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRED = PHENIKAA_ROOT / "outputs" / "benchmark_maptr" / "Normal" / "final_run" / "no_metric_full_normal" / "predictions.json"
DEFAULT_INFOS = PHENIKAA_ROOT / "outputs" / "benchmark_maptr" / "Normal" / "phenikaa_maptr_infos_full_normal.pkl"
DEFAULT_OUT = PHENIKAA_ROOT / "outputs" / "benchmark_maptr" / "Normal" / "final_run" / "no_metric_full_normal" / "predicted_vector_map_thin.osm"

CLASS_TO_OSM_TAGS = {
    "lane_divider": {"colour": "white", "type": "line_thin", "subtype": "dashed", "semantic_class": "lane_divider"},
    "divider": {"colour": "white", "type": "line_thin", "subtype": "dashed", "semantic_class": "lane_divider"},
    "road_edge_marking": {"colour": "white", "type": "line_thin", "subtype": "solid", "semantic_class": "road_edge_marking"},
    "stop_line": {"colour": "white", "type": "stop_line", "subtype": "solid", "semantic_class": "stop_line"},
    "boundary": {"type": "road_border", "subtype": "solid", "semantic_class": "road_boundary"},
    "ped_crossing": {"colour": "white", "type": "pedestrian_marking", "subtype": "crosswalk", "semantic_class": "ped_crossing"},
    "speed_bump": {"type": "speed_bump", "subtype": "solid", "semantic_class": "speed_bump"},
}

MERGEABLE_LONG_CLASSES = {"lane_divider", "divider", "road_edge_marking", "boundary"}

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
    parser.add_argument(
        "--merge-mode",
        choices=("spatial_graph", "geometry", "station_lateral"),
        default=str(thin_cfg.get("merge_mode", "spatial_graph")),
        help="spatial_graph = tao net mong bang luoi hinh hoc; geometry = chi dedup representative; station_lateral = cach cu.",
    )
    parser.add_argument("--min-length-m", type=float, default=float(thin_cfg.get("min_length_m", 0.40)))
    parser.add_argument("--merge-chamfer-m", type=float, default=float(thin_cfg.get("merge_chamfer_m", 0.45)))
    parser.add_argument("--merge-center-m", type=float, default=float(thin_cfg.get("merge_center_m", 2.0)))
    parser.add_argument("--merge-heading-deg", type=float, default=float(thin_cfg.get("merge_heading_deg", 25.0)))
    parser.add_argument("--thin-grid-m", type=float, default=float(thin_cfg.get("thin_grid_m", 0.75)))
    parser.add_argument("--min-cell-observations", type=int, default=int(thin_cfg.get("min_cell_observations", 3)))
    parser.add_argument("--min-edge-observations", type=int, default=int(thin_cfg.get("min_edge_observations", 2)))
    parser.add_argument("--min-graph-points", type=int, default=int(thin_cfg.get("min_graph_points", 4)))
    parser.add_argument("--graph-smooth-window", type=int, default=int(thin_cfg.get("graph_smooth_window", 3)))
    parser.add_argument("--stop-cluster-m", type=float, default=float(thin_cfg.get("stop_cluster_m", 3.0)))
    parser.add_argument("--stop-heading-deg", type=float, default=float(thin_cfg.get("stop_heading_deg", 45.0)))
    parser.add_argument("--min-stop-cluster-size", type=int, default=int(thin_cfg.get("min_stop_cluster_size", 4)))
    parser.add_argument("--stop-min-length-m", type=float, default=float(thin_cfg.get("stop_min_length_m", 1.0)))
    parser.add_argument("--stop-max-length-m", type=float, default=float(thin_cfg.get("stop_max_length_m", 12.0)))
    parser.add_argument("--stop-line-points", type=int, default=int(thin_cfg.get("stop_line_points", 5)))
    parser.add_argument(
        "--debug-tags",
        action="store_true",
        default=bool(thin_cfg.get("debug_tags", False)),
        help="Them source/model_class/score/sample_token de debug. Mac dinh tat de OSM giong bang quy chuan.",
    )
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


def normalize_class_name(class_name: str) -> str:
    if class_name == "divider":
        return "lane_divider"
    return class_name


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def compact_polyline(points: np.ndarray, max_points: int = 32) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).round().astype(int)
    return points[idx]


def mean_bidirectional_nearest(a: np.ndarray, b: np.ndarray) -> float:
    a = compact_polyline(np.asarray(a, dtype=np.float64))
    b = compact_polyline(np.asarray(b, dtype=np.float64))
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float((d.min(axis=1).mean() + d.min(axis=0).mean()) * 0.5)


def undirected_heading(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    vec = pts[-1] - pts[0]
    if np.linalg.norm(vec) < 1e-6:
        vec = pts[min(len(pts) - 1, 1)] - pts[0]
    return float(math.atan2(vec[1], vec[0]))


def heading_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    da = undirected_heading(a)
    db = undirected_heading(b)
    diff = abs((da - db + math.pi) % (2.0 * math.pi) - math.pi)
    diff = min(diff, abs(math.pi - diff))
    return float(math.degrees(diff))


def center_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.mean(a, axis=0) - np.mean(b, axis=0)))


def line_is_duplicate(candidate: dict, representative: dict, args: argparse.Namespace) -> bool:
    pts_a = candidate["points"]
    pts_b = representative["points"]
    if center_distance(pts_a, pts_b) > args.merge_center_m:
        return False
    if mean_bidirectional_nearest(pts_a, pts_b) > args.merge_chamfer_m:
        return False
    if candidate["class_name"] in MERGEABLE_LONG_CLASSES:
        return heading_delta_deg(pts_a, pts_b) <= args.merge_heading_deg
    return True


def representative_score(line: dict) -> float:
    return float(line.get("score", 0.0)) + 0.01 * min(polyline_length(line["points"]), 20.0)


def collect_lines(predictions: list[dict], infos: dict[str, dict], score_thresh: float, min_length_m: float) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for sample_idx, sample in enumerate(predictions):
        token = sample.get("token")
        if token not in infos:
            continue
        T = infos[token]["_T_map_lidar"]
        for pred_idx, rec in enumerate(sample.get("pred", []) or []):
            score = float(rec.get("score", 0.0))
            if score < score_thresh:
                continue
            cls = normalize_class_name(rec.get("class_name", str(rec.get("label", ""))))
            pts = np.asarray(rec.get("points", []), dtype=np.float64)
            if pts.ndim != 2 or len(pts) < 2:
                continue
            pts_h = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(len(pts)), np.ones(len(pts))])
            pts_map = (T @ pts_h.T).T[:, :2]
            length = polyline_length(pts_map)
            if length < min_length_m:
                continue
            groups.setdefault(cls, []).append({
                "class_name": cls,
                "score": score,
                "length": length,
                "sample_token": token,
                "sample_index": sample_idx,
                "pred_index": pred_idx,
                "points": pts_map,
            })
    return groups


def build_geometry_lines(groups: dict[str, list[dict]], args: argparse.Namespace) -> list[dict]:
    """Dedup theo hinh hoc, khong noi cac nhanh moi."""
    out: list[dict] = []
    cell_size = max(float(args.merge_center_m), 0.5)
    for cls, candidates in groups.items():
        representatives: list[dict] = []
        grid: dict[tuple[int, int], list[dict]] = {}
        for line in sorted(candidates, key=representative_score, reverse=True):
            center = np.mean(line["points"], axis=0)
            cx = int(math.floor(float(center[0]) / cell_size))
            cy = int(math.floor(float(center[1]) / cell_size))
            nearby = []
            for gx in range(cx - 1, cx + 2):
                for gy in range(cy - 1, cy + 2):
                    nearby.extend(grid.get((gx, gy), []))
            if any(line_is_duplicate(line, rep, args) for rep in nearby):
                continue
            line["center"] = center
            representatives.append(line)
            grid.setdefault((cx, cy), []).append(line)
        out.extend(representatives)
    out.sort(key=lambda item: (item["class_name"], -float(item.get("score", 0.0))))
    return out


def stop_line_is_same_cluster(line: dict, cluster: list[dict], args: argparse.Namespace) -> bool:
    center = np.mean(line["points"], axis=0)
    cluster_center = np.mean([np.mean(item["points"], axis=0) for item in cluster], axis=0)
    if float(np.linalg.norm(center - cluster_center)) > args.stop_cluster_m:
        return False
    ref = max(cluster, key=representative_score)
    return heading_delta_deg(line["points"], ref["points"]) <= args.stop_heading_deg


def fit_stop_line_cluster(cluster: list[dict], args: argparse.Namespace) -> dict | None:
    weighted_points = []
    weights = []
    for line in cluster:
        score = max(float(line.get("score", 1.0)), 1e-3)
        for point in line["points"]:
            weighted_points.append(np.asarray(point, dtype=np.float64))
            weights.append(score)
    pts = np.asarray(weighted_points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if len(pts) < 2:
        return None
    mean = (pts * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-9)
    centered = pts - mean
    cov = (centered * w[:, None]).T @ centered / max(float(w.sum()), 1e-9)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, int(np.argmax(eigvals))]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
    proj = centered @ direction
    lo, hi = np.percentile(proj, [5.0, 95.0])
    length = float(hi - lo)
    if length < args.stop_min_length_m:
        return None
    if length > args.stop_max_length_m:
        mid = float(np.median(proj))
        half = args.stop_max_length_m * 0.5
        lo, hi = mid - half, mid + half
    num_points = max(int(args.stop_line_points), 2)
    line_pts = np.asarray([mean + direction * t for t in np.linspace(lo, hi, num_points)], dtype=np.float64)
    return {
        "class_name": "stop_line",
        "score": float(np.mean([float(item.get("score", 0.0)) for item in cluster])),
        "points": line_pts,
        "source_mode": "stop_line_cluster",
    }


def build_stop_line_lines(candidates: list[dict], args: argparse.Namespace) -> list[dict]:
    clusters: list[list[dict]] = []
    for line in sorted(candidates, key=representative_score, reverse=True):
        for cluster in clusters:
            if stop_line_is_same_cluster(line, cluster, args):
                cluster.append(line)
                break
        else:
            clusters.append([line])

    out = []
    for cluster in clusters:
        if len(cluster) < args.min_stop_cluster_size:
            continue
        fitted = fit_stop_line_cluster(cluster, args)
        if fitted is not None:
            out.append(fitted)
    return out


def grid_key(point: np.ndarray, grid_m: float) -> tuple[int, int]:
    return (int(math.floor(float(point[0]) / grid_m)), int(math.floor(float(point[1]) / grid_m)))


def canonical_edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def extract_graph_paths(adjacency: dict[tuple[int, int], set[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    """Lay path dai nhat trong moi connected component cua graph cell.

    Jitter cua prediction thuong lam mot vach thanh graph co nhieu node bac cao.
    Neu cat path tai moi node bac cao, output se thanh rat nhieu doan ngan.
    Lay graph diameter giu lai than chinh dai nhat cua moi bo net.
    """
    paths: list[list[tuple[int, int]]] = []
    unseen = set(adjacency)

    def component_from(start: tuple[int, int]) -> set[tuple[int, int]]:
        stack = [start]
        comp = set()
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            stack.extend(adjacency.get(node, ()) - comp)
        return comp

    def farthest_with_parent(start: tuple[int, int], comp: set[tuple[int, int]]):
        queue = [start]
        parent = {start: None}
        dist = {start: 0}
        for node in queue:
            for nxt in adjacency.get(node, ()):
                if nxt not in comp or nxt in parent:
                    continue
                parent[nxt] = node
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
        farthest = max(dist, key=lambda key: dist[key])
        return farthest, parent

    while unseen:
        seed = next(iter(unseen))
        comp = component_from(seed)
        unseen -= comp
        if len(comp) < 2:
            continue
        a, _ = farthest_with_parent(seed, comp)
        b, parent = farthest_with_parent(a, comp)
        path = []
        cur = b
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        paths.append(path)
    return paths


def build_spatial_graph_lines(groups: dict[str, list[dict]], args: argparse.Namespace) -> list[dict]:
    """Bien bo prediction day dac thanh graph mong theo tung class.

    Cell graph chi them edge tu cac segment model da du doan, vi vay khong tao
    duong noi moi giua hai nhanh doc lap trong giao lo.
    """
    out: list[dict] = []
    short_groups = {cls: rows for cls, rows in groups.items() if cls not in MERGEABLE_LONG_CLASSES and cls != "stop_line"}
    out.extend(build_geometry_lines(short_groups, args))
    out.extend(build_stop_line_lines(groups.get("stop_line", []), args))

    grid_m = max(float(args.thin_grid_m), 0.1)
    for cls, candidates in groups.items():
        if cls not in MERGEABLE_LONG_CLASSES:
            continue
        cells: dict[tuple[int, int], dict] = {}
        edge_counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        for line in candidates:
            score = float(line.get("score", 1.0))
            raw_keys = []
            for point in line["points"]:
                key = grid_key(point, grid_m)
                item = cells.setdefault(key, {"sum": np.zeros(2, dtype=np.float64), "weight": 0.0, "count": 0, "score_sum": 0.0})
                weight = max(score, 1e-3)
                item["sum"] += np.asarray(point, dtype=np.float64) * weight
                item["weight"] += weight
                item["count"] += 1
                item["score_sum"] += score
                if not raw_keys or raw_keys[-1] != key:
                    raw_keys.append(key)
            for a, b in zip(raw_keys[:-1], raw_keys[1:]):
                if a == b:
                    continue
                edge = canonical_edge(a, b)
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

        active = {key for key, item in cells.items() if item["count"] >= args.min_cell_observations}
        adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {key: set() for key in active}
        for (a, b), count in edge_counts.items():
            if count < args.min_edge_observations or a not in active or b not in active:
                continue
            adjacency[a].add(b)
            adjacency[b].add(a)
        adjacency = {key: neighbors for key, neighbors in adjacency.items() if neighbors}

        for path_keys in extract_graph_paths(adjacency):
            if len(path_keys) < args.min_graph_points:
                continue
            pts = []
            scores = []
            for key in path_keys:
                item = cells[key]
                pts.append(item["sum"] / max(item["weight"], 1e-9))
                scores.append(item["score_sum"] / max(item["count"], 1))
            pts_arr = moving_average(np.asarray(pts, dtype=np.float64), args.graph_smooth_window)
            if len(pts_arr) < args.min_graph_points or polyline_length(pts_arr) < args.min_length_m:
                continue
            out.append({
                "class_name": cls,
                "score": float(np.mean(scores)) if scores else 1.0,
                "points": pts_arr,
                "source_mode": "spatial_graph",
            })
    out.sort(key=lambda item: (item["class_name"], -float(item.get("score", 0.0))))
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


def write_osm(lines: list[dict], out_path: Path, create_lanelet_relations: bool, debug_tags: bool = False) -> None:
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
        cls = normalize_class_name(line["class_name"])
        for key, value in CLASS_TO_OSM_TAGS.get(cls, {}).items():
            add_tag(way, key, value)
        if debug_tags:
            add_tag(way, "source", str(line.get("source_mode", "maptr_prediction_merged")))
            add_tag(way, "model_class", cls)
            add_tag(way, "score", f"{float(line.get('score', 0.0)):.4f}")
            if line.get("sample_token"):
                add_tag(way, "sample_token", str(line["sample_token"]))
            if line.get("pred_index") is not None:
                add_tag(way, "pred_index", str(line["pred_index"]))

        way_infos.append({
            "id": way_id,
            "class_name": cls,
            "lat_center": float(sum(line.get("lat_range", [0.0, 0.0])) * 0.5),
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
    if args.merge_mode == "station_lateral":
        groups = collect_points(predictions, infos, stations, args.score_thresh)
        thin_lines = build_thin_lines(groups, args)
        class_summary = ", ".join(f"{k}={len(v)}pts" for k, v in groups.items())
    else:
        groups = collect_lines(predictions, infos, args.score_thresh, args.min_length_m)
        if args.merge_mode == "spatial_graph":
            thin_lines = build_spatial_graph_lines(groups, args)
        else:
            thin_lines = build_geometry_lines(groups, args)
        class_summary = ", ".join(f"{k}={len(v)}lines" for k, v in groups.items())
    write_osm(thin_lines, args.out_osm.expanduser().resolve(), args.create_lanelet_relations, args.debug_tags)
    print("[DONE]")
    print(f"Predictions : {args.predictions}")
    print(f"Infos       : {args.infos}")
    print(f"Output OSM  : {args.out_osm}")
    print(f"Mode        : {args.merge_mode}")
    print(f"Classes     : {class_summary}")
    print(f"Thin lines  : {len(thin_lines)}")
    print(f"Lanelet rel : {'on' if args.create_lanelet_relations else 'off'}")
    print(f"Debug tags  : {'on' if args.debug_tags else 'off'}")
    for line in thin_lines[:50]:
        lat_range = line.get("lat_range")
        if lat_range:
            extra = f" lat={lat_range[0]:.2f}..{lat_range[1]:.2f}"
        else:
            extra = f" score={float(line.get('score', 0.0)):.3f}"
        print(f"  {line['class_name']:18s} points={len(line['points']):5d}{extra}")
    if len(thin_lines) > 50:
        print(f"  ... {len(thin_lines) - 50} more lines")


if __name__ == "__main__":
    main()
