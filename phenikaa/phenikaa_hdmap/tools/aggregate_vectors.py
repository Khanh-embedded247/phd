"""
B3 Step 2/3 - aggregate local MapTR vectors with Phenikaa Pose and export OSM.

Input is the lanes_local_manifest.json written by infer_phenikaa.py.
Pose files are 3x4 transforms from local ego/sensor frame to Phenikaa map frame.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from phenikaa_hdmap.utils.pose import load_pose_3x4, pose_to_4x4

try:
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge, unary_union
except ImportError:  # pragma: no cover - fallback is exercised when shapely is absent.
    LineString = None
    MultiLineString = None
    linemerge = None
    unary_union = None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _transform_points(points_xy: list[list[float]], pose_path: Path) -> list[list[float]]:
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"Expected [N,2+] points, got {pts.shape}")

    pts_h = np.ones((pts.shape[0], 4), dtype=np.float64)
    pts_h[:, :2] = pts[:, :2]
    pts_h[:, 2] = pts[:, 2] if pts.shape[1] > 2 else 0.0

    T = pose_to_4x4(load_pose_3x4(pose_path))
    out = (T @ pts_h.T).T
    return out[:, :2].tolist()


def _distance(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dedupe_without_shapely(vectors: list[dict[str, Any]], merge_dist: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for vec in sorted(vectors, key=lambda item: item.get("score", 0.0), reverse=True):
        pts = vec["points_global"]
        midpoint = pts[len(pts) // 2]
        duplicate = False
        for other in kept:
            other_pts = other["points_global"]
            if vec["class_name"] == other["class_name"] and _distance(midpoint, other_pts[len(other_pts) // 2]) < merge_dist:
                duplicate = True
                break
        if not duplicate:
            kept.append(vec)
    return kept


def _explode_lines(geom: Any) -> list[Any]:
    if geom.is_empty:
        return []
    if LineString is not None and isinstance(geom, LineString):
        return [geom]
    if MultiLineString is not None and isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        lines = []
        for part in geom.geoms:
            lines.extend(_explode_lines(part))
        return lines
    return []


def _merge_with_shapely(vectors: list[dict[str, Any]], merge_dist: float) -> list[dict[str, Any]]:
    # Do not unary_union raw MapTR predictions here: intersecting polylines are
    # split into many fragments, which makes OSM previews look far worse than the
    # per-frame predictions. Keep this stage conservative until GT/association is
    # available.
    return _dedupe_without_shapely(vectors, merge_dist)


def aggregate(manifest_path: Path, score_thresh: float, merge_dist: float) -> list[dict[str, Any]]:
    manifest = _load_json(manifest_path)
    vectors: list[dict[str, Any]] = []
    for frame_ref in manifest.get("frames", []):
        frame_path = Path(frame_ref["path"])
        frame = _load_json(frame_path)
        pose_path_value = frame.get("pose_path") or frame_ref.get("pose_path")
        if not pose_path_value:
            raise ValueError(f"Missing pose_path for frame: {frame_path}")
        pose_path = Path(pose_path_value)
        if not pose_path.is_file():
            raise FileNotFoundError(f"Pose file not found: {pose_path}")

        for vec in frame.get("vectors", []):
            if float(vec.get("score", 1.0)) < score_thresh:
                continue
            points_global = _transform_points(vec["points"], pose_path)
            vectors.append(
                {
                    "frame_index": int(frame.get("frame_index", frame_ref.get("frame_index", -1))),
                    "token": str(frame.get("token", frame_ref.get("token", ""))),
                    "class_id": int(vec.get("class_id", -1)),
                    "class_name": str(vec.get("class_name", "unknown")),
                    "score": float(vec.get("score", 1.0)),
                    "points_global": points_global,
                }
            )
    return _merge_with_shapely(vectors, merge_dist)


def load_ego_poses(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = _load_json(manifest_path)
    poses = []
    for frame_ref in manifest.get("frames", []):
        pose_path_value = frame_ref.get("pose_path")
        if not pose_path_value:
            continue
        pose_path = Path(pose_path_value)
        if not pose_path.is_file():
            continue
        pose = load_pose_3x4(pose_path)
        poses.append(
            {
                "frame_index": int(frame_ref.get("frame_index", len(poses))),
                "token": str(frame_ref.get("token", "")),
                "x": float(pose[0, 3]),
                "y": float(pose[1, 3]),
                "z": float(pose[2, 3]),
            }
        )
    return poses


def write_geojson(vectors: list[dict[str, Any]], path: Path) -> None:
    features = []
    for idx, vec in enumerate(vectors):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": idx,
                    "class_id": vec.get("class_id", -1),
                    "class_name": vec.get("class_name", "unknown"),
                    "score": vec.get("score", 1.0),
                    "source_frame_count": vec.get("source_frame_count", 1),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": vec["points_global"],
                },
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")


def write_ego_poses_geojson(poses: list[dict[str, Any]], path: Path) -> None:
    features = []
    for pose in poses:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "frame_index": pose["frame_index"],
                    "token": pose["token"],
                    "kind": "ego_pose",
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [pose["x"], pose["y"], pose["z"]],
                },
            }
        )
    if len(poses) >= 2:
        features.append(
            {
                "type": "Feature",
                "properties": {"kind": "ego_trajectory"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[pose["x"], pose["y"], pose["z"]] for pose in poses],
                },
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")


def _fake_lat_lon(x: float, y: float, origin_x: float, origin_y: float) -> tuple[str, str]:
    # Lanelet2 tooling can reproject local maps later; keep true x/y in tags.
    lat = (y - origin_y) * 1e-7
    lon = (x - origin_x) * 1e-7
    return f"{lat:.10f}", f"{lon:.10f}"


def write_osm(vectors: list[dict[str, Any]], path: Path) -> None:
    root = ET.Element("osm", version="0.6", generator="phenikaa_hdmap_b3")
    all_points = [pt for vec in vectors for pt in vec["points_global"]]
    origin_x = min((pt[0] for pt in all_points), default=0.0)
    origin_y = min((pt[1] for pt in all_points), default=0.0)

    node_id = -1
    way_id = -1
    for vec in vectors:
        refs = []
        for x, y in vec["points_global"]:
            lat, lon = _fake_lat_lon(float(x), float(y), origin_x, origin_y)
            node = ET.SubElement(root, "node", id=str(node_id), lat=lat, lon=lon)
            ET.SubElement(node, "tag", k="local_x", v=f"{float(x):.6f}")
            ET.SubElement(node, "tag", k="local_y", v=f"{float(y):.6f}")
            refs.append(node_id)
            node_id -= 1

        way = ET.SubElement(root, "way", id=str(way_id))
        for ref in refs:
            ET.SubElement(way, "nd", ref=str(ref))
        ET.SubElement(way, "tag", k="type", v="line_thin")
        ET.SubElement(way, "tag", k="subtype", v=str(vec.get("class_name", "unknown")))
        ET.SubElement(way, "tag", k="phenikaa:class_id", v=str(vec.get("class_id", -1)))
        ET.SubElement(way, "tag", k="phenikaa:score", v=f"{float(vec.get('score', 1.0)):.4f}")
        way_id -= 1

    tree = ET.ElementTree(root)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phenikaa_vis/phenikaa_b3/global_map"))
    parser.add_argument("--score-thresh", type=float, default=0.35)
    parser.add_argument("--merge-dist", type=float, default=1.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vectors = aggregate(args.manifest, score_thresh=args.score_thresh, merge_dist=args.merge_dist)
    ego_poses = load_ego_poses(args.manifest)

    json_path = args.out_dir / "global_vectors.json"
    geojson_path = args.out_dir / "global_vectors.geojson"
    ego_path = args.out_dir / "ego_poses.geojson"
    osm_path = args.out_dir / "global_vectors.osm"
    json_path.write_text(json.dumps({"vectors": vectors}, indent=2), encoding="utf-8")
    write_geojson(vectors, geojson_path)
    write_ego_poses_geojson(ego_poses, ego_path)
    write_osm(vectors, osm_path)

    print(f"Step 2/3 done: {len(vectors)} global vectors")
    print(f"JSON   : {json_path}")
    print(f"GeoJSON: {geojson_path}")
    print(f"Ego    : {ego_path}")
    print(f"OSM   : {osm_path}")


if __name__ == "__main__":
    main()
