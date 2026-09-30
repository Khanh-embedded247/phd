#!/usr/bin/env python3
"""Check GT OSM semantic tags against GT_OSM_DRAWING_GUIDE.md."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from pipeline_config import load_pipeline_config  # noqa: E402


VISIBLE_TO_MODEL = {
    "lane_divider": "lane_divider",
    "road_edge_marking": "road_edge_marking",
    "stop_line": "stop_line",
    "ped_crossing": "ped_crossing",
    "road_boundary": "boundary",
    "speed_bump": "speed_bump",
}

TOPOLOGY_CLASSES = {
    "virtual_lane_divider",
    "virtual_boundary",
    "virtual_transition_boundary",
    "virtual_road_boundary",
    "centerline",
    "virtual_lanelet",
    "drivable_lane",
    "sidewalk",
}

LEGACY_TOPOLOGY_ALIASES = {
    "virtual_driving_lane": "virtual_lanelet",
    "transition_lane": "virtual_lanelet",
    "junction_connector": "virtual_lanelet",
}

FALLBACK_WAY_TYPE_TO_MODEL = {
    "line_thin": "lane_divider",
    "road_border": "boundary",
    "stop_line": "stop_line",
    "speed_bump": "speed_bump",
}

KNOWN_SEMANTIC = set(VISIBLE_TO_MODEL) | TOPOLOGY_CLASSES | set(LEGACY_TOPOLOGY_ALIASES)


def parse_args() -> argparse.Namespace:
    cfg = load_pipeline_config()
    vars_cfg = cfg["_vars"]
    paths = cfg.get("_paths", {})
    gt_cfg = cfg.get("gt_qa", {})
    default_osm = paths.get("gt_osm") or (Path(vars_cfg["DATA_ROOT"]) / f"{vars_cfg['SCENARIO']}.osm")
    parser = argparse.ArgumentParser(description="Check GT OSM semantic_class/type/subtype consistency.")
    parser.add_argument("--osm", type=Path, default=Path(default_osm))
    parser.add_argument("--strict-missing-semantic", action="store_true", default=bool(gt_cfg.get("strict_missing_semantic", False)))
    parser.add_argument("--out-report", type=Path, default=Path(vars_cfg["FINAL_DIR"]) / "qa" / "gt_osm_semantic_report.json")
    return parser.parse_args()


def elem_tags(elem: ET.Element) -> dict[str, str]:
    return {tag.attrib.get("k", "").strip(): tag.attrib.get("v", "").strip() for tag in elem.findall("tag")}


def way_is_closed(refs: list[str]) -> bool:
    return len(refs) >= 4 and refs[0] == refs[-1]


def main() -> None:
    args = parse_args()
    osm_path = args.osm.expanduser().resolve()
    if not osm_path.exists():
        raise FileNotFoundError(osm_path)

    root = ET.parse(osm_path).getroot()
    nodes = set()
    way_refs: dict[str, list[str]] = {}
    way_tags: dict[str, dict[str, str]] = {}
    relation_tags: dict[str, dict[str, str]] = {}
    relation_members: dict[str, list[dict[str, str]]] = {}

    for elem in root:
        if elem.tag == "node":
            nodes.add(elem.attrib.get("id", ""))
        elif elem.tag == "way":
            way_id = elem.attrib.get("id", "")
            way_refs[way_id] = [nd.attrib.get("ref", "") for nd in elem.findall("nd")]
            way_tags[way_id] = elem_tags(elem)
        elif elem.tag == "relation":
            rel_id = elem.attrib.get("id", "")
            relation_tags[rel_id] = elem_tags(elem)
            relation_members[rel_id] = [member.attrib for member in elem.findall("member")]

    model_counts = Counter()
    topology_counts = Counter()
    way_combo_counts = Counter()
    relation_combo_counts = Counter()
    missing_semantic: list[dict[str, Any]] = []
    unknown_semantic: list[dict[str, Any]] = []
    legacy_semantic: list[dict[str, Any]] = []
    bad_refs: list[dict[str, Any]] = []
    crosswalk_relations = 0
    speed_bump_polygons = 0

    for way_id, tags in way_tags.items():
        refs = way_refs[way_id]
        missing = [ref for ref in refs if ref not in nodes]
        if missing:
            bad_refs.append({"type": "way", "id": way_id, "missing_refs": missing[:20]})
        sem = tags.get("semantic_class", "")
        typ = tags.get("type", "")
        subtype = tags.get("subtype", "")
        way_combo_counts[(typ, subtype, sem)] += 1
        if sem:
            if sem in VISIBLE_TO_MODEL:
                model_counts[VISIBLE_TO_MODEL[sem]] += 1
                if sem == "speed_bump" and way_is_closed(refs):
                    speed_bump_polygons += 1
            elif sem in TOPOLOGY_CLASSES or typ == "virtual":
                topology_counts[sem or typ] += 1
            elif sem in LEGACY_TOPOLOGY_ALIASES:
                topology_counts[LEGACY_TOPOLOGY_ALIASES[sem]] += 1
                legacy_semantic.append({"type": "way", "id": way_id, "semantic_class": sem, "use_instead": LEGACY_TOPOLOGY_ALIASES[sem]})
            else:
                unknown_semantic.append({"type": "way", "id": way_id, "semantic_class": sem, "osm_type": typ, "subtype": subtype})
        else:
            if typ in FALLBACK_WAY_TYPE_TO_MODEL or typ == "virtual":
                missing_semantic.append({"type": "way", "id": way_id, "osm_type": typ, "subtype": subtype})
                if typ in FALLBACK_WAY_TYPE_TO_MODEL and not args.strict_missing_semantic:
                    model_counts[FALLBACK_WAY_TYPE_TO_MODEL[typ]] += 1
                elif typ == "virtual":
                    topology_counts["virtual_without_semantic"] += 1

    for rel_id, tags in relation_tags.items():
        sem = tags.get("semantic_class", "")
        typ = tags.get("type", "")
        subtype = tags.get("subtype", "")
        relation_combo_counts[(typ, subtype, sem)] += 1
        is_crosswalk = typ == "lanelet" and subtype == "crosswalk"
        if is_crosswalk or sem == "ped_crossing":
            crosswalk_relations += 1
            model_counts["ped_crossing"] += 1
            roles = {member.get("role", "") for member in relation_members.get(rel_id, [])}
            if "left" not in roles or "right" not in roles:
                unknown_semantic.append({"type": "relation", "id": rel_id, "problem": "crosswalk_missing_left_or_right", "roles": sorted(roles)})
            continue
        if sem in TOPOLOGY_CLASSES:
            topology_counts[sem] += 1
        elif sem in LEGACY_TOPOLOGY_ALIASES:
            topology_counts[LEGACY_TOPOLOGY_ALIASES[sem]] += 1
            legacy_semantic.append({"type": "relation", "id": rel_id, "semantic_class": sem, "use_instead": LEGACY_TOPOLOGY_ALIASES[sem]})
        elif sem and sem not in KNOWN_SEMANTIC:
            unknown_semantic.append({"type": "relation", "id": rel_id, "semantic_class": sem, "osm_type": typ, "subtype": subtype})

    report = {
        "osm": str(osm_path),
        "node_count": len(nodes),
        "way_count": len(way_tags),
        "relation_count": len(relation_tags),
        "model_class_counts": dict(sorted(model_counts.items())),
        "topology_class_counts": dict(sorted(topology_counts.items())),
        "crosswalk_relations": crosswalk_relations,
        "speed_bump_closed_ways": speed_bump_polygons,
        "missing_semantic_count": len(missing_semantic),
        "missing_semantic": missing_semantic[:300],
        "unknown_semantic_count": len(unknown_semantic),
        "unknown_semantic": unknown_semantic[:300],
        "legacy_semantic_count": len(legacy_semantic),
        "legacy_semantic": legacy_semantic[:300],
        "bad_ref_count": len(bad_refs),
        "bad_refs": bad_refs[:100],
        "way_combos": [
            {"count": count, "type": key[0], "subtype": key[1], "semantic_class": key[2]}
            for key, count in way_combo_counts.most_common()
        ],
        "relation_combos": [
            {"count": count, "type": key[0], "subtype": key[1], "semantic_class": key[2]}
            for key, count in relation_combo_counts.most_common()
        ],
    }

    out_report = args.out_report.expanduser().resolve()
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[GT OSM SEMANTIC QA]")
    print(f"OSM                 : {osm_path}")
    print(f"Report              : {out_report}")
    print(f"Model class counts  : {dict(sorted(model_counts.items()))}")
    print(f"Topology counts     : {dict(sorted(topology_counts.items()))}")
    print(f"Missing semantic    : {len(missing_semantic)}")
    print(f"Unknown semantic    : {len(unknown_semantic)}")
    print(f"Legacy semantic     : {len(legacy_semantic)}")
    print(f"Bad refs            : {len(bad_refs)}")
    if unknown_semantic or bad_refs or (args.strict_missing_semantic and missing_semantic):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
