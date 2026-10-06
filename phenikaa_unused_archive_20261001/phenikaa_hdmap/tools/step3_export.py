"""
B3 Step 3 — Export aggregated map as lanelet2 OSM format + create visualizations.

Input:
  - outputs/aggregated_maps/phenikaa_lanes_global.json (from Step 2)

Output:
  - outputs/aggregated_maps/phenikaa_lanelet.osm (OpenStreetMap format)
  - outputs/phenikaa_vis/final/map_visualization.json (for further viz)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ============================================================================
# Paths (hardcoded in file)
# ============================================================================
BASE = Path(__file__).resolve().parent.parent.parent
STEP2_OUT = BASE / "outputs" / "aggregated_maps" / "phenikaa_lanes_global.json"
OUT_DIR = BASE / "outputs" / "aggregated_maps"
VIS_DIR = BASE / "outputs" / "phenikaa_vis" / "final"
OUT_OSM = OUT_DIR / "phenikaa_lanelet.osm"
OUT_VIS = VIS_DIR / "map_visualization.json"

# ============================================================================
# Constants
# ============================================================================
LANELET_VERSION = "1.0"
LANELET_BASE_ID = 1000  # start IDs from here to avoid conflicts with real OSM


# ============================================================================
# Lanelet2 OSM Export
# ============================================================================
def create_osm_node(node_id: int, lat: float, lon: float) -> str:
    """Create OSM node XML element (simplified — using x, y as lat/lon)."""
    # Note: Phenikaa coordinates are in local frame, not WGS84
    # For lanelet2, we just map local x,y to lat,lon for demonstration
    return f'  <node id="{node_id}" lat="{lon:.7f}" lon="{lat:.7f}" />'


def create_osm_way(way_id: int, node_ids: list[int], tags: dict) -> str:
    """Create OSM way XML element."""
    xml = f'  <way id="{way_id}">\n'
    for nid in node_ids:
        xml += f'    <nd ref="{nid}" />\n'
    for key, value in tags.items():
        xml += f'    <tag k="{key}" v="{value}" />\n'
    xml += f"  </way>\n"
    return xml


def export_to_osm(lanes: list[dict]) -> str:
    """Convert lane vectors to OSM XML format."""
    node_counter = LANELET_BASE_ID
    way_counter = LANELET_BASE_ID + 10000
    nodes_xml = ""
    ways_xml = ""
    node_id_map = {}  # (x, y) -> node_id

    for lane_idx, lane in enumerate(lanes):
        points = lane.get("points", [])
        if len(points) < 2:
            continue

        class_name = lane.get("class_name", "divider")
        score = lane.get("score", 1.0)

        # Create nodes for this lane
        node_ids = []
        for pt_idx, point in enumerate(points):
            x, y = point[0], point[1]
            key = (round(x, 3), round(y, 3))

            if key not in node_id_map:
                nid = node_counter
                node_counter += 1
                node_id_map[key] = nid
                nodes_xml += create_osm_node(nid, x, y) + "\n"

            node_ids.append(node_id_map[key])

        # Create way for this lane
        wid = way_counter
        way_counter += 1
        tags = {
            "type": "lanelet",
            "subtype": class_name,
            "confidence": f"{score:.2f}",
        }
        ways_xml += create_osm_way(wid, node_ids, tags)

    # Build full OSM document
    osm = f"""<?xml version="1.0" encoding="UTF-8"?>
<osm version="{LANELET_VERSION}" generator="Phenikaa-B3">
{nodes_xml}
{ways_xml}</osm>
"""
    return osm


# ============================================================================
# Visualization JSON
# ============================================================================
def create_viz_json(lanes: list[dict]) -> dict:
    """Create a visualization-friendly JSON."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "class": lane.get("class_name", "unknown"),
                    "score": lane.get("score", 1.0),
                    "frame_token": lane.get("token", "unknown"),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[float(p[0]), float(p[1])] for p in lane.get("points", [])],
                },
            }
            for lane in lanes
        ],
    }


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    print("=" * 80)
    print("B3 STEP 3: Export Lanelet + Visualizations")
    print("=" * 80)

    # Check input
    if not STEP2_OUT.exists():
        print(f"ERROR: Step 2 output not found at {STEP2_OUT}")
        print(f"  Please run step2_aggregate.py first")
        return

    # Load aggregated map
    print(f"\nLoading aggregated map...")
    with open(STEP2_OUT) as f:
        aggregate_data = json.load(f)

    lanes = aggregate_data.get("lanes", [])
    print(f"  Loaded {len(lanes)} lanes")

    # Export to OSM
    print(f"\nExporting to Lanelet2 OSM format...")
    osm_content = export_to_osm(lanes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_OSM, "w") as f:
        f.write(osm_content)
    print(f"  ✓ OSM export: {OUT_OSM}")

    # Create visualization JSON
    print(f"\nCreating visualization JSON...")
    viz_json = create_viz_json(lanes)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_VIS, "w") as f:
        json.dump(viz_json, f, indent=2)
    print(f"  ✓ Visualization JSON: {OUT_VIS}")

    # Summary
    print(f"\n{'='*80}")
    print(f"✓ B3 Complete!")
    print(f"{'='*80}")
    print(f"\nOutputs:")
    print(f"  1. OSM Lanelet: {OUT_OSM}")
    print(f"  2. Viz JSON: {OUT_VIS}")
    print(f"  3. Per-frame vectors: {BASE / 'outputs/phenikaa_vis/b3_step1_local_vectors'}")
    print(f"  4. Aggregated map: {STEP2_OUT}")
    print(f"\nYou can now:")
    print(f"  - Open {OUT_OSM} in JOSM or other OSM editor")
    print(f"  - Visualize {OUT_VIS} with GeoJSON viewers")
    print(f"  - Use aggregated lane data for further analysis")


if __name__ == "__main__":
    main()
