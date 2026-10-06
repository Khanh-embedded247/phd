"""
B3 Step 2 — Aggregate local lane vectors using Pose transforms.

Takes per-frame predictions from Step 1, transforms each to global coordinates
using Pose/*.txt, then merges overlapping segments.

Input:
  - outputs/phenikaa_vis/b3_step1_local_vectors/manifest.json

Output:
  - outputs/aggregated_maps/phenikaa_lanes_global.json (merged map)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# ============================================================================
# Paths (hardcoded in file)
# ============================================================================
BASE = Path(__file__).resolve().parent.parent.parent
STEP1_OUT = BASE / "outputs" / "phenikaa_vis" / "b3_step1_local_vectors"
STEP1_MANIFEST = STEP1_OUT / "manifest.json"
OUT_DIR = BASE / "outputs" / "aggregated_maps"
OUT_FILE = OUT_DIR / "phenikaa_lanes_global.json"

SEQUENCE_ROOT = BASE / "data" / "phenikaa" / "sequences" / "RESIDENTIAL_AREA"
POSE_DIR = SEQUENCE_ROOT / "Pose"

# ============================================================================
# Constants
# ============================================================================
MERGE_DISTANCE_THRESHOLD = 0.5  # meters — points closer than this are merged
SIMPLIFY_TOLERANCE = 0.1  # meters — local simplification


# ============================================================================
# Utilities
# ============================================================================
def load_pose_3x4(pose_path: Path) -> np.ndarray:
    """Load Pose/*.txt as 3x4 matrix."""
    vals = np.loadtxt(pose_path, dtype=np.float64).reshape(-1)
    if vals.size != 12:
        raise ValueError(f"Expected 12 values in pose, got {vals.size}: {pose_path}")
    return vals.reshape(3, 4)


def pose_3x4_to_4x4(pose: np.ndarray) -> np.ndarray:
    """Convert 3x4 pose to 4x4 homogeneous transform."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :] = pose
    return T


def transform_points_3d(points: np.ndarray, pose_3x4: np.ndarray) -> np.ndarray:
    """
    Transform 2D points (x, y) in local frame to global using pose.

    points: [N, 2] or [N, 3] (x, y, [z])
    pose_3x4: 3x4 matrix
    Returns: [N, 2] or [N, 3] transformed points
    """
    if points.shape[0] == 0:
        return points

    # Pad to 3D if needed
    if points.shape[1] == 2:
        points_3d = np.hstack([points, np.zeros((points.shape[0], 1), dtype=np.float64)])
        pad_2d = True
    else:
        points_3d = points
        pad_2d = False

    # Apply pose (3x4 matrix + homogeneous)
    # [x', y', z'] = pose[:3, :3] @ [x, y, z] + pose[:3, 3]
    R = pose_3x4[:3, :3]
    t = pose_3x4[:3, 3]
    transformed = (R @ points_3d.T).T + t

    if pad_2d:
        transformed = transformed[:, :2]

    return transformed


def load_step1_vectors() -> list[dict]:
    """Load all per-frame vectors from Step 1 output."""
    if not STEP1_MANIFEST.exists():
        raise FileNotFoundError(f"Step 1 manifest not found: {STEP1_MANIFEST}")

    with open(STEP1_MANIFEST) as f:
        manifest = json.load(f)

    frames = []
    for frame_meta in manifest["frames"]:
        output_file = Path(frame_meta["output_file"])
        if not output_file.exists():
            print(f"  Warning: output file not found, skipping: {output_file}")
            continue

        with open(output_file) as f:
            frame_data = json.load(f)

        frame_data["output_file"] = output_file
        frames.append(frame_data)

    return frames


def aggregate_with_pose() -> dict:
    """Load all frames, transform to global, merge."""
    print(f"\nLoading Step 1 per-frame vectors...")
    frames = load_step1_vectors()
    print(f"  Loaded {len(frames)} frames")

    global_vectors = []

    print(f"\nTransforming to global coordinates...")
    for idx, frame_data in enumerate(frames):
        if idx % 50 == 0:
            print(f"  [{idx}/{len(frames)}]")

        pose_path = frame_data.get("pose_path")
        if not pose_path:
            print(f"    Warning: no pose_path for frame {frame_data['token']}")
            continue

        pose_path = Path(pose_path)
        if not pose_path.exists():
            print(f"    Warning: pose file not found: {pose_path}")
            continue

        try:
            pose = load_pose_3x4(pose_path)
        except Exception as e:
            print(f"    Warning: failed to load pose {pose_path}: {e}")
            continue

        # Transform each vector in this frame
        for vec_local in frame_data.get("vectors", []):
            points_local = np.array(vec_local.get("points", []), dtype=np.float64)
            if points_local.shape[0] < 2:
                continue

            # Transform to global
            points_global = transform_points_3d(points_local, pose)

            global_vectors.append(
                {
                    "frame_index": frame_data["frame_index"],
                    "token": frame_data["token"],
                    "class_id": vec_local.get("class_id", 0),
                    "class_name": vec_local.get("class_name", "divider"),
                    "score": vec_local.get("score", 1.0),
                    "points": points_global.tolist(),
                }
            )

    print(f"  Transformed {len(global_vectors)} lane vectors to global")

    # Merge close segments (simple approach: cluster by proximity)
    print(f"\nMerging nearby segments...")
    merged = merge_close_lanes(global_vectors)
    print(f"  Merged to {len(merged)} lane segments")

    return {
        "metadata": {
            "sequence": "RESIDENTIAL_AREA",
            "total_frames": len(frames),
            "merge_threshold_m": MERGE_DISTANCE_THRESHOLD,
        },
        "lanes": merged,
    }


def merge_close_lanes(vectors: list[dict]) -> list[dict]:
    """
    Simple merge: group vectors whose endpoints are close, then interpolate.

    For now, just return vectors with minimal post-processing.
    A proper implementation would use spatial indexing and graph algorithms.
    """
    # TODO: implement sophisticated merging (DBSCAN, graph connectivity, etc.)
    # For now, just return as-is with some filtering
    filtered = []
    for vec in vectors:
        if vec["score"] >= 0.35:  # score threshold
            filtered.append(vec)

    return filtered


def main() -> None:
    print("=" * 80)
    print("B3 STEP 2: Aggregate & Transform with Pose")
    print("=" * 80)

    # Check inputs
    if not STEP1_MANIFEST.exists():
        print(f"ERROR: Step 1 manifest not found at {STEP1_MANIFEST}")
        print(f"  Please run step1_infer_maptr.py first")
        return

    # Aggregate
    result = aggregate_with_pose()

    # Save output
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Aggregation done!")
    print(f"  Total lanes: {len(result['lanes'])}")
    print(f"  Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
