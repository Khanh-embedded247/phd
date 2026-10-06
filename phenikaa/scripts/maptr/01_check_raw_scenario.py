#!/usr/bin/env python3
"""Check raw Phenikaa scenario before building MapTR infos.

This step does not modify data. It verifies camera/LiDAR/trajectory presence
and timestamp alignment, then writes a compact JSON report.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

from pathlib import Path

PHENIKAA_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PHENIKAA_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from typing import Any

import cv2

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from phenikaa_maptr.common import load_lidar_trajectory, timestamp_from_name  # noqa: E402
from phenikaa_maptr.pipeline_config import load_pipeline_config  # noqa: E402


CAMERAS = [
    "CAM_P_F",
    "CAM_P_FL",
    "CAM_P_FR",
    "CAM_P_B",
    "CAM_P_L",
    "CAM_P_R",
    "CAM_P_LB",
    "CAM_P_RB",
    "CAM_F_F",
    "CAM_F_L",
    "CAM_F_R",
    "CAM_F_B",
]


def parse_args() -> argparse.Namespace:
    cfg = load_pipeline_config()
    vars_cfg = cfg["_vars"]
    raw_cfg = cfg.get("raw_qa", {})
    parser = argparse.ArgumentParser(description="Check raw camera/LiDAR/trajectory scenario data.")
    parser.add_argument("--scenario", default=vars_cfg.get("SCENARIO", "Normal"))
    parser.add_argument("--data-root", type=Path, default=Path(vars_cfg["DATA_ROOT"]))
    parser.add_argument("--reference-camera", default=raw_cfg.get("reference_camera", "CAM_P_F"))
    parser.add_argument("--max-camera-dt-sec", type=float, default=float(raw_cfg.get("max_camera_dt_sec", 0.025)))
    parser.add_argument("--max-image-checks", type=int, default=int(raw_cfg.get("max_image_checks", 24)))
    parser.add_argument("--out-report", type=Path, default=Path(vars_cfg["FINAL_DIR"]) / "qa" / "raw_scenario_report.json")
    return parser.parse_args()


def list_timestamped(directory: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not directory.exists():
        return []
    paths = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes]
    return sorted(paths, key=timestamp_from_name)


def nearest_delta(value: float, sorted_values: list[float]) -> float | None:
    if not sorted_values:
        return None
    import bisect

    idx = bisect.bisect_left(sorted_values, value)
    candidates = []
    if idx > 0:
        candidates.append(abs(sorted_values[idx - 1] - value))
    if idx < len(sorted_values):
        candidates.append(abs(sorted_values[idx] - value))
    return min(candidates) if candidates else None


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None}
    return {
        "min": float(min(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
    }


def sample_paths(paths: list[Path], max_checks: int) -> list[Path]:
    if max_checks <= 0 or len(paths) <= max_checks:
        return paths
    step = max(len(paths) // max_checks, 1)
    return paths[::step][:max_checks]


def image_report(paths: list[Path], max_checks: int) -> dict[str, Any]:
    checked = 0
    unreadable = []
    shapes: dict[str, int] = {}
    for path in sample_paths(paths, max_checks):
        checked += 1
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            unreadable.append(str(path))
            continue
        h, w = img.shape[:2]
        shapes[f"{w}x{h}"] = shapes.get(f"{w}x{h}", 0) + 1
    return {"checked": checked, "unreadable": unreadable, "shapes": shapes}


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    camera_root = data_root / "CAMERA"
    lidar_dir = data_root / "dump" / "frames" / "laz"
    traj_path = data_root / "dump" / "traj_lidar.txt"

    report: dict[str, Any] = {
        "scenario": args.scenario,
        "data_root": str(data_root),
        "camera_root": str(camera_root),
        "lidar_dir": str(lidar_dir),
        "trajectory": str(traj_path),
        "max_camera_dt_sec": args.max_camera_dt_sec,
        "cameras": {},
        "errors": [],
        "warnings": [],
    }

    camera_times: dict[str, list[float]] = {}
    for camera in CAMERAS:
        paths = list_timestamped(camera_root / camera, (".jpg", ".jpeg", ".png"))
        times = [timestamp_from_name(p) for p in paths]
        camera_times[camera] = times
        report["cameras"][camera] = {
            "dir": str(camera_root / camera),
            "count": len(paths),
            "first_ts": float(times[0]) if times else None,
            "last_ts": float(times[-1]) if times else None,
            "image_check": image_report(paths, args.max_image_checks),
        }
        if not paths:
            report["errors"].append(f"Missing images for {camera}")

    ref_times = camera_times.get(args.reference_camera, [])
    for camera, times in camera_times.items():
        if camera == args.reference_camera or not ref_times or not times:
            continue
        deltas = []
        bad = 0
        for ts in ref_times:
            dt = nearest_delta(ts, times)
            if dt is None:
                continue
            deltas.append(dt)
            if dt > args.max_camera_dt_sec:
                bad += 1
        report["cameras"][camera]["dt_to_reference_sec"] = summarize(deltas)
        report["cameras"][camera]["frames_over_max_camera_dt"] = bad
        if bad:
            report["warnings"].append(f"{camera}: {bad} frames exceed max_camera_dt_sec")

    lidar_paths = list_timestamped(lidar_dir, (".laz", ".las"))
    lidar_times = [timestamp_from_name(p) for p in lidar_paths]
    report["lidar"] = {
        "count": len(lidar_paths),
        "first_ts": float(lidar_times[0]) if lidar_times else None,
        "last_ts": float(lidar_times[-1]) if lidar_times else None,
    }
    if not lidar_paths:
        report["errors"].append("Missing LiDAR .laz/.las files")

    if traj_path.exists():
        poses = load_lidar_trajectory(traj_path)
        report["trajectory_info"] = {
            "count": len(poses),
            "first_ts": float(poses[0].timestamp) if poses else None,
            "last_ts": float(poses[-1].timestamp) if poses else None,
        }
        if not poses:
            report["errors"].append("Trajectory file has no poses")
    else:
        report["errors"].append("Missing trajectory file")

    args.out_report.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.out_report.expanduser().resolve().open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[RAW SCENARIO QA]")
    print(f"Scenario : {args.scenario}")
    print(f"Data root: {data_root}")
    print(f"Report   : {args.out_report}")
    print(f"Errors   : {len(report['errors'])}")
    print(f"Warnings : {len(report['warnings'])}")
    for msg in report["errors"][:20]:
        print(f"ERROR: {msg}")
    for msg in report["warnings"][:20]:
        print(f"WARN : {msg}")
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
