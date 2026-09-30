#!/usr/bin/env python3
"""Check trajectory quality used for MapTR GT slicing and LiDAR sync."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import load_lidar_trajectory  # noqa: E402
from pipeline_config import load_pipeline_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    cfg = load_pipeline_config()
    vars_cfg = cfg["_vars"]
    pose_cfg = cfg.get("pose_qa", {})
    data_root = Path(vars_cfg["DATA_ROOT"])
    default_out = Path(vars_cfg["FINAL_DIR"]) / "qa"
    parser = argparse.ArgumentParser(description="Check lidar trajectory pose quality.")
    parser.add_argument("--traj", type=Path, default=data_root / "dump" / "traj_lidar.txt")
    parser.add_argument("--max-speed-mps", type=float, default=float(pose_cfg.get("max_speed_mps", 30.0)))
    parser.add_argument("--max-yaw-rate-dps", type=float, default=float(pose_cfg.get("max_yaw_rate_dps", 120.0)))
    parser.add_argument("--max-dt-sec", type=float, default=float(pose_cfg.get("max_dt_sec", 0.20)))
    parser.add_argument("--out-json", type=Path, default=default_out / "pose_quality_report.json")
    parser.add_argument("--out-csv", type=Path, default=default_out / "pose_quality_steps.csv")
    return parser.parse_args()


def yaw_from_rot(R: np.ndarray) -> float:
    return float(math.atan2(R[1, 0], R[0, 0]))


def wrap_angle(rad: float) -> float:
    return float((rad + math.pi) % (2.0 * math.pi) - math.pi)


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None}
    return {
        "min": float(min(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
    }


def main() -> None:
    args = parse_args()
    traj = args.traj.expanduser().resolve()
    if not traj.exists():
        raise FileNotFoundError(traj)

    poses = load_lidar_trajectory(traj)
    rows: list[dict[str, Any]] = []
    dts: list[float] = []
    distances: list[float] = []
    speeds: list[float] = []
    yaw_rates: list[float] = []
    warnings: list[str] = []

    for idx in range(1, len(poses)):
        prev = poses[idx - 1]
        cur = poses[idx]
        dt = float(cur.timestamp - prev.timestamp)
        p0 = prev.T_map_lidar[:3, 3]
        p1 = cur.T_map_lidar[:3, 3]
        dist = float(np.linalg.norm(p1[:2] - p0[:2]))
        speed = dist / dt if dt > 1e-9 else float("inf")
        yaw0 = yaw_from_rot(prev.T_map_lidar[:3, :3])
        yaw1 = yaw_from_rot(cur.T_map_lidar[:3, :3])
        yaw_delta_deg = math.degrees(wrap_angle(yaw1 - yaw0))
        yaw_rate_dps = abs(yaw_delta_deg) / dt if dt > 1e-9 else float("inf")
        flag = ""
        if dt <= 0:
            flag = "non_positive_dt"
        elif dt > args.max_dt_sec:
            flag = "large_dt"
        elif speed > args.max_speed_mps:
            flag = "large_speed"
        elif yaw_rate_dps > args.max_yaw_rate_dps:
            flag = "large_yaw_rate"
        if flag:
            warnings.append(f"idx={idx} {flag} dt={dt:.6f} speed={speed:.3f} yaw_rate={yaw_rate_dps:.3f}")
        dts.append(dt)
        distances.append(dist)
        speeds.append(speed)
        yaw_rates.append(yaw_rate_dps)
        rows.append({
            "idx": idx,
            "timestamp_prev": prev.timestamp,
            "timestamp": cur.timestamp,
            "dt_sec": dt,
            "distance_xy_m": dist,
            "speed_mps": speed,
            "yaw_delta_deg": yaw_delta_deg,
            "yaw_rate_dps": yaw_rate_dps,
            "flag": flag,
        })

    report = {
        "trajectory": str(traj),
        "pose_count": len(poses),
        "first_ts": float(poses[0].timestamp) if poses else None,
        "last_ts": float(poses[-1].timestamp) if poses else None,
        "thresholds": {
            "max_speed_mps": args.max_speed_mps,
            "max_yaw_rate_dps": args.max_yaw_rate_dps,
            "max_dt_sec": args.max_dt_sec,
        },
        "dt_sec": summarize(dts),
        "distance_xy_m": summarize(distances),
        "speed_mps": summarize(speeds),
        "yaw_rate_dps": summarize(yaw_rates),
        "warning_count": len(warnings),
        "warnings": warnings[:200],
    }

    args.out_json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.expanduser().resolve().open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with args.out_csv.expanduser().resolve().open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["idx", "timestamp_prev", "timestamp", "dt_sec", "distance_xy_m", "speed_mps", "yaw_delta_deg", "yaw_rate_dps", "flag"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[POSE QA]")
    print(f"Trajectory: {traj}")
    print(f"Poses     : {len(poses)}")
    print(f"JSON      : {args.out_json}")
    print(f"CSV       : {args.out_csv}")
    print(f"Warnings  : {len(warnings)}")
    for msg in warnings[:20]:
        print(f"WARN: {msg}")


if __name__ == "__main__":
    main()
