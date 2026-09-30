#!/usr/bin/env python3
"""Tinh metric don gian cho ket qua inference MapTR Phenikaa.

Script nay doc cac file prediction.json do 11_infer_visualize_maptr_phenikaa.py tao ra.
Moi duong GT se duoc ghep voi duong prediction cung class gan nhat theo Chamfer distance.

Y nghia:
  - Chamfer cang nho thi duong du doan cang gan GT.
  - threshold 0.5m / 1.0m giup dem ti le duong du doan dat muc chap nhan duoc.
  - Day la metric nhanh de benchmark noi bo, khong thay the evaluator day du cua MapTR.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PHENIKAA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN_DIR = PHENIKAA_ROOT / "outputs" / "benchmark_maptr" / "Normal" / "inference_vis"
CLASS_NAMES = [
    "lane_divider",
    "road_edge_marking",
    "stop_line",
    "ped_crossing",
    "boundary",
    "speed_bump",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MapTR Phenikaa prediction.json files.")
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 1.0])
    return parser.parse_args()


def chamfer_distance(points_a: list[list[float]], points_b: list[list[float]]) -> float:
    """Tinh Chamfer trung binh hai chieu giua 2 polyline."""
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or len(a) == 0 or len(b) == 0:
        return float("inf")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float((d.min(axis=1).mean() + d.min(axis=0).mean()) * 0.5)


def line_length(points: list[list[float]]) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def load_prediction_files(in_dir: Path) -> list[Path]:
    paths = sorted(p for p in in_dir.glob("*/prediction.json") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"Khong tim thay */prediction.json trong {in_dir}")
    return paths


def match_sample(sample: dict) -> list[dict]:
    """Ghep tung GT voi prediction cung class gan nhat."""
    rows = []
    preds = sample.get("pred", []) or []
    used_pred_ids = set()

    for gt_id, gt in enumerate(sample.get("gt", []) or []):
        same_class = [
            (pred_id, pred)
            for pred_id, pred in enumerate(preds)
            if int(pred.get("label", -1)) == int(gt.get("label", -2))
        ]
        if not same_class:
            rows.append({
                "gt_id": gt_id,
                "pred_id": "",
                "matched": 0,
                "chamfer_m": "",
                "score": "",
                "class_name": gt.get("class_name", ""),
                "gt_length_m": line_length(gt.get("points", [])),
                "pred_length_m": "",
            })
            continue

        best_pred_id, best_pred = min(
            same_class,
            key=lambda item: chamfer_distance(gt.get("points", []), item[1].get("points", [])),
        )
        best_cd = chamfer_distance(gt.get("points", []), best_pred.get("points", []))
        used_pred_ids.add(best_pred_id)
        rows.append({
            "gt_id": gt_id,
            "pred_id": best_pred_id,
            "matched": 1,
            "chamfer_m": best_cd,
            "score": float(best_pred.get("score", 0.0)),
            "class_name": gt.get("class_name", ""),
            "gt_length_m": line_length(gt.get("points", [])),
            "pred_length_m": line_length(best_pred.get("points", [])),
        })

    for pred_id, pred in enumerate(preds):
        if pred_id in used_pred_ids:
            continue
        rows.append({
            "gt_id": "",
            "pred_id": pred_id,
            "matched": 0,
            "chamfer_m": "",
            "score": float(pred.get("score", 0.0)),
            "class_name": pred.get("class_name", ""),
            "gt_length_m": "",
            "pred_length_m": line_length(pred.get("points", [])),
        })

    return rows


def summarize(rows: list[dict], thresholds: list[float]) -> dict:
    gt_rows = [r for r in rows if r["gt_id"] != ""]
    pred_rows = [r for r in rows if r["pred_id"] != ""]
    matched_rows = [r for r in gt_rows if r["chamfer_m"] != ""]
    chamfers = np.asarray([float(r["chamfer_m"]) for r in matched_rows], dtype=np.float64)

    summary = {
        "samples": len(set(r["token"] for r in rows)),
        "gt_lines": len(gt_rows),
        "pred_lines": len(pred_rows),
        "matched_gt_lines": len(matched_rows),
        "mean_chamfer_m": float(chamfers.mean()) if len(chamfers) else None,
        "median_chamfer_m": float(np.median(chamfers)) if len(chamfers) else None,
        "max_chamfer_m": float(chamfers.max()) if len(chamfers) else None,
    }
    for threshold in thresholds:
        ok = int((chamfers <= threshold).sum()) if len(chamfers) else 0
        summary[f"gt_recall_at_{threshold:g}m"] = ok / len(gt_rows) if gt_rows else None
        summary[f"matched_precision_at_{threshold:g}m"] = ok / len(pred_rows) if pred_rows else None
    return summary


def main() -> None:
    args = parse_args()
    in_dir = args.in_dir.expanduser().resolve()
    out_csv = args.out_csv or (in_dir / "evaluation_metrics.csv")
    out_csv = out_csv.expanduser().resolve()

    rows = []
    for pred_file in load_prediction_files(in_dir):
        with pred_file.open("r", encoding="utf-8") as f:
            sample = json.load(f)
        sample_rows = match_sample(sample)
        for row in sample_rows:
            row.update({
                "sample_index": sample.get("index", ""),
                "token": sample.get("token", ""),
                "sample_dir": str(pred_file.parent),
            })
        rows.extend(sample_rows)

    fieldnames = [
        "sample_index",
        "token",
        "class_name",
        "gt_id",
        "pred_id",
        "matched",
        "chamfer_m",
        "score",
        "gt_length_m",
        "pred_length_m",
        "sample_dir",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, args.thresholds)
    summary_path = out_csv.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE]")
    print(f"Input dir : {in_dir}")
    print(f"CSV       : {out_csv}")
    print(f"Summary   : {summary_path}")
    print("")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:24s}: {value:.6f}")
        else:
            print(f"{key:24s}: {value}")


if __name__ == "__main__":
    main()
