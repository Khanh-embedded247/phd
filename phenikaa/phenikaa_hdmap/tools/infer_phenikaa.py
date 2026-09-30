"""
B3 Step 1 - save per-frame MapTR vectors in the Phenikaa local frame.

This tool consumes either:
  * MapTR formatted JSON: {"results": [{"sample_token": ..., "vectors": ...}]}
  * MMDetection/MapTR pickle: list[{"pts_bbox": {"scores_3d", "labels_3d", "pts_3d"}}]

Direct Phenikaa forward inference still needs a MMDetection3D dataset adapter. Until
that adapter is wired, run MapTR with --out/--format-only, then normalize the
result with this script.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from phenikaa_hdmap.paths import PHENIKAA_DATA

MAPTR_CLASSES = {
    0: "divider",
    1: "ped_crossing",
    2: "boundary",
}


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)


def _load_infos(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "infos" in payload:
        return list(payload["infos"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported infos format: {path}")


def _load_result_payload(path: Path) -> Any:
    if path.suffix.lower() in {".json", ".js"}:
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("rb") as f:
        return pickle.load(f)


def _normalize_json_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    frames = []
    for frame in payload.get("results", []):
        vectors = []
        for vec in frame.get("vectors", []):
            pts = np.asarray(vec.get("pts", []), dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            vectors.append(
                {
                    "class_id": int(vec.get("type", vec.get("label", -1))),
                    "class_name": str(vec.get("cls_name", "unknown")),
                    "score": float(vec.get("confidence_level", vec.get("score", 1.0))),
                    "points": pts[:, :2].tolist(),
                }
            )
        frames.append(
            {
                "token": str(frame.get("sample_token", "")),
                "vectors": vectors,
            }
        )
    return frames


def _unwrap_detection(result: Any) -> dict[str, Any]:
    if isinstance(result, dict) and "pts_bbox" in result:
        return result["pts_bbox"]
    if isinstance(result, dict):
        return result
    raise ValueError(f"Unsupported result item type: {type(result)!r}")


def _normalize_pickle_results(payload: Any, infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and "bbox_results" in payload:
        payload = payload["bbox_results"]
    if not isinstance(payload, list):
        raise ValueError("Pickle result must be a list or contain bbox_results")

    frames = []
    for idx, result in enumerate(payload):
        det = _unwrap_detection(result)
        scores = _to_numpy(det["scores_3d"]).reshape(-1)
        labels = _to_numpy(det["labels_3d"]).reshape(-1)
        points = _to_numpy(det["pts_3d"])
        if points.ndim != 3:
            raise ValueError(f"Expected pts_3d as [N,P,D], got {points.shape}")

        vectors = []
        for score, label, pts in zip(scores, labels, points):
            pts = np.asarray(pts, dtype=np.float64)
            if pts.shape[0] < 2:
                continue
            class_id = int(label)
            vectors.append(
                {
                    "class_id": class_id,
                    "class_name": MAPTR_CLASSES.get(class_id, f"class_{class_id}"),
                    "score": float(score),
                    "points": pts[:, :2].tolist(),
                }
            )
        token = infos[idx].get("token", str(idx)) if idx < len(infos) else str(idx)
        frames.append({"token": str(token), "vectors": vectors})
    return frames


def _filter_frames(
    frames: list[dict[str, Any]],
    infos: list[dict[str, Any]],
    *,
    max_frames: int,
    score_thresh: float,
) -> list[dict[str, Any]]:
    pose_by_token = {str(info["token"]): info.get("pose_path") for info in infos if "token" in info}
    fallback_tokens = [str(info.get("token", i)) for i, info in enumerate(infos)]

    selected = []
    for idx, frame in enumerate(frames[:max_frames]):
        token = str(frame.get("token") or (fallback_tokens[idx] if idx < len(fallback_tokens) else idx))
        vectors = [
            vec
            for vec in frame.get("vectors", [])
            if float(vec.get("score", 1.0)) >= score_thresh and len(vec.get("points", [])) >= 2
        ]
        selected.append(
            {
                "frame_index": idx,
                "token": token,
                "pose_path": pose_by_token.get(token),
                "vectors": vectors,
            }
        )
    return selected


def write_local_vectors(frames: list[dict[str, Any]], out_dir: Path) -> Path:
    local_dir = out_dir / "lanes_local"
    local_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"frames": []}
    for frame in frames:
        name = f"{frame['frame_index']:06d}_{frame['token']}.json"
        out_path = local_dir / name
        out_path.write_text(json.dumps(frame, indent=2), encoding="utf-8")
        manifest["frames"].append(
            {
                "frame_index": frame["frame_index"],
                "token": frame["token"],
                "path": str(out_path),
                "pose_path": frame.get("pose_path"),
                "vector_count": len(frame["vectors"]),
            }
        )

    manifest_path = out_dir / "lanes_local_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infos", type=Path, default=PHENIKAA_DATA / "infos_residential.pkl")
    parser.add_argument(
        "--maptr-results",
        type=Path,
        default=Path("outputs/phenikaa_vis/maptr_results.pkl"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phenikaa_vis/phenikaa_b3"))
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--score-thresh", type=float, default=0.35)
    args = parser.parse_args()

    if not args.maptr_results.exists():
        raise FileNotFoundError(
            f"MapTR results file not found: {args.maptr_results}\n"
            "Run MapTR inference/eval first to produce the result file, or pass "
            "--maptr-results /path/to/results.pkl"
        )

    infos = _load_infos(args.infos)
    payload = _load_result_payload(args.maptr_results)
    if isinstance(payload, dict) and "results" in payload:
        frames = _normalize_json_results(payload)
    else:
        frames = _normalize_pickle_results(payload, infos)

    frames = _filter_frames(
        frames,
        infos,
        max_frames=args.max_frames,
        score_thresh=args.score_thresh,
    )
    manifest = write_local_vectors(frames, args.out_dir)

    vector_count = sum(len(frame["vectors"]) for frame in frames)
    print(f"Step 1 done: {len(frames)} frames, {vector_count} vectors")
    print(f"Local vectors manifest: {manifest}")


if __name__ == "__main__":
    main()
