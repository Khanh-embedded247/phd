#!/usr/bin/env python3
"""Run MapTR inference for Phenikaa data.

Main outputs:
  - predictions.pkl: raw model outputs.
  - predictions.json: compact vector predictions for export/postprocess/visualization.

Debug BEV images are disabled by default because they are slow and not used by
professional visualization. Enable them only with --save-debug-images.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

from pathlib import Path

PHENIKAA_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PHENIKAA_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from typing import Any

import numpy as np


SURVEY_ROOT = PHENIKAA_ROOT.parents[1]
MAPTR_ROOT = PHENIKAA_ROOT / "third_party" / "MapTR_phenikaa"
DEFAULT_CONFIG = PHENIKAA_ROOT / "configs" / "maptr" / "maptr_tiny_r50_phenikaa_12cam_train_osm.py"
DEFAULT_CKPT = (
    PHENIKAA_ROOT
    / "outputs"
    / "only_camera"
    / "Normal"
    / "work_dirs"
    / "maptr_12cam_osm"
    / "latest.pth"
)
DEFAULT_OUT_DIR = PHENIKAA_ROOT / "outputs" / "only_camera" / "Normal" / "inference_vis"
DEFAULT_INFOS = PHENIKAA_ROOT / "outputs" / "only_camera" / "Normal" / "phenikaa_maptr_infos_val.pkl"

DEFAULT_CLASS_NAMES = [
    "lane_divider",
    "road_edge_marking",
    "stop_line",
    "ped_crossing",
    "boundary",
    "speed_bump",
]
DEFAULT_CLASS_COLORS = {
    0: "#ff9f1c",  # lane divider
    1: "#00a6a6",  # road edge marking
    2: "#e74c3c",  # stop line
    3: "#2f80ed",  # ped crossing
    4: "#27ae60",  # boundary
    5: "#8e44ad",  # speed bump
}

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from phenikaa_maptr.pipeline_config import load_pipeline_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    pipe_cfg = load_pipeline_config()
    vars_cfg = pipe_cfg["_vars"]
    infer_cfg = pipe_cfg.get("inference", {})
    parser = argparse.ArgumentParser(description="Inference MapTR Phenikaa and save predictions.")
    parser.add_argument("--config", type=Path, default=Path(vars_cfg.get("CONFIG", DEFAULT_CONFIG)))
    parser.add_argument("--checkpoint", type=Path, default=Path(vars_cfg.get("CHECKPOINT", DEFAULT_CKPT)))
    parser.add_argument("--out-dir", type=Path, default=Path(vars_cfg.get("INTERMEDIATE_DIR", DEFAULT_OUT_DIR)))
    parser.add_argument("--infos", type=Path, default=Path(vars_cfg.get("INFOS", DEFAULT_INFOS)),
                        help="Ghi de cfg.data.test.ann_file bang infos pkl khac.")
    parser.add_argument("--num-samples", type=int, default=int(infer_cfg.get("num_samples", 20)))
    parser.add_argument("--start-index", type=int, default=int(infer_cfg.get("start_index", 0)))
    parser.add_argument("--score-thresh", type=float, default=float(infer_cfg.get("score_thresh", 0.30)))
    parser.add_argument("--image-scale", type=float, default=float(infer_cfg.get("image_scale", 0.25)))
    parser.add_argument("--workers-per-gpu", type=int, default=int(infer_cfg.get("workers_per_gpu", 0)))
    parser.add_argument("--skip-empty-gt", action="store_true")
    parser.add_argument("--no-gt", action="store_true",
                        default=bool(infer_cfg.get("no_gt", False)),
                        help="Chay inference thuc te: khong lay GT OSM vao output/visualize.")
    parser.add_argument("--save-debug-images", action="store_true",
                        default=bool(infer_cfg.get("save_debug_images", False)),
                        help="Luu gt_bev/pred_bev/overlay_bev debug images. Mac dinh tat de chay nhanh.")
    return parser.parse_args()


def make_runtime_env() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/phenikaa_maptr_matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/phenikaa_maptr_matplotlib")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def patch_numpy_legacy_aliases() -> None:
    for name, value in {
        "long": int,
        "int": int,
        "float": float,
        "bool": bool,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def add_maptr_paths() -> None:
    for path in (MAPTR_ROOT / "mmdetection3d", MAPTR_ROOT):
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)


def set_random_scale(pipeline: Any, scale: float) -> None:
    """Doi moi RandomScaleImageMultiViewImage trong pipeline sang scale mong muon."""
    if isinstance(pipeline, list):
        for item in pipeline:
            set_random_scale(item, scale)
    elif isinstance(pipeline, dict):
        if pipeline.get("type") == "RandomScaleImageMultiViewImage":
            pipeline["scales"] = [scale]
        for value in pipeline.values():
            set_random_scale(value, scale)


def tensor_to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def sample_token_from_meta(img_metas: list[dict]) -> str:
    meta0 = img_metas[0]
    token = meta0.get("sample_idx") or meta0.get("sample_token")
    if token:
        return str(token)
    pts_filename = Path(meta0.get("pts_filename", "sample")).stem
    return pts_filename


def prediction_to_records(result: dict, score_thresh: float, class_names: list[str]) -> list[dict]:
    pred = result["pts_bbox"]
    scores = tensor_to_numpy(pred["scores_3d"])
    labels = tensor_to_numpy(pred["labels_3d"]).astype(int)
    pts = tensor_to_numpy(pred["pts_3d"])
    boxes = tensor_to_numpy(pred["boxes_3d"])

    records = []
    for score, label, line_pts, box in zip(scores, labels, pts, boxes):
        if float(score) < score_thresh:
            continue
        records.append({
            "score": float(score),
            "label": int(label),
            "class_name": class_names[int(label)] if int(label) < len(class_names) else str(int(label)),
            "points": np.asarray(line_pts, dtype=float).tolist(),
            "box": np.asarray(box, dtype=float).tolist(),
        })
    return records


def gt_to_records(gt_bboxes, gt_labels, class_names: list[str]) -> list[dict]:
    labels = tensor_to_numpy(gt_labels).astype(int)
    try:
        gt_points = tensor_to_numpy(gt_bboxes.fixed_num_sampled_points)
    except AssertionError:
        return []
    records = []
    for label, line_pts in zip(labels, gt_points):
        records.append({
            "label": int(label),
            "class_name": class_names[int(label)] if int(label) < len(class_names) else str(int(label)),
            "points": np.asarray(line_pts, dtype=float).tolist(),
        })
    return records


def draw_bev(records: list[dict], pc_range: list[float], out_path: Path, title: str, class_names: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 8), dpi=180)
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.25, alpha=0.3)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x lidar/base (m)")
    ax.set_ylabel("y lidar/base (m)")
    ax.add_patch(Rectangle((-1.1, -1.6), 2.2, 3.2, fill=False, edgecolor="black", linewidth=1.0))

    for rec in records:
        label = int(rec["label"])
        color = DEFAULT_CLASS_COLORS.get(label, "black")
        pts = np.asarray(rec["points"], dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        alpha = 0.45 + 0.45 * min(float(rec.get("score", 1.0)), 1.0)
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.25, alpha=alpha)
        ax.scatter(pts[:, 0], pts[:, 1], color=color, s=3, alpha=alpha)
        if "score" in rec:
            ax.text(pts[0, 0], pts[0, 1], f"{rec['score']:.2f}", fontsize=5, color=color)

    handles = []
    for idx, name in enumerate(class_names):
        handles.append(plt.Line2D([0], [0], color=DEFAULT_CLASS_COLORS.get(idx, "black"), lw=2, label=name))
    ax.legend(handles=handles, loc="upper right", fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_overlay(gt_records: list[dict], pred_records: list[dict], pc_range: list[float], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 8), dpi=180)
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.25, alpha=0.3)
    ax.set_title("GT dashed / PRED solid", fontsize=9)
    ax.set_xlabel("x lidar/base (m)")
    ax.set_ylabel("y lidar/base (m)")
    ax.add_patch(Rectangle((-1.1, -1.6), 2.2, 3.2, fill=False, edgecolor="black", linewidth=1.0))

    for rec in gt_records:
        color = DEFAULT_CLASS_COLORS.get(int(rec["label"]), "black")
        pts = np.asarray(rec["points"], dtype=float)
        if pts.ndim == 2 and pts.shape[0] >= 2:
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.0, linestyle="--", alpha=0.75)

    for rec in pred_records:
        color = DEFAULT_CLASS_COLORS.get(int(rec["label"]), "black")
        pts = np.asarray(rec["points"], dtype=float)
        if pts.ndim == 2 and pts.shape[0] >= 2:
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5, alpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    make_runtime_env()
    patch_numpy_legacy_aliases()
    add_maptr_paths()

    import torch
    import mmcv
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.datasets import build_dataset
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    import projects.mmdet3d_plugin  # noqa: F401

    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch chua thay CUDA. Hay chay script nay trong terminal maptr co GPU.")

    cfg = Config.fromfile(str(config_path))
    class_names = list(cfg.get("map_classes", cfg.get("PHENIKAA_MAP_CLASSES", DEFAULT_CLASS_NAMES)))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    if args.infos is not None:
        infos_path = args.infos.expanduser().resolve()
        if not infos_path.exists():
            raise FileNotFoundError(infos_path)
        cfg.data.test.ann_file = str(infos_path)
    cfg.data.test.pop("samples_per_gpu", None)
    cfg.data.workers_per_gpu = args.workers_per_gpu
    set_random_scale(cfg.data.test.pipeline, args.image_scale)

    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = not args.no_gt
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    pc_range = list(cfg.point_cloud_range)
    all_json = []
    raw_results = []

    print("[INFO] Begin inference")
    processed = 0
    for data_idx, data in enumerate(data_loader):
        if data_idx < args.start_index:
            continue
        if args.num_samples > 0 and processed >= args.num_samples:
            break

        img_metas = data["img_metas"][0].data[0]
        token = sample_token_from_meta(img_metas)

        gt_records = []
        gt_bboxes = None if args.no_gt else data.get("gt_bboxes_3d")
        gt_labels = None if args.no_gt else data.get("gt_labels_3d")
        if gt_bboxes is not None and gt_labels is not None:
            gt_records = gt_to_records(gt_bboxes.data[0][0], gt_labels.data[0][0], class_names)
        if args.skip_empty_gt and not gt_records:
            continue

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        pred_records = prediction_to_records(result[0], args.score_thresh, class_names)
        sample_dir = out_dir / f"{data_idx:06d}_{token}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        if args.save_debug_images:
            draw_bev(gt_records, pc_range, sample_dir / "gt_bev.png", f"GT {token}", class_names)
            draw_bev(pred_records, pc_range, sample_dir / "pred_bev.png", f"PRED {token}", class_names)
            draw_overlay(gt_records, pred_records, pc_range, sample_dir / "overlay_bev.png")

        sample_json = {
            "index": data_idx,
            "token": token,
            "score_thresh": args.score_thresh,
            "class_names": class_names,
            "gt": gt_records,
            "pred": pred_records,
            "sample_dir": str(sample_dir),
        }
        with (sample_dir / "prediction.json").open("w", encoding="utf-8") as f:
            json.dump(sample_json, f, ensure_ascii=False, indent=2)

        all_json.append(sample_json)
        raw_results.append(result[0])
        processed += 1
        print(f"[{processed}/{args.num_samples}] {token}: gt={len(gt_records)} pred={len(pred_records)}")

    with (out_dir / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump(all_json, f, ensure_ascii=False, indent=2)
    with (out_dir / "predictions.pkl").open("wb") as f:
        pickle.dump(raw_results, f)

    print("[DONE]")
    print(f"Output dir      : {out_dir}")
    print(f"Predictions json: {out_dir / 'predictions.json'}")
    print(f"Predictions pkl : {out_dir / 'predictions.pkl'}")


if __name__ == "__main__":
    main()
