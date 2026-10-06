"""
Visualize B3 Phenikaa local vector predictions.

Creates MapTR-like qualitative previews:
  - *_surround_view.jpg
  - *_PRED_MAP_plot.png
  - *_SAMPLE_VIS.jpg
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phenikaa_hdmap.paths import PHENIKAA_DATA

CLASS_COLORS = {
    "divider": "#00d1ff",
    "ped_crossing": "#ffcc00",
    "boundary": "#ff4d4d",
}


def _load_infos(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "infos" in payload:
        return list(payload["infos"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported infos format: {path}")


def _read_image(path: str, size: tuple[int, int]) -> np.ndarray:
    import mmcv

    img = mmcv.imread(path, channel_order="rgb")
    if img is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return mmcv.imresize(img, size)


def _write_image(path: Path, img: np.ndarray) -> None:
    import mmcv

    path.parent.mkdir(parents=True, exist_ok=True)
    mmcv.imwrite(img[..., ::-1], str(path))


def _make_surround_view(info: dict[str, Any], out_path: Path) -> None:
    cams = info.get("cams", {})
    order = [
        "CAM_P_L",
        "CAM_P_FL",
        "CAM_P_F",
        "CAM_P_FR",
        "CAM_P_R",
        "CAM_P_B",
        "CAM_F_L",
        "CAM_F_F",
        "CAM_F_R",
        "CAM_F_B",
    ]
    tile_w, tile_h = 320, 180
    tiles = []
    for cam in order:
        path = cams.get(cam, "")
        img = _read_image(path, (tile_w, tile_h))
        tiles.append(img)
    row1 = np.concatenate(tiles[:5], axis=1)
    row2 = np.concatenate(tiles[5:], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
    _write_image(out_path, grid)


def _plot_pred_map(frame: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="black")
    ax.set_facecolor("black")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-16, 16)
    ax.set_ylim(-32, 32)
    ax.grid(color="#333333", linewidth=0.5)
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.axvline(0, color="#555555", linewidth=0.7)
    ax.plot([0], [0], marker="^", color="white", markersize=8)

    for vec in frame.get("vectors", []):
        pts = np.asarray(vec.get("points", []), dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        color = CLASS_COLORS.get(vec.get("class_name", "unknown"), "#00ffff")
        score = float(vec.get("score", 1.0))
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.2, alpha=max(0.35, min(1.0, score * 3)))

    ax.set_title(f"{frame.get('token', '')} | vectors={len(frame.get('vectors', []))}", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#777777")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _make_sample_vis(surround_path: Path, pred_path: Path, out_path: Path) -> None:
    import mmcv

    left = mmcv.imread(str(surround_path), channel_order="rgb")
    right = mmcv.imread(str(pred_path), channel_order="rgb")
    h = max(left.shape[0], right.shape[0])
    left = mmcv.imresize(left, (int(left.shape[1] * h / left.shape[0]), h))
    right = mmcv.imresize(right, (int(right.shape[1] * h / right.shape[0]), h))
    combo = np.concatenate([left, right], axis=1)
    _write_image(out_path, combo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infos", type=Path, default=PHENIKAA_DATA / "infos_residential.pkl")
    parser.add_argument("--lanes-dir", type=Path, default=Path("outputs/phenikaa_vis/phenikaa_b3/lanes_local"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phenikaa_vis/phenikaa_b3/preview"))
    parser.add_argument("--max-frames", type=int, default=10)
    args = parser.parse_args()

    infos = _load_infos(args.infos)
    info_by_token = {str(info.get("token")): info for info in infos}
    lane_files = sorted(args.lanes_dir.glob("*.json"))[: args.max_frames]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    for lane_file in lane_files:
        frame = json.loads(lane_file.read_text(encoding="utf-8"))
        token = str(frame.get("token", ""))
        info = info_by_token.get(token)
        if info is None:
            continue
        stem = f"{int(frame.get('frame_index', made)):06d}_{token}"
        surround = args.out_dir / f"{stem}_surround_view.jpg"
        pred = args.out_dir / f"{stem}_PRED_MAP_plot.png"
        sample = args.out_dir / f"{stem}_SAMPLE_VIS.jpg"
        _make_surround_view(info, surround)
        _plot_pred_map(frame, pred)
        _make_sample_vis(surround, pred, sample)
        made += 1

    print(f"B3 preview done: {made} frames -> {args.out_dir}")


if __name__ == "__main__":
    main()
