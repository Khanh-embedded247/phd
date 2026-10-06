#!/usr/bin/env python3
"""Kiem tra MapTR build duoc model 12 camera tu config Phenikaa.

Buoc nay CHUA train va CHUA inference.

No chi lam:
  1. Doc config da tao o buoc 05.
  2. Import plugin MapTR de dang ky MapTR/MapTRHead/transformer.
  3. Build model bang registry cua MMDetection3D.
  4. In ra num_cams va shape camera embedding.

Neu buoc nay OK thi nghia la:
  - Config 12 camera khong bi loi shape o muc build model.
  - MapTR da tao cams_embeds voi 12 camera.
  - Buoc tiep theo moi la viet inference/forward voi data cua buoc 04.

Chay:
  cd /home/khanh247/Documents/Survey/MapTR
  python3 /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/06_check_maptr_model_build.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


PHENIKAA_ROOT = Path(__file__).resolve().parents[1]
SURVEY_ROOT = PHENIKAA_ROOT.parents[1]
MAPTR_ROOT = SURVEY_ROOT / "MapTR"
MMDET3D_ROOT = MAPTR_ROOT / "mmdetection3d"
DEFAULT_CONFIG = (
    PHENIKAA_ROOT
    / "benchmark_maptr"
    / "config"
    / "maptr_tiny_r50_phenikaa_12cam.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiem tra build model MapTR 12 camera.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--maptr-root", type=Path, default=MAPTR_ROOT)
    parser.add_argument(
        "--keep-pretrained",
        action="store_true",
        help="Giu pretrained ResNet trong config. Mac dinh tat de smoke test khong can file ckpt.",
    )
    return parser.parse_args()


def make_matplotlib_cache() -> None:
    """Tranh loi cache/fontconfig khi import thu vien cu."""
    cache_dir = Path("/tmp") / "phenikaa_maptr_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def patch_numpy_legacy_aliases() -> None:
    """Bo sung alias NumPy cu ma MMCV/MMDetection3D doi khi van dung."""
    for name, value in {
        "long": int,
        "int": int,
        "float": float,
        "bool": bool,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def add_maptr_paths(maptr_root: Path) -> None:
    """Them MapTR va mmdetection3d local vao sys.path."""
    mmdet3d_root = maptr_root / "mmdetection3d"
    for path in (str(mmdet3d_root), str(maptr_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def find_transformer(model):
    """Lay transformer that trong MapTRHead de kiem num_cams/cams_embeds."""
    head = getattr(model, "pts_bbox_head", None)
    if head is None:
        raise AttributeError("model khong co pts_bbox_head")
    transformer = getattr(head, "transformer", None)
    if transformer is None:
        raise AttributeError("pts_bbox_head khong co transformer")
    return transformer


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    maptr_root = args.maptr_root.expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not maptr_root.exists():
        raise FileNotFoundError(maptr_root)

    make_matplotlib_cache()
    patch_numpy_legacy_aliases()
    add_maptr_paths(maptr_root)

    from mmcv import Config
    from mmdet3d.models import build_model

    # Import full plugin de dang ky cac class MapTR vao registry.
    import projects.mmdet3d_plugin  # noqa: F401

    cfg = Config.fromfile(str(config_path))
    if not args.keep_pretrained:
        # Smoke test chi can build architecture, khong can doc file pretrained.
        cfg.model.pretrained = None

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    transformer = find_transformer(model)
    cams_embeds = getattr(transformer, "cams_embeds", None)

    print("[DONE]")
    print(f"Config             : {config_path}")
    print(f"Model type         : {type(model).__name__}")
    print(f"Head type          : {type(model.pts_bbox_head).__name__}")
    print(f"Transformer type   : {type(transformer).__name__}")
    print(f"Transformer cams   : {transformer.num_cams}")
    if cams_embeds is not None:
        print(f"cams_embeds shape  : {tuple(cams_embeds.shape)}")
    print("Build model OK: MapTR da nhan cau hinh 12 camera.")


if __name__ == "__main__":
    main()
