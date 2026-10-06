#!/usr/bin/env python3
"""Tao checkpoint khoi tao cho MapTR 12 camera tu checkpoint nuScenes.

Ly do can buoc nay:
  - checkpoint nuScenes thuong hoc 6 camera.
  - model Phenikaa co cams_embeds shape (12, 256).
  - Load truc tiep co the loi shape mismatch.

Script nay build model 12 camera, doc checkpoint nguon, chi giu nhung weight
co ten va shape khop voi model hien tai, bo qua cac key lech shape.

Chay:
  python3 phd/phenikaa/scripts/maptr/08_make_12cam_init_checkpoint.py
"""

from __future__ import annotations

import argparse
import os
import sys

from pathlib import Path

PHENIKAA_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PHENIKAA_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch


SURVEY_ROOT = PHENIKAA_ROOT.parents[1]
MAPTR_ROOT = PHENIKAA_ROOT / "third_party" / "MapTR_phenikaa"
DEFAULT_CONFIG = PHENIKAA_ROOT / "configs" / "maptr" / "maptr_tiny_r50_phenikaa_12cam_train_osm.py"
DEFAULT_SRC_CKPT = PHENIKAA_ROOT / "third_party" / "MapTR" / "ckpts" / "maptrv2_nusc_r50_24ep.pth"
DEFAULT_OUT_CKPT = PHENIKAA_ROOT / "outputs" / "only_camera" / "Normal" / "ckpts" / "maptr_init_12cam_partial.pth"

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from phenikaa_maptr.pipeline_config import expand_config_value, load_pipeline_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    pipe_cfg = load_pipeline_config()
    vars_cfg = pipe_cfg["_vars"]
    train_cfg = pipe_cfg.get("train", {})
    parser = argparse.ArgumentParser(description="Tao partial checkpoint MapTR 12 camera.")
    parser.add_argument("--config", type=Path, default=Path(vars_cfg.get("CONFIG", DEFAULT_CONFIG)))
    parser.add_argument("--src-ckpt", type=Path, default=Path(expand_config_value(train_cfg.get("source_checkpoint", DEFAULT_SRC_CKPT), pipe_cfg)))
    parser.add_argument("--out-ckpt", type=Path, default=Path(vars_cfg.get("INIT_CHECKPOINT", DEFAULT_OUT_CKPT)))
    return parser.parse_args()


def make_matplotlib_cache() -> None:
    cache_dir = Path("/tmp") / "phenikaa_maptr_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


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


def get_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model" in ckpt:
            return ckpt["model"]
    return ckpt


def strip_module_prefix(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    src_ckpt = args.src_ckpt.expanduser().resolve()
    out_ckpt = args.out_ckpt.expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not src_ckpt.exists():
        raise FileNotFoundError(src_ckpt)

    make_matplotlib_cache()
    patch_numpy_legacy_aliases()
    add_maptr_paths()

    from mmcv import Config
    from mmdet3d.models import build_model

    import projects.mmdet3d_plugin  # noqa: F401

    cfg = Config.fromfile(str(config_path))
    cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    model_state = model.state_dict()

    src = torch.load(str(src_ckpt), map_location="cpu")
    src_state = get_state_dict(src)

    kept = {}
    skipped_shape = []
    skipped_missing = []
    for src_key, value in src_state.items():
        key = strip_module_prefix(src_key)
        if key not in model_state:
            skipped_missing.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        kept[key] = value

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": kept,
            "meta": {
                "source_checkpoint": str(src_ckpt),
                "target_config": str(config_path),
                "kept_keys": len(kept),
                "skipped_missing": len(skipped_missing),
                "skipped_shape": len(skipped_shape),
            },
        },
        str(out_ckpt),
    )

    print("[DONE]")
    print(f"Source ckpt       : {src_ckpt}")
    print(f"Output ckpt       : {out_ckpt}")
    print(f"Model keys        : {len(model_state)}")
    print(f"Source keys       : {len(src_state)}")
    print(f"Kept keys         : {len(kept)}")
    print(f"Skipped missing   : {len(skipped_missing)}")
    print(f"Skipped bad shape : {len(skipped_shape)}")
    print("First shape mismatches:")
    for item in skipped_shape[:20]:
        print(f"  {item[0]}: src={item[1]} target={item[2]}")


if __name__ == "__main__":
    main()
