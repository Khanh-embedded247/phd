#!/usr/bin/env python3
"""Train MapTR Phenikaa from pipeline.yaml using a Python entrypoint."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PHENIKAA_ROOT = THIS_DIR.parents[0]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from pipeline_config import load_pipeline_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    cfg = load_pipeline_config()
    vars_cfg = cfg["_vars"]
    parser = argparse.ArgumentParser(description="Train MapTR Phenikaa OSM.")
    parser.add_argument("--config", type=Path, default=Path(vars_cfg["CONFIG"]))
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--validate", action="store_true", help="Enable validation during train.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_pipeline_config()
    vars_cfg = cfg["_vars"]
    runtime = cfg.get("runtime", {})

    maptr_root = Path(vars_cfg["MAPTR_ROOT"]).expanduser().resolve()
    conda_py = Path(vars_cfg.get("CONDA_PY", sys.executable)).expanduser()
    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(runtime.get("cuda_visible_devices", env.get("CUDA_VISIBLE_DEVICES", "0")))
    env["PYTORCH_CUDA_ALLOC_CONF"] = str(runtime.get("pytorch_cuda_alloc_conf", env.get("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")))
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("NVIDIA_TF32_OVERRIDE", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/phenikaa_maptr_matplotlib")
    env.setdefault("XDG_CACHE_HOME", "/tmp/phenikaa_maptr_matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    pythonpath = [str(maptr_root / "mmdetection3d"), str(maptr_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    ld_parts = [str(Path(p).expanduser()) for p in runtime.get("ld_library_path_extra", [])]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    if ld_parts:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts)

    print("[TRAIN MAPTR]")
    print(f"Python : {conda_py}")
    print(f"Config : {config_path}")
    print(f"CWD    : {maptr_root}")
    print(f"GPU    : {env['CUDA_VISIBLE_DEVICES']}")

    cuda_check = (
        "import torch; "
        "print('torch:', torch.__version__, 'compiled cuda:', torch.version.cuda); "
        "print('cuda available:', torch.cuda.is_available(), 'device count:', torch.cuda.device_count()); "
        "assert torch.cuda.is_available(), 'PyTorch chua thay CUDA'"
    )
    subprocess.run([str(conda_py), "-c", cuda_check], cwd=str(maptr_root), env=env, check=True)

    cmd = [str(conda_py), "tools/train.py", str(config_path), "--gpus", str(args.gpus)]
    if not args.validate:
        cmd.append("--no-validate")
    subprocess.run(cmd, cwd=str(maptr_root), env=env, check=True)


if __name__ == "__main__":
    main()
