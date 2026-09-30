"""Helper nội bộ cho run_b1_*.py / run_b2_*.py — không cần sửa file này."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def maptr_root(project_root: Path) -> Path:
    return project_root / "third_party" / "MapTR"


def build_env(maptr: Path) -> dict[str, str]:
    import torch

    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{maptr}{os.pathsep}{env.get('PYTHONPATH', '')}"
    old_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{torch_lib}{os.pathsep}{old_ld}" if old_ld else torch_lib
    return env


def ensure_nuscenes_symlink(project_root: Path) -> None:
    data_dir = maptr_root(project_root) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    links = {
        "nuscenes": project_root / "data" / "nuscenes" / "raw",
        "can_bus": project_root / "data" / "nuscenes" / "can_bus",
    }
    for name, target in links.items():
        link = data_dir / name
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target, target_is_directory=True)


def run_eval(
    *,
    project_root: Path,
    config_rel: str,
    checkpoint_rel: str,
    output_dir: Path,
    log_name: str,
    gpu_count: int,
) -> Path:
    project_root = project_root.resolve()
    maptr = maptr_root(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_name

    ensure_nuscenes_symlink(project_root)

    config_path = maptr / config_rel
    ckpt_path = maptr / checkpoint_rel
    if not config_path.is_file():
        raise FileNotFoundError(f"Không thấy config: {config_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Không thấy checkpoint: {ckpt_path}")

    env = build_env(maptr)
    cmd = [
        "bash",
        str(maptr / "tools" / "dist_test_map.sh"),
        config_rel,
        checkpoint_rel,
        str(gpu_count),
    ]

    print("=== MapTR eval ===")
    print(f"PROJECT_ROOT = {project_root}")
    print(f"MAPTR        = {maptr}")
    print(f"CONFIG       = {config_rel}")
    print(f"CHECKPOINT   = {checkpoint_rel}")
    print(f"OUTPUT       = {output_dir}")
    print(f"CMD          = {' '.join(cmd)}")
    print()

    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(maptr),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_f.write(proc.stdout)
        print(proc.stdout)
        if proc.returncode != 0:
            print(f"\nLỖI: eval thoát với mã {proc.returncode}. Xem log: {log_path}")
            raise SystemExit(proc.returncode)

    print(f"\nXONG. Log lưu tại: {log_path}")
    return log_path


def run_vis(
    *,
    project_root: Path,
    vis_script_rel: str,
    config_rel: str,
    checkpoint_rel: str,
    show_dir: Path,
    score_thresh: float,
    max_samples: int | None,
    extra_args: list[str] | None = None,
) -> None:
    project_root = project_root.resolve()
    maptr = maptr_root(project_root)
    show_dir.mkdir(parents=True, exist_ok=True)
    ensure_nuscenes_symlink(project_root)

    env = build_env(maptr)
    cmd = [
        sys.executable,
        vis_script_rel,
        config_rel,
        checkpoint_rel,
        "--show-dir",
        str(show_dir.resolve()),
        "--score-thresh",
        str(score_thresh),
    ]
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])
    if extra_args:
        cmd.extend(extra_args)

    print("=== MapTR visualize ===")
    print(f"CMD = {' '.join(cmd)}")
    print()

    proc = subprocess.run(cmd, cwd=str(maptr), env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    print(f"\nXONG. Hình lưu tại: {show_dir}")


def parse_metrics(log_path: Path) -> dict[str, float]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics: dict[str, float] = {}

    for key in ("divider", "ped_crossing", "boundary", "map"):
        m = re.search(rf"^{key}:\s*([0-9.]+)\s*$", text, re.MULTILINE)
        if m:
            metrics[key] = float(m.group(1))

    if "map" not in metrics:
        m = re.search(r"'NuscMap_chamfer/mAP':\s*([0-9.]+)", text)
        if m:
            metrics["map"] = float(m.group(1))

    if len(metrics) < 4:
        raise ValueError(
            f"Không đọc đủ 4 số từ log: {log_path}\n"
            "Chạy eval trước, hoặc mở eval_log.txt xem có dòng divider:/map: không."
        )
    return metrics
