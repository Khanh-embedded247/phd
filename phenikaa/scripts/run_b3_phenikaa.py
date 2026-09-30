#!/usr/bin/env python3
"""
B3 - Run 200-frame Phenikaa vector aggregation pipeline.

Before running:
    conda activate maptr

Run:
    cd /home/khanh247/Documents/Survey/phenikaa
    python scripts/run_b3_phenikaa.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


# =============================================================================
# CONFIG MAC DINH
# =============================================================================

PROJECT_ROOT = Path("/home/khanh247/Documents/Survey/phenikaa")

# Build this first with:
#   python -m phenikaa_hdmap.converters.build_phenikaa_infos
INFOS = PROJECT_ROOT / "data/phenikaa/infos_residential.pkl"

# File output cua MapTR sau khi forward 200 frame.
# Chap nhan:
#   - nuscmap_results.json
#   - result.pkl / result.pickle co pts_bbox: scores_3d, labels_3d, pts_3d
OUT_DIR = PROJECT_ROOT / "outputs/phenikaa_vis/phenikaa_b3"
DEFAULT_MAPTR_RESULTS = OUT_DIR / "maptr_results/nuscmap_results.json"

# Neu MAPTR_RESULTS chua dung, script se tu tim them trong cac vi tri nay.
DEFAULT_MAPTR_RESULT_CANDIDATES = [
    OUT_DIR / "maptr_results/nuscmap_results.json",
    OUT_DIR / "maptr_results/result.pkl",
    OUT_DIR / "maptr_results/result.pickle",
    PROJECT_ROOT / "outputs/phenikaa_vis/maptr_results/nuscmap_results.json",
    PROJECT_ROOT / "outputs/phenikaa_vis/maptr_results/result.pkl",
    PROJECT_ROOT / "outputs/phenikaa_vis/maptr_results/result.pickle",
]
DEFAULT_MAX_FRAMES = 200
DEFAULT_SCORE_THRESH = 0.15
DEFAULT_MERGE_DIST = 1.0
DEFAULT_BASELINE = "b2_fusion"
DEFAULT_CAMERA_SET = None

# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infos", type=Path, default=INFOS)
    parser.add_argument("--maptr-results", type=Path, default=DEFAULT_MAPTR_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--baseline",
        choices=("b1_camera", "b2_fusion"),
        default=DEFAULT_BASELINE,
        help="b1_camera = MapTRv2 camera-only; b2_fusion = MapTR camera+LiDAR fusion from B2.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--score-thresh", type=float, default=DEFAULT_SCORE_THRESH)
    parser.add_argument("--merge-dist", type=float, default=DEFAULT_MERGE_DIST)
    parser.add_argument("--preview-frames", type=int, default=10)
    parser.add_argument(
        "--camera-set",
        choices=("phenikaa6", "all10"),
        default=DEFAULT_CAMERA_SET,
        help="Default is phenikaa6 for b2_fusion and all10 for b1_camera.",
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="Disable image undistortion from Camera_Intrinsics.json distortion coefficients.",
    )
    parser.add_argument(
        "--no-step1",
        action="store_true",
        help="Do not run MapTR forward automatically when --maptr-results is missing.",
    )
    parser.add_argument(
        "--force-step1",
        action="store_true",
        help="Run MapTR forward and overwrite the default Step 1 result even if it already exists.",
    )
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phenikaa_matplotlib"))
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _preflight_maptr_runtime() -> None:
    code = (
        "import torch, laspy\n"
        "print('cuda_available=', torch.cuda.is_available())\n"
        "print('cuda_device_count=', torch.cuda.device_count())\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Khong kiem tra duoc runtime B3 fusion.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    if "cuda_available= True" not in proc.stdout:
        raise RuntimeError(
            "B3 can CUDA vi MapTR trong repo nay dung CUDA-only BEV/sparse ops.\n"
            "Nhung runtime hien tai khong thay GPU CUDA.\n\n"
            f"{proc.stdout.strip()}\n\n"
            "Kiem tra tren may cua ban:\n"
            "  nvidia-smi\n"
            "  python - <<'PY'\n"
            "  import torch\n"
            "  print(torch.__version__, torch.version.cuda)\n"
            "  print(torch.cuda.is_available(), torch.cuda.device_count())\n"
            "  PY\n\n"
            "Khi `nvidia-smi` va `torch.cuda.is_available()` deu OK, chay lai lenh B3."
        )


def _candidate_maptr_results(maptr_results: Path, out_dir: Path) -> list[Path]:
    return list(
        dict.fromkeys(
            [
                maptr_results,
                out_dir / "maptr_results/nuscmap_results.json",
                out_dir / "maptr_results/result.pkl",
                out_dir / "maptr_results/result.pickle",
                *DEFAULT_MAPTR_RESULT_CANDIDATES,
            ]
        )
    )


def _find_maptr_results(maptr_results: Path, out_dir: Path) -> Optional[Path]:
    for path in _candidate_maptr_results(maptr_results, out_dir):
        if path.is_file():
            return path
    return None


def _raise_missing_maptr_results(maptr_results: Path, out_dir: Path) -> None:
    checked_paths = _candidate_maptr_results(maptr_results, out_dir)
    checked = "\n".join(f"  - {path}" for path in checked_paths)
    raise FileNotFoundError(
        "Khong thay file ket qua MapTR cho Step 1.\n"
        "Script da kiem tra:\n"
        f"{checked}\n\n"
        "Chay Step 1 truoc bang:\n"
        "  python -m phenikaa_hdmap.tools.step1_infer_maptr\n\n"
        "Hoac neu file cua ban nam cho khac, chay:\n"
        "  python scripts/run_b3_phenikaa.py --maptr-results /duong/dan/toi/result.pkl"
    )


def _ensure_maptr_results(args: argparse.Namespace) -> Path:
    step1_out = args.out_dir / "maptr_results/nuscmap_results.json"
    if args.force_step1:
        print("\n--force-step1: chay lai B3 Step 1 va ghi de MapTR result...")
        _run_step1(args, step1_out)
        return step1_out

    found = _find_maptr_results(args.maptr_results, args.out_dir)
    if found is not None:
        return found
    if args.no_step1:
        _raise_missing_maptr_results(args.maptr_results, args.out_dir)

    print("\nChua co MapTR result, tu chay B3 Step 1 truoc...")
    _run_step1(args, step1_out)
    found = _find_maptr_results(step1_out, args.out_dir)
    if found is None:
        _raise_missing_maptr_results(step1_out, args.out_dir)
    return found


def _run_step1(args: argparse.Namespace, step1_out: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "phenikaa_hdmap.tools.step1_infer_maptr",
        "--infos",
        str(args.infos),
        "--out",
        str(step1_out),
        "--max-frames",
        str(args.max_frames),
        "--score-thresh",
        str(args.score_thresh),
        "--baseline",
        args.baseline,
    ]
    if args.camera_set is not None:
        cmd.extend(["--camera-set", args.camera_set])
    if args.config is not None:
        cmd.extend(["--config", str(args.config)])
    if args.checkpoint is not None:
        cmd.extend(["--checkpoint", str(args.checkpoint)])
    if args.no_undistort:
        cmd.append("--no-undistort")
    _run(cmd)


def _resolve_maptr_results(maptr_results: Path, out_dir: Path) -> Path:
    """Backward-compatible wrapper for external imports/tests."""
    candidates = [
        maptr_results,
        out_dir / "maptr_results/nuscmap_results.json",
        out_dir / "maptr_results/result.pkl",
        out_dir / "maptr_results/result.pickle",
        *DEFAULT_MAPTR_RESULT_CANDIDATES,
    ]
    for path in candidates:
        if path.is_file():
            return path
    checked_paths = list(dict.fromkeys(candidates))
    checked = "\n".join(f"  - {path}" for path in checked_paths)
    raise FileNotFoundError(
        "Khong thay file ket qua MapTR cho Step 1.\n"
        "Script da kiem tra:\n"
        f"{checked}\n\n"
        "Ban can chay forward MapTR tren 200 frame Phenikaa truoc, hoac copy file ket qua vao:\n"
        f"  {out_dir / 'maptr_results/nuscmap_results.json'}\n\n"
        "Neu file cua ban nam cho khac, chay:\n"
        "  python scripts/run_b3_phenikaa.py --maptr-results /duong/dan/toi/result.pkl"
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "maptr_results").mkdir(parents=True, exist_ok=True)

    if not args.infos.is_file():
        raise FileNotFoundError(
            f"Khong thay infos: {args.infos}\n"
            "Chay truoc: python -m phenikaa_hdmap.converters.build_phenikaa_infos"
        )
    if not args.no_step1:
        _preflight_maptr_runtime()
    maptr_results = _ensure_maptr_results(args)

    _run(
        [
            sys.executable,
            "-m",
            "phenikaa_hdmap.tools.infer_phenikaa",
            "--infos",
            str(args.infos),
            "--maptr-results",
            str(maptr_results),
            "--out-dir",
            str(args.out_dir),
            "--max-frames",
            str(args.max_frames),
            "--score-thresh",
            str(args.score_thresh),
        ]
    )

    manifest = args.out_dir / "lanes_local_manifest.json"
    _run(
        [
            sys.executable,
            "-m",
            "phenikaa_hdmap.tools.aggregate_vectors",
            "--manifest",
            str(manifest),
            "--out-dir",
            str(args.out_dir / "global_map"),
            "--score-thresh",
            str(args.score_thresh),
            "--merge-dist",
            str(args.merge_dist),
        ]
    )

    if args.preview_frames > 0:
        _run(
            [
                sys.executable,
                "-m",
                "phenikaa_hdmap.tools.vis_b3_predictions",
                "--infos",
                str(args.infos),
                "--lanes-dir",
                str(args.out_dir / "lanes_local"),
                "--out-dir",
                str(args.out_dir / "preview"),
                "--max-frames",
                str(args.preview_frames),
            ]
        )

    print("\nB3 xong.")
    print(f"Local vectors : {args.out_dir / 'lanes_local'}")
    print(f"Global map    : {args.out_dir / 'global_map'}")
    print(f"Preview       : {args.out_dir / 'preview'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
