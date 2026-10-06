#!/usr/bin/env python3
"""Create professional visualizations from existing MapTR outputs."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/khanh247/miniconda3/envs/maptr/bin/python")
TARGET = ROOT / "scripts" / "maptr" / "17_visualize_maptr_professional.py"
raise SystemExit(subprocess.call([str(PYTHON), str(TARGET), *sys.argv[1:]], cwd=str(ROOT)))
