#!/usr/bin/env python3
"""Run Phenikaa MapTR pipeline from the project root."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/khanh247/miniconda3/envs/maptr/bin/python")
TARGET = ROOT / "scripts" / "maptr" / "30_run_pipeline_from_config.py"
raise SystemExit(subprocess.call([str(PYTHON), str(TARGET), *sys.argv[1:]], cwd=str(ROOT)))
