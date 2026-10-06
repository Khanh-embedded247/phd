#!/usr/bin/env python3
"""Re-export graph/thin OSM from existing predictions."""
from __future__ import annotations
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/khanh247/miniconda3/envs/maptr/bin/python")
TARGET = ROOT / "scripts" / "maptr" / "30_run_pipeline_from_config.py"
raise SystemExit(subprocess.call([str(PYTHON), str(TARGET), "--steps", "export"], cwd=str(ROOT)))
