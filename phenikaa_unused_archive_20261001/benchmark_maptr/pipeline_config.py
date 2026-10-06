"""Doc config/pipeline.yaml cho cac script benchmark_maptr.

Y tuong:
  - Chay file le van lay default tu YAML.
  - Neu can test nhanh, CLI args co the override YAML.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = THIS_DIR / "config" / "pipeline.yaml"
PHENIKAA_ROOT = THIS_DIR.parents[0]


def _expand_text(value: str, variables: dict[str, str]) -> str:
    pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}")

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            return match.group(0)
        return variables[key]

    old = None
    current = value
    for _ in range(20):
        if current == old:
            break
        old = current
        current = pattern.sub(repl, current)
    return os.path.expanduser(current)


def _expand_obj(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _expand_text(value, variables)
    if isinstance(value, list):
        return [_expand_obj(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _expand_obj(item, variables) for key, item in value.items()}
    return value


def load_pipeline_config(path: Path | None = None) -> dict[str, Any]:
    """Load YAML va them cac field da expand san de script dung truc tiep."""
    config_path = (path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    project = cfg.get("project", {})
    scenario = cfg.get("scenario", {})
    variables = {
        "PHENIKAA_ROOT": str(Path(project.get("phenikaa_root", PHENIKAA_ROOT)).expanduser().resolve()),
        "CONDA_PY": str(Path(project.get("conda_python", "python3")).expanduser()),
        "SCENARIO": str(scenario.get("name", "Normal")),
        "CHECKPOINT_SCENARIO": str(scenario.get("checkpoint_scenario", scenario.get("name", "Normal"))),
    }
    variables["MAPTR_ROOT"] = _expand_text(str(project.get("maptr_root", "${PHENIKAA_ROOT}/benchmark_maptr/vendor/MapTR")), variables)
    variables["DATA_ROOT"] = _expand_text(str(scenario.get("data_root", "${PHENIKAA_ROOT}/data/${SCENARIO}")), variables)

    raw_paths = cfg.get("paths", {})
    first_paths: dict[str, Any] = {}
    for key in ("model_config", "checkpoint", "infos", "final_dir", "gt_osm", "init_checkpoint"):
        if key in raw_paths:
            first_paths[key] = _expand_obj(raw_paths[key], variables)
    variables.update({
        "CONFIG": str(first_paths.get("model_config", "")),
            "CHECKPOINT": str(first_paths.get("checkpoint", "")),
            "INFOS": str(first_paths.get("infos", "")),
            "FINAL_DIR": str(first_paths.get("final_dir", "")),
            "GT_OSM": str(first_paths.get("gt_osm", "")),
            "INIT_CHECKPOINT": str(first_paths.get("init_checkpoint", "")),
    })
    paths = _expand_obj(raw_paths, variables)
    variables.update({
        "CONFIG": str(paths.get("model_config", variables["CONFIG"])),
        "CHECKPOINT": str(paths.get("checkpoint", variables["CHECKPOINT"])),
        "INFOS": str(paths.get("infos", variables["INFOS"])),
        "FINAL_DIR": str(paths.get("final_dir", variables["FINAL_DIR"])),
        "GT_OSM": str(paths.get("gt_osm", variables["GT_OSM"])),
        "INIT_CHECKPOINT": str(paths.get("init_checkpoint", variables["INIT_CHECKPOINT"])),
    })
    variables["INTERMEDIATE_DIR"] = _expand_text(str(paths.get("intermediate_dir", "${FINAL_DIR}/_intermediate")), variables)

    cfg["_config_path"] = str(config_path)
    cfg["_vars"] = variables
    cfg["_paths"] = _expand_obj(paths, variables)
    return cfg


def path_from_config(value: str | Path) -> Path:
    return Path(value).expanduser()


def expand_config_value(value: Any, cfg: dict[str, Any]) -> Any:
    """Expand ${...} trong mot gia tri lay tu YAML da load."""
    return _expand_obj(value, cfg["_vars"])
