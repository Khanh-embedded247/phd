#!/usr/bin/env python3
"""Chay pipeline MapTR Phenikaa bang mot file YAML.

Muc tieu:
  - Khong phai sua tung file Python khi doi scenario/duong dan/tham so.
      - Co the chay tung buoc ro rang: raw_qa, pose_qa, gt_qa, build_infos,
      filter_infos, make_config, make_init, check_dataset, train, infer,
      export, visualize.
  - Tat ca output cuoi cung nam trong mot final_dir de de kiem tra.

Vi du:
  python3 scripts/maptr/30_run_pipeline_from_config.py \
      --config configs/maptr/pipeline.yaml \
      --steps build_infos

  python3 scripts/maptr/30_run_pipeline_from_config.py \
      --config configs/maptr/pipeline.yaml \
      --steps infer,export,visualize
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from typing import Any

import yaml


THIS_DIR = Path(__file__).resolve().parent
PHENIKAA_ROOT_DEFAULT = THIS_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phenikaa MapTR pipeline from YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=THIS_DIR.parents[1] / "configs" / "maptr" / "pipeline.yaml",
        help="File YAML chua tham so pipeline.",
    )
    parser.add_argument(
        "--steps",
        default="all",
        help="all/deploy hoac danh sach cach nhau boi dau phay: raw_qa,pose_qa,gt_qa,build_infos,filter_infos,make_config,make_init,check_dataset,train,infer,export,visualize",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chi in lenh se chay, khong thuc thi.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML phai la dict: {path}")
    return data


def expand_text(value: str, variables: dict[str, str]) -> str:
    pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}")

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            raise KeyError(f"Bien ${{{key}}} chua duoc dinh nghia trong config.")
        return variables[key]

    previous = None
    current = value
    for _ in range(20):
        if current == previous:
            break
        previous = current
        current = pattern.sub(repl, current)
    return os.path.expanduser(current)


def expand_obj(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        return expand_text(value, variables)
    if isinstance(value, list):
        return [expand_obj(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: expand_obj(item, variables) for key, item in value.items()}
    return value


def bool_arg(enabled: bool, flag: str) -> list[str]:
    return [flag] if enabled else []


def int_arg(name: str, value: Any) -> list[str]:
    if value is None:
        return []
    return [name, str(int(value))]


def float_arg(name: str, value: Any) -> list[str]:
    if value is None:
        return []
    return [name, str(float(value))]


class Runner:
    def __init__(self, cfg: dict[str, Any], dry_run: bool) -> None:
        self.cfg = cfg
        self.dry_run = dry_run

        project = cfg.get("project", {})
        scenario = cfg.get("scenario", {})
        raw_paths = cfg.get("paths", {})

        base_vars = {
            "PHENIKAA_ROOT": str(Path(project.get("phenikaa_root", PHENIKAA_ROOT_DEFAULT)).expanduser().resolve()),
            "CONDA_PY": str(Path(project.get("conda_python", "python3")).expanduser()),
            "SCENARIO": str(scenario.get("name", "Normal")),
            "CHECKPOINT_SCENARIO": str(scenario.get("checkpoint_scenario", scenario.get("name", "Normal"))),
        }
        base_vars["MAPTR_ROOT"] = expand_text(str(project.get("maptr_root", "${PHENIKAA_ROOT}/third_party/MapTR_phenikaa")), base_vars)
        base_vars["DATA_ROOT"] = expand_text(str(scenario.get("data_root", "${PHENIKAA_ROOT}/data/raw/${SCENARIO}")), base_vars)

        first_paths: dict[str, Any] = {}
        for key in ("model_config", "checkpoint", "infos", "final_dir", "gt_osm", "init_checkpoint"):
            if key in raw_paths:
                first_paths[key] = expand_obj(raw_paths[key], base_vars)
        base_vars.update({
            "CONFIG": str(first_paths.get("model_config", "")),
            "CHECKPOINT": str(first_paths.get("checkpoint", "")),
            "INFOS": str(first_paths.get("infos", "")),
            "FINAL_DIR": str(first_paths.get("final_dir", "")),
            "GT_OSM": str(first_paths.get("gt_osm", "")),
            "INIT_CHECKPOINT": str(first_paths.get("init_checkpoint", "")),
        })
        paths = expand_obj(raw_paths, base_vars)
        base_vars.update({
            "CONFIG": str(paths.get("model_config", "")),
            "CHECKPOINT": str(paths.get("checkpoint", "")),
            "INFOS": str(paths.get("infos", "")),
            "FINAL_DIR": str(paths.get("final_dir", "")),
            "GT_OSM": str(paths.get("gt_osm", "")),
            "INIT_CHECKPOINT": str(paths.get("init_checkpoint", "")),
        })
        base_vars["INTERMEDIATE_DIR"] = expand_text(str(paths.get("intermediate_dir", "${FINAL_DIR}/_intermediate")), base_vars)

        self.vars = base_vars
        self.paths = expand_obj(paths, base_vars)
        self.phenikaa_root = Path(self.vars["PHENIKAA_ROOT"])
        self.maptr_root = Path(self.vars["MAPTR_ROOT"])
        self.conda_py = self.vars["CONDA_PY"]
        self.scenario = self.vars["SCENARIO"]
        self.data_root = Path(self.vars["DATA_ROOT"])
        self.infos = Path(self.vars["INFOS"])
        self.final_dir = Path(self.vars["FINAL_DIR"])
        self.intermediate_dir = Path(self.vars["INTERMEDIATE_DIR"])
        self.gt_osm = Path(self.vars["GT_OSM"]) if self.vars.get("GT_OSM") else self.data_root / f"{self.scenario}.osm"
        self.init_checkpoint = Path(self.vars["INIT_CHECKPOINT"]) if self.vars.get("INIT_CHECKPOINT") else self.phenikaa_root / "outputs" / "only_camera" / self.scenario / "ckpts" / "maptr_init_12cam_partial.pth"

    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        runtime = self.cfg.get("runtime", {})
        env["CUDA_VISIBLE_DEVICES"] = str(runtime.get("cuda_visible_devices", env.get("CUDA_VISIBLE_DEVICES", "0")))
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(runtime.get("pytorch_cuda_alloc_conf", env.get("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")))
        env.setdefault("MPLCONFIGDIR", "/tmp/phenikaa_maptr_matplotlib")
        env.setdefault("XDG_CACHE_HOME", "/tmp/phenikaa_maptr_matplotlib")
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

        pythonpath_parts = [
            str(self.maptr_root / "mmdetection3d"),
            str(self.maptr_root),
        ]
        if env.get("PYTHONPATH"):
            pythonpath_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

        ld_parts = [str(Path(p).expanduser()) for p in runtime.get("ld_library_path_extra", [])]
        if env.get("LD_LIBRARY_PATH"):
            ld_parts.append(env["LD_LIBRARY_PATH"])
        if ld_parts:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts)
        return env

    def run(self, cmd: list[str], cwd: Path | None = None) -> None:
        cwd = cwd or self.phenikaa_root
        print()
        print(f"[RUN] cwd={cwd}")
        print(shlex.join(cmd))
        if self.dry_run:
            return
        subprocess.run(cmd, cwd=str(cwd), env=self.env(), check=True)

    def copy_if_exists(self, src: Path, dst: Path) -> None:
        print(f"[COPY] {src} -> {dst}")
        if self.dry_run:
            return
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            print(f"[WARN] Khong thay file: {src}")

    def step_raw_qa(self) -> None:
        raw = self.cfg.get("raw_qa", {})
        if not bool(raw.get("enabled", True)):
            print("[SKIP] raw_qa.enabled=false")
            return
        cmd = [
            self.conda_py,
            str(self.phenikaa_root / "scripts" / "maptr" / "01_check_raw_scenario.py"),
            "--scenario", self.scenario,
            "--data-root", str(self.data_root),
            "--max-camera-dt-sec", str(float(raw.get("max_camera_dt_sec", self.cfg.get("build_infos", {}).get("max_camera_dt_sec", 0.001)))),
            "--max-image-checks", str(int(raw.get("max_image_checks", 24))),
            "--out-report", str(self.final_dir / "qa" / "raw_scenario_report.json"),
        ]
        self.run(cmd)

    def step_pose_qa(self) -> None:
        pose = self.cfg.get("pose_qa", {})
        if not bool(pose.get("enabled", True)):
            print("[SKIP] pose_qa.enabled=false")
            return
        cmd = [
            self.conda_py,
            str(self.phenikaa_root / "scripts" / "maptr" / "02_check_pose_quality.py"),
            "--traj", str(self.data_root / "dump" / "traj_lidar.txt"),
            "--max-speed-mps", str(float(pose.get("max_speed_mps", 30.0))),
            "--max-yaw-rate-dps", str(float(pose.get("max_yaw_rate_dps", 120.0))),
            "--max-dt-sec", str(float(pose.get("max_dt_sec", 0.20))),
            "--out-json", str(self.final_dir / "qa" / "pose_quality_report.json"),
            "--out-csv", str(self.final_dir / "qa" / "pose_quality_steps.csv"),
        ]
        self.run(cmd)

    def step_gt_qa(self) -> None:
        gt = self.cfg.get("gt_qa", {})
        if not bool(gt.get("enabled", True)):
            print("[SKIP] gt_qa.enabled=false")
            return
        cmd = [
            self.conda_py,
            str(self.phenikaa_root / "scripts" / "maptr" / "04_check_gt_osm_semantic.py"),
            "--osm", str(self.gt_osm),
            "--out-report", str(self.final_dir / "qa" / "gt_osm_semantic_report.json"),
        ]
        if bool(gt.get("strict_missing_semantic", False)):
            cmd.append("--strict-missing-semantic")
        self.run(cmd)

    def step_build_infos(self) -> None:
        build = self.cfg.get("build_infos", {})
        rebuild = bool(build.get("rebuild", True))
        if self.infos.exists() and not rebuild:
            print(f"[SKIP] Infos da ton tai: {self.infos}")
            return

        cmd = [
            self.conda_py,
            str(self.phenikaa_root / "scripts" / "maptr" / "03_build_phenikaa_infos.py"),
            "--scenario", self.scenario,
            "--data-root", str(self.data_root),
            "--split", str(build.get("split", "full")),
            "--start-index", str(int(build.get("start_index", 0))),
            "--max-samples", str(int(build.get("max_samples", 0))),
            "--max-camera-dt-sec", str(float(build.get("max_camera_dt_sec", 0.001))),
            "--warn-lidar-dt-sec", str(float(build.get("warn_lidar_dt_sec", 0.05))),
            "--image-mode", str(build.get("image_mode", "undistorted")),
            "--out-pkl", str(self.infos),
        ]
        cmd += bool_arg(bool(build.get("overwrite_synced_lidar", False)), "--overwrite-synced-lidar")
        self.run(cmd)

    def step_filter_infos(self) -> None:
        filt = self.cfg.get("filter_infos", {})
        if not bool(filt.get("enabled", True)):
            print("[SKIP] filter_infos.enabled=false")
            return

        datasets = self.cfg.get("train_data", {}).get("datasets", [])
        if not datasets:
            source_infos = self.infos
            out_infos = source_infos.with_name(source_infos.stem + "_gt_positive.pkl")
            cmd = [
                self.conda_py,
                str(self.phenikaa_root / "scripts" / "maptr" / "06_filter_infos_with_osm_gt.py"),
                "--infos", str(source_infos),
                "--osm", str(self.gt_osm),
                "--out-pkl", str(out_infos),
                "--min-gt-vectors", str(int(filt.get("min_gt_vectors", 1))),
                "--progress-every", str(int(filt.get("progress_every", 500))),
            ]
            self.run(cmd)
            return

        for item in datasets:
            out_infos = Path(expand_text(str(item["infos"]), self.vars))
            if out_infos.name.endswith("_gt_positive.pkl"):
                source_infos = out_infos.with_name(out_infos.name.replace("_gt_positive.pkl", ".pkl"))
            else:
                source_infos = out_infos
                out_infos = source_infos.with_name(source_infos.stem + "_gt_positive.pkl")
            gt_osm = Path(expand_text(str(item["gt_osm"]), self.vars))
            cmd = [
                self.conda_py,
                str(self.phenikaa_root / "scripts" / "maptr" / "06_filter_infos_with_osm_gt.py"),
                "--infos", str(source_infos),
                "--osm", str(gt_osm),
                "--out-pkl", str(out_infos),
                "--min-gt-vectors", str(int(filt.get("min_gt_vectors", 1))),
                "--progress-every", str(int(filt.get("progress_every", 500))),
            ]
            self.run(cmd)

    def step_make_config(self) -> None:
        train = self.cfg.get("train", {})
        if not bool(train.get("make_config", True)):
            print("[SKIP] train.make_config=false")
            return
        cmd = [
            self.conda_py,
            str(self.phenikaa_root / "scripts" / "maptr" / "07_make_phenikaa_train_config.py"),
            "--osm", str(self.gt_osm),
            "--infos", str(self.infos),
            "--out-config", str(self.vars["CONFIG"]),
            "--work-dir", str(Path(expand_text(str(train.get("work_dir", "${PHENIKAA_ROOT}/outputs/only_camera/${SCENARIO}/work_dirs/maptr_12cam_osm")), self.vars))),
            "--load-from", str(self.init_checkpoint),
            "--max-epochs", str(int(train.get("max_epochs", 12))),
            "--lr", str(float(train.get("lr", 2e-4))),
            "--train-image-scale", str(float(train.get("image_scale", 0.25))),
            "--samples-per-gpu", str(int(train.get("samples_per_gpu", 1))),
            "--workers-per-gpu", str(int(train.get("workers_per_gpu", 1))),
        ]
        self.run(cmd)

    def step_make_init(self) -> None:
        train = self.cfg.get("train", {})
        if not bool(train.get("make_init_checkpoint", True)):
            print("[SKIP] train.make_init_checkpoint=false")
            return
        cmd = [
            self.conda_py,
            str(self.phenikaa_root / "scripts" / "maptr" / "08_make_12cam_init_checkpoint.py"),
            "--config", str(self.vars["CONFIG"]),
            "--src-ckpt", str(Path(expand_text(str(train.get("source_checkpoint", "${PHENIKAA_ROOT}/third_party/MapTR/ckpts/maptrv2_nusc_r50_24ep.pth")), self.vars))),
            "--out-ckpt", str(self.init_checkpoint),
        ]
        self.run(cmd, cwd=self.maptr_root)

    def step_check_dataset(self) -> None:
        train = self.cfg.get("train", {})
        if not bool(train.get("check_dataset", True)):
            print("[SKIP] train.check_dataset=false")
            return
        cmd = [
            self.conda_py,
            str(self.phenikaa_root / "scripts" / "maptr" / "09_check_train_dataset_osm.py"),
            "--config", str(self.vars["CONFIG"]),
            "--num-samples", str(int(train.get("check_num_samples", 30))),
        ]
        self.run(cmd, cwd=self.maptr_root)

    def step_train(self) -> None:
        train = self.cfg.get("train", {})
        if not bool(train.get("enabled", False)):
            print("[SKIP] train.enabled=false")
            return
        cmd = [
            self.conda_py,
            str(self.phenikaa_root / "scripts" / "maptr" / "10_train_maptr_phenikaa_osm.py"),
            "--config", str(self.vars["CONFIG"]),
            "--gpus", str(int(train.get("gpus", 1))),
        ]
        self.run(cmd, cwd=self.maptr_root)

    def step_infer(self) -> None:
        infer = self.cfg.get("inference", {})
        if not bool(infer.get("enabled", True)):
            print("[SKIP] inference.enabled=false")
            return
        if not self.dry_run:
            self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.conda_py,
            str(self.phenikaa_root / "scripts" / "maptr" / "11_infer_visualize_maptr_phenikaa.py"),
            "--config", str(self.vars["CONFIG"]),
            "--checkpoint", str(self.vars["CHECKPOINT"]),
            "--infos", str(self.infos),
            "--out-dir", str(self.intermediate_dir),
            "--num-samples", str(int(infer.get("num_samples", 0))),
            "--start-index", str(int(infer.get("start_index", 0))),
            "--score-thresh", str(float(infer.get("score_thresh", 0.30))),
            "--image-scale", str(float(infer.get("image_scale", 0.25))),
            "--workers-per-gpu", str(int(infer.get("workers_per_gpu", 0))),
        ]
        cmd += bool_arg(bool(infer.get("no_gt", True)), "--no-gt")
        self.run(cmd, cwd=self.maptr_root)

        self.copy_if_exists(self.intermediate_dir / "predictions.json", self.final_dir / "predictions.json")
        self.copy_if_exists(self.intermediate_dir / "predictions.pkl", self.final_dir / "predictions.pkl")

    def step_export(self) -> None:
        export = self.cfg.get("export_osm", {})
        if not bool(export.get("enabled", True)):
            print("[SKIP] export_osm.enabled=false")
            return

        predictions = self.final_dir / "predictions.json"
        graph = export.get("graph", {})
        if bool(graph.get("enabled", True)):
            cmd = [
                self.conda_py,
                str(self.phenikaa_root / "scripts" / "maptr" / "22_export_prediction_osm_graph.py"),
                "--predictions", str(predictions),
                "--infos", str(self.infos),
                "--out-osm", str(Path(expand_text(str(graph.get("out_osm", "${FINAL_DIR}/predicted_vector_map_graph.osm")), self.vars))),
                "--score-thresh", str(float(export.get("score_thresh", 0.45))),
            ]
            cmd += float_arg("--min-length-m", graph.get("min_length_m", 0.40))
            cmd += float_arg("--dedup-chamfer-m", graph.get("dedup_chamfer_m", 0.65))
            cmd += float_arg("--dedup-cell-m", graph.get("dedup_cell_m", 2.0))
            cmd += int_arg("--sample-step", graph.get("sample_step", 1))
            cmd += int_arg("--max-lines", graph.get("max_lines", 0))
            self.run(cmd)

        thin = export.get("thin", {})
        if bool(thin.get("enabled", True)):
            cmd = [
                self.conda_py,
                str(self.phenikaa_root / "scripts" / "maptr" / "21_postprocess_predictions_to_thin_osm.py"),
                "--predictions", str(predictions),
                "--infos", str(self.infos),
                "--out-osm", str(Path(expand_text(str(thin.get("out_osm", "${FINAL_DIR}/predicted_vector_map_thin.osm")), self.vars))),
                "--score-thresh", str(float(export.get("score_thresh", 0.45))),
            ]
            if thin.get("merge_mode") is not None:
                cmd += ["--merge-mode", str(thin.get("merge_mode"))]
            cmd += float_arg("--min-length-m", thin.get("min_length_m", 0.40))
            cmd += float_arg("--merge-chamfer-m", thin.get("merge_chamfer_m", 0.45))
            cmd += float_arg("--merge-center-m", thin.get("merge_center_m", 2.0))
            cmd += float_arg("--merge-heading-deg", thin.get("merge_heading_deg", 25.0))
            cmd += float_arg("--thin-grid-m", thin.get("thin_grid_m", 0.75))
            cmd += int_arg("--min-cell-observations", thin.get("min_cell_observations", 3))
            cmd += int_arg("--min-edge-observations", thin.get("min_edge_observations", 2))
            cmd += int_arg("--min-graph-points", thin.get("min_graph_points", 4))
            cmd += int_arg("--graph-smooth-window", thin.get("graph_smooth_window", 3))
            cmd += float_arg("--stop-cluster-m", thin.get("stop_cluster_m", 3.0))
            cmd += float_arg("--stop-heading-deg", thin.get("stop_heading_deg", 45.0))
            cmd += int_arg("--min-stop-cluster-size", thin.get("min_stop_cluster_size", 4))
            cmd += float_arg("--stop-min-length-m", thin.get("stop_min_length_m", 1.0))
            cmd += float_arg("--stop-max-length-m", thin.get("stop_max_length_m", 12.0))
            cmd += int_arg("--stop-line-points", thin.get("stop_line_points", 5))
            cmd += bool_arg(bool(thin.get("debug_tags", False)), "--debug-tags")
            cmd += bool_arg(bool(thin.get("create_lanelet_relations", False)), "--create-lanelet-relations")
            cmd += float_arg("--lateral-bin-m", thin.get("lateral_bin_m", 0.25))
            cmd += float_arg("--min-lateral-bin-ratio", thin.get("min_lateral_bin_ratio", 0.02))
            cmd += int_arg("--min-lateral-cluster-points", thin.get("min_lateral_cluster_points", 500))
            cmd += float_arg("--s-bin-m", thin.get("s_bin_m", 1.0))
            cmd += int_arg("--min-bin-observations", thin.get("min_bin_observations", 3))
            cmd += int_arg("--min-line-points", thin.get("min_line_points", 8))
            cmd += int_arg("--smooth-window", thin.get("smooth_window", 7))
            self.run(cmd)

    def step_visualize(self) -> None:
        vis = self.cfg.get("visualize", {})
        if not bool(vis.get("enabled", True)):
            print("[SKIP] visualize.enabled=false")
            return

        base_cmd = [
            self.conda_py,
            str(self.phenikaa_root / "scripts" / "maptr" / "16_visualize_maptr_paper_style.py"),
            "--pred-dir", str(self.intermediate_dir),
            "--infos", str(self.infos),
            "--start", str(int(vis.get("start", 0))),
            "--max-samples", str(int(vis.get("max_samples", 20))),
            "--score-thresh", str(float(vis.get("score_thresh", 0.30))),
            "--cam-width", str(int(vis.get("cam_width", 320))),
            "--map-width", str(int(vis.get("map_width", 900))),
            "--map-height", str(int(vis.get("map_height", 1600))),
        ]

        paper_out = Path(expand_text(str(vis.get("paper_out_dir", "${FINAL_DIR}/visualizations/paper_style")), self.vars))
        self.run(base_cmd + ["--style", "paper", "--out-dir", str(paper_out)])

        map_out = Path(expand_text(str(vis.get("map_out_dir", "${FINAL_DIR}/visualizations/map_style")), self.vars))
        self.run(base_cmd + ["--style", "map", "--out-dir", str(map_out)])

        professional_out = Path(expand_text(str(vis.get("professional_out_dir", "${FINAL_DIR}/visualizations/professional")), self.vars))
        professional_cmd = [
            self.conda_py,
            str(self.phenikaa_root / "scripts" / "maptr" / "17_visualize_maptr_professional.py"),
            "--infos", str(self.infos),
            "--predictions", str(self.final_dir / "predictions.json"),
            "--intermediate-dir", str(self.intermediate_dir),
            "--graph-osm", str(self.final_dir / "predicted_vector_map_graph.osm"),
            "--thin-osm", str(self.final_dir / "predicted_vector_map_thin.osm"),
            "--out-dir", str(professional_out),
            "--start", str(int(vis.get("start", 0))),
            "--max-samples", str(int(vis.get("professional_max_samples", 6))),
            "--step", str(int(vis.get("professional_step", 250))),
            "--score-thresh", str(float(vis.get("score_thresh", 0.30))),
            "--global-width", str(int(vis.get("professional_global_width", 2200))),
            "--global-height", str(int(vis.get("professional_global_height", 1500))),
            "--local-size", str(int(vis.get("professional_local_size", 900))),
        ]
        self.run(professional_cmd)

    def write_readme(self) -> None:
        readme = self.final_dir / "README_RESULT.txt"
        text = (
            "Phenikaa MapTR final run\n"
            f"scenario     : {self.scenario}\n"
            f"infos        : {self.infos}\n"
            f"checkpoint   : {self.vars['CHECKPOINT']}\n"
            f"intermediate : {self.intermediate_dir}\n"
            "\n"
            "Main outputs:\n"
            "  predictions.json/pkl\n"
            "  predicted_vector_map_graph.osm\n"
            "  predicted_vector_map_thin.osm\n"
            "  visualizations/paper_style\n"
            "  visualizations/map_style\n"
        )
        print(f"[WRITE] {readme}")
        if self.dry_run:
            return
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text(text, encoding="utf-8")


def normalize_steps(value: str) -> list[str]:
    presets = {
        "all": ["raw_qa", "pose_qa", "gt_qa", "build_infos", "filter_infos", "make_config", "make_init", "check_dataset", "train", "infer", "export", "visualize"],
        "train_all": ["raw_qa", "pose_qa", "gt_qa", "build_infos", "filter_infos", "make_config", "make_init", "check_dataset", "train"],
        "deploy": ["raw_qa", "pose_qa", "build_infos", "infer", "export", "visualize"],
    }
    if value.strip() in presets:
        return presets[value.strip()]
    aliases = {
        "build": "build_infos",
        "infos": "build_infos",
        "inference": "infer",
        "osm": "export",
        "vis": "visualize",
        "raw": "raw_qa",
        "pose": "pose_qa",
        "gt": "gt_qa",
        "filter": "filter_infos",
        "config": "make_config",
        "init": "make_init",
        "dataset": "check_dataset",
    }
    steps = []
    for item in value.split(","):
        step = aliases.get(item.strip(), item.strip())
        if step:
            steps.append(step)
    valid = {"raw_qa", "pose_qa", "gt_qa", "build_infos", "filter_infos", "make_config", "make_init", "check_dataset", "train", "infer", "export", "visualize"}
    unknown = [step for step in steps if step not in valid]
    if unknown:
        raise ValueError(f"Step khong hop le: {unknown}. Hop le: {sorted(valid)}")
    return steps


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    runner = Runner(cfg, dry_run=args.dry_run)
    steps = normalize_steps(args.steps)

    print("[CONFIG]")
    print(f"scenario     : {runner.scenario}")
    print(f"data_root    : {runner.data_root}")
    print(f"infos        : {runner.infos}")
    print(f"checkpoint   : {runner.vars['CHECKPOINT']}")
    print(f"gt_osm       : {runner.gt_osm}")
    print(f"init_ckpt    : {runner.init_checkpoint}")
    print(f"final_dir    : {runner.final_dir}")
    print(f"steps        : {', '.join(steps)}")

    for step in steps:
        getattr(runner, f"step_{step}")()

    runner.write_readme()
    print("\n[DONE]")
    print(f"Output chinh: {runner.final_dir}")


if __name__ == "__main__":
    main()
