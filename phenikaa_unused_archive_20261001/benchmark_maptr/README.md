# benchmark_maptr

Thu muc nay la pipeline MapTR/OSM dang dung that cho du lieu Phenikaa.

Chay tu root project bang wrapper:

```bash
cd /home/khanh247/Documents/Survey/phd/phenikaa
/home/khanh247/miniconda3/envs/maptr/bin/python scripts/00_run_maptr_pipeline.py --steps infer,export,visualize
```

Hoac goi truc tiep:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python benchmark_maptr/30_run_pipeline_from_config.py --steps <steps>
```

## Pipeline thuc te tu raw `dump/`

Neu chay tu dau, `30_run_pipeline_from_config.py` se goi cac file sau:

| Step | File | Tac dung |
|---|---|---|
| `raw_qa` | `01_check_raw_scenario.py` | Check camera/raw timestamp |
| `pose_qa` | `02_check_pose_quality.py` | Check trajectory/pose |
| `gt_qa` | `04_check_gt_osm_semantic.py` | Check tag GT OSM |
| `build_infos` | `03_build_phenikaa_infos.py` | Tao `infos.pkl`, anh undistorted, lidar synced cache |
| `filter_infos` | `06_filter_infos_with_osm_gt.py` | Loc frame co GT vector de train |
| `make_config` | `07_make_phenikaa_train_config.py` | Sinh config train/infer |
| `make_init` | `08_make_12cam_init_checkpoint.py` | Tao init checkpoint 12cam/6class |
| `check_dataset` | `09_check_train_dataset_osm.py` | Check dataset truoc train |
| `train` | `10_train_maptr_phenikaa_osm.py` | Train model |
| `infer` | `11_infer_visualize_maptr_phenikaa.py` | Inference, tao predictions |
| `export` | `22_export_prediction_osm_graph.py` | Export OSM graph |
| `export` | `21_postprocess_predictions_to_thin_osm.py` | Lam mong/thin OSM |
| `visualize` | `16_visualize_maptr_paper_style.py` | Visualize debug/paper-style |
| `visualize` | `17_visualize_maptr_professional.py` | Visualize professional |

File ho tro bat buoc:

```text
30_run_pipeline_from_config.py
pipeline_config.py
common.py
config/pipeline.yaml
config/maptr_tiny_r50_phenikaa_12cam.py
config/maptr_tiny_r50_phenikaa_12cam_train_osm.py
```

## File nao da dua ra khoi pipeline

Cac file cu/optional da duoc chuyen vao:

```text
_legacy_removed_from_pipeline/
```

Trong do co:

- `_archive_old_experiments/`: cac pipeline cu, lidar geometry thu nghiem, legacy scripts.
- `13_evaluate_maptr_predictions.py`: chi can khi tinh metric co GT, khong nam trong pipeline tu dump hien tai.
- `20_run_after_train_clean.sh`: script shell cu, da thay bang `30_run_pipeline_from_config.py`.
- cac README/notes cu da thay bang docs chuan o `/docs`.

Thu muc legacy nay khong duoc pipeline hien tai goi. Neu chac chan khong can lich su, co the xoa han sau.

## Config

Chi sua config source trong:

```text
config/pipeline.yaml
config/maptr_tiny_r50_phenikaa_12cam.py
config/maptr_tiny_r50_phenikaa_12cam_train_osm.py
```

Config copy trong `outputs/.../work_dirs/` chi la snapshot cua lan train.
