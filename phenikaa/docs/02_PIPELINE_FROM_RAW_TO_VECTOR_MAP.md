# Pipeline Tu Raw Data Den Vector HD Map

Lenh tong:

```bash
cd /home/khanh247/Documents/Survey/phd/phenikaa
/home/khanh247/miniconda3/envs/maptr/bin/python scripts/maptr/30_run_pipeline_from_config.py --steps <steps>
```

Config tong nam o:

```text
configs/maptr/pipeline.yaml
```

## Thu Tu Chay

| Step | Script | Tac dung | Dau ra chinh |
|---|---|---|---|
| `raw_qa` | `01_check_raw_scenario.py` | Kiem tra raw camera/lidar/pose/timestamp | Bao cao QA tren terminal |
| `pose_qa` | `02_check_pose_quality.py` | Kiem tra pose co nhay bat thuong khong | Bao cao pose QA |
| `gt_qa` | `04_check_gt_osm_semantic.py` | Kiem tra OSM GT theo rule | Bao loi tag/class neu co |
| `build_infos` | `03_build_phenikaa_infos.py` | Dong goi sample MapTR tu raw dump | `outputs/only_camera/<scenario>/phenikaa_maptr_infos_full.pkl` |
| `filter_infos` | `06_filter_infos_with_osm_gt.py` | Giu sample co GT vector hop le | `*_gt_positive.pkl` |
| `make_config` | `07_make_phenikaa_train_config.py` | Cap nhat config train 6 class | `configs/maptr/*train_osm.py` |
| `make_init` | `08_make_12cam_init_checkpoint.py` | Tao init checkpoint, bo head cu neu can train lai | `checkpoints/*init_12cam_partial.pth` |
| `check_dataset` | `09_check_train_dataset_osm.py` | Smoke test dataset truoc train | Log dataset/class count |
| `train` | `10_train_maptr_phenikaa_osm.py` | Train MapTR | `latest.pth`, `epoch_*.pth`, `.log` |
| `infer` | `11_infer_visualize_maptr_phenikaa.py` | Du doan vector map | `predictions.json`, anh quick view |
| `export` | `22_*`, `21_*` | Export OSM graph va thin OSM | `predicted_vector_map_*.osm` |
| `visualize` | `17_visualize_maptr_professional.py` | Hinh tong quan/bao cao | `visualizations/professional/*.png` |

## Vi Du

Infer/export/visualize sau khi train xong:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python scripts/maptr/30_run_pipeline_from_config.py --steps infer,export,visualize
```

Chi tao lai OSM thin tu prediction da co:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python scripts/maptr/30_run_pipeline_from_config.py --steps export
```
