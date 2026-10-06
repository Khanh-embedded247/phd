# Phenikaa MapTR Pipeline, Output, and Cleanup Guide

Tai lieu nay giai thich pipeline hien tai tu raw `dump/` den train/inference/export/visualize, va file nao co the giu/xoa de thu muc output gon hon.

## 1. File Python nao can giu

### Bat buoc cho pipeline hien tai

| File | Vai tro |
|---|---|
| `30_run_pipeline_from_config.py` | Orchestrator chay pipeline theo `config/pipeline.yaml` |
| `pipeline_config.py` | Doc va expand bien trong YAML cho cac script le |
| `common.py` | Ham chung doc timestamp/trajectory, duoc dung boi QA/build infos |
| `01_check_raw_scenario.py` | QA raw data: camera, timestamp, anh |
| `02_check_pose_quality.py` | QA trajectory/pose |
| `03_build_phenikaa_infos.py` | Tao `infos.pkl`, anh undistorted, lidar synced neu can |
| `04_check_gt_osm_semantic.py` | QA tag GT OSM |
| `06_filter_infos_with_osm_gt.py` | Tao infos chi giu frame co GT vector de train nhanh hon |
| `07_make_phenikaa_train_config.py` | Sinh config train MapTR 12 camera/6 class |
| `08_make_12cam_init_checkpoint.py` | Tao checkpoint init 12 camera, bo head cu |
| `09_check_train_dataset_osm.py` | Check dataset train truoc khi train that |
| `10_train_maptr_phenikaa_osm.py` | Train MapTR |
| `11_infer_visualize_maptr_phenikaa.py` | Inference va tao prediction raw/intermediate |
| `22_export_prediction_osm_graph.py` | Export prediction raw thanh OSM graph |
| `21_postprocess_predictions_to_thin_osm.py` | Lam mong/dedup prediction thanh OSM thin |
| `16_visualize_maptr_paper_style.py` | Visualize debug/paper-style cu |
| `17_visualize_maptr_professional.py` | Visualize professional: global overview + dashboard frame |

### Tuy chon

| File | Co the xoa/bo qua khi nao |
|---|---|
| `13_evaluate_maptr_predictions.py` | Chi can khi tinh metric voi GT. Neu chua dung metric thi optional |
| `20_run_after_train_clean.sh` | Script shell cu sau train. Pipeline chinh da thay bang `30_run_pipeline_from_config.py` |
| `_archive/` | Script cu/thu nghiem. Nen giu neu can truy vet, khong can cho pipeline hien tai |
| `00_README_RUN_THIS.md`, `PIPELINE_STEP_BY_STEP.md`, `MASTER_THESIS_NOTES_AND_QA.md` | Tai lieu hoc/luan van, khong anh huong runtime |

Khong nen xoa `config/maptr_tiny_r50_phenikaa_12cam.py` vi `07_make_phenikaa_train_config.py` dung no lam base config. Khong nen xoa `config/maptr_tiny_r50_phenikaa_12cam_train_osm.py` vi day la config model dang train/infer.

## 2. Thu tu chay tu raw dump

Raw data dau vao moi scenario:

```text
data/<scenario>/
  CAMERA/<camera_name>/*.jpg
  dump/frames/laz/*.laz
  dump/traj_lidar.txt
  <scenario>.osm hoac file GT .osm tuong ung
```

### Cach chay gop bang orchestrator

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/30_run_pipeline_from_config.py --steps raw_qa,pose_qa,gt_qa,build_infos,filter_infos,make_config,make_init,check_dataset,train,infer,export,visualize
```

### Cac buoc va output

| Buoc | Script | Dau vao | Dau ra | Bat buoc? |
|---|---|---|---|---|
| raw_qa | `01_check_raw_scenario.py` | `CAMERA/`, `dump/` | `final_run/.../qa/raw_scenario_report.json` | Nen co |
| pose_qa | `02_check_pose_quality.py` | `dump/traj_lidar.txt` | `qa/pose_quality_report.json`, `.csv` | Nen co |
| gt_qa | `04_check_gt_osm_semantic.py` | GT `.osm` | `qa/gt_osm_semantic_report.json` | Nen co neu train |
| build_infos | `03_build_phenikaa_infos.py` | camera, traj, lidar raw | `phenikaa_maptr_infos_full.pkl`, `images_undistorted/`, `lidar_synced_to_camera/` | Co |
| filter_infos | `06_filter_infos_with_osm_gt.py` | `infos.pkl`, GT `.osm` | `phenikaa_maptr_infos_full_gt_positive.pkl` | Chi can train |
| make_config | `07_make_phenikaa_train_config.py` | YAML, infos, GT | `config/maptr_tiny_r50_phenikaa_12cam_train_osm.py` | Can train/infer |
| make_init | `08_make_12cam_init_checkpoint.py` | checkpoint MapTR goc | `ckpts/maptr_init_12cam_partial.pth` | Chi can train lai |
| check_dataset | `09_check_train_dataset_osm.py` | train config | log check dataset | Nen co truoc train |
| train | `10_train_maptr_phenikaa_osm.py` | train config, init ckpt | `work_dirs/.../epoch_*.pth`, log | Chi khi hoc lai |
| infer | `11_infer_visualize_maptr_phenikaa.py` | checkpoint, infos, anh | `_intermediate/`, `predictions.json/pkl` | Can de predict |
| export graph | `22_export_prediction_osm_graph.py` | predictions, infos | `predicted_vector_map_graph.osm` | Can de xem raw vector map |
| export thin | `21_postprocess_predictions_to_thin_osm.py` | predictions, infos | `predicted_vector_map_thin.osm` | Can de co HD map gon |
| visualize | `16`, `17` | predictions, OSM, infos, anh | `visualizations/` | Can cho bao cao/demo |

## 3. LiDAR lech timestamp co bat buoc tao file moi khong?

Voi MapTR hien tai, model train/infer la camera-only. No can:

- anh 12 camera,
- calibration,
- pose ego/base_link,
- GT OSM khi train.

No khong truc tiep dung point cloud LiDAR lam input model.

Tuy nhien `03_build_phenikaa_infos.py` van tao `lidar_synced_to_camera/` de:

- giu `pts_filename` hop voi format MMDetection3D/MapTR,
- co point cloud dung timestamp camera de QA/visualize/fusion sau nay,
- tranh sai lech khi sau nay mo rong camera+LiDAR.

Neu chi chay camera-only va can tiet kiem dung luong, co the coi `lidar_synced_to_camera/` la cache co the tai tao. Khong nen xoa neu ban dang dung Vector Map Builder/QA point cloud tu output nay.

## 4. Output nao giu, output nao co the xoa

### Nen giu de mang model sang may khac

| Path | Ly do |
|---|---|
| `work_dirs/maptr_12cam_osm/epoch_4.pth` | Bo nao model da hoc |
| `config/maptr_tiny_r50_phenikaa_12cam_train_osm.py` | Kien truc/model config dung voi checkpoint |
| `config/pipeline.yaml` | Duong dan/tham so pipeline |
| `predicted_vector_map_thin.osm` | Ket qua HD map gon |
| `predicted_vector_map_graph.osm` | Ket qua prediction graph de debug/so sanh |
| `phenikaa_maptr_infos_full.pkl` | Can neu infer/visualize lai tren scenario nay |

### Co the xoa de giam dung luong neu khong train/infer lai

| Path | Dung luong hien tai | Ghi chu |
|---|---:|---|
| `images_undistorted/` | ~18GB | Cache anh da undistort. Co the tao lai tu raw camera |
| `lidar_synced_to_camera/` | ~3.4GB | Cache LiDAR synced. Co the tao lai tu raw lidar+traj |
| `_intermediate/` | ~624MB | Raw prediction tung frame + BEV debug. Xoa thi muon export/visualize lai phai infer lai hoac giu `predictions.json` |
| `epoch_2.pth`, `epoch_3.pth` | ~826MB | Checkpoint cu. Giu `epoch_4.pth` la du neu chi dung model tot nhat/cuoi |
| `ckpts/maptr_init_12cam_partial.pth` | ~113MB | Chi can neu train lai tu init |
| `*.log`, `*.log.json` | vai MB | Giu log cuoi de viet bao cao; log thu nghiem co the xoa |
| `phenikaa_maptr_infos_full_gt_positive.pkl` | ~22MB | Chi can train lai |

### Khong nen xoa neu con muon debug/develop

| Path | Ly do |
|---|---|
| `final_run/no_metric_full_data/predictions.json` | Can export OSM lai nhanh, khong can infer lai |
| `final_run/no_metric_full_data/predictions.pkl` | Raw prediction dang Python/pickle, huu ich neu viet evaluator |
| `visualizations/professional/` | Anh bao cao/demo moi |
| `qa/` | Bang chung data/pose/GT hop le cho luan van |

## 5. Pipeline gon khuyen nghi sau khi train xong

Neu muc tieu la luu thanh qua cuoi cung, giu:

```text
work_dirs/maptr_12cam_osm/epoch_4.pth
phenikaa_maptr_infos_full.pkl
final_run/no_metric_full_data/predictions.json
final_run/no_metric_full_data/predicted_vector_map_graph.osm
final_run/no_metric_full_data/predicted_vector_map_thin.osm
final_run/no_metric_full_data/qa/
final_run/no_metric_full_data/visualizations/professional/
config/pipeline.yaml
config/maptr_tiny_r50_phenikaa_12cam_train_osm.py
```

Neu can tiet kiem toi da va chi mang sang may khac de infer du lieu moi, giu:

```text
epoch_4.pth
config/maptr_tiny_r50_phenikaa_12cam_train_osm.py
benchmark_maptr source code
vendor/MapTR source code
environment/conda tuong thich
```

Du lieu moi tren may khac van phai build `infos.pkl` rieng tu camera/calib/pose cua may do.
