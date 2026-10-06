# Pipeline MapTR Phenikaa

Muc tieu: dung 12 camera Phenikaa de train MapTR voi GT `normal_main.osm`, sau do chay thuc te tren data `Normal` va xuat vector HD map `.osm`.

Tai lieu nen doc truoc:

```text
GT_OSM_DRAWING_GUIDE.md          # quy uoc ve GT OSM
PIPELINE_STEP_BY_STEP.md         # thu tu chay tung file python
MASTER_THESIS_NOTES_AND_QA.md    # giai thich de tai + cau hoi bao ve
```

## MapTR vendor

Phan code MapTR can de train/inference da duoc copy vao:

```text
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/vendor/MapTR/
```

Pipeline chinh trong thu muc nay khong con can chay tu repo MapTR ngoai nua.

Ben trong `vendor/MapTR` co cac phan can thiet:

```text
projects/
mmdetection3d/
tools/
README.md
LICENSE
requirement.txt
```

## Viec vua chot ve undistort

Anh dua vao MapTR la anh da undistort, nam trong:

```text
/home/khanh247/Documents/Survey/phd/phenikaa/outputs/benchmark_maptr/Normal/images_undistorted/
```

Code tao anh nay la:

```text
03_build_phenikaa_infos.py
-> common.py::undistort_for_projection()
```

Hien tai dang dung:

```text
CAM_P_*  -> cv2.undistort, PINHOLE_UNDISTORT_BALANCE = 1.0
CAM_F_*  -> cv2.fisheye.*, FISHEYE_UNDISTORT_BALANCE = 0.0
```

Sau khi sua balance/calib/sync, phai tao lai `.pkl` de anh trong `images_undistorted` duoc ghi lai. Khong can xoa thu muc bang tay, script se ghi de anh cung ten.

## Viec vua chot ve frame ego/base_link

`03_build_phenikaa_infos.py` bay gio dat ego frame la `base_link` thuc te cua xe,
khong con tam coi ego = `lidar_top`.

Cu the:

```text
traj_lidar.txt        : T_map_lidar_top
calib LIDAR_TOP       : T_base_link_lidar_top
ego2global trong pkl  : T_map_base_link
lidar2ego trong pkl   : T_base_link_lidar_top
```

Vi vay sau khi sua frame nay, can tao lai infos train va train lai checkpoint.
Checkpoint cu neu da train tren infos ego=`lidar_top` thi prediction/visualize co
the van bi lech theo goc LiDAR top.

## File can giu

### 1. Tao manifest va anh undistorted

```text
03_build_phenikaa_infos.py
```

File nay doc data `Normal`, dong bo 12 camera, doc pose/lidar, tao `.pkl` theo format MapTR va ghi anh undistorted.

Output quan trong:

```text
phenikaa_maptr_infos_train_osm.pkl
phenikaa_maptr_infos_full_normal.pkl
images_undistorted/
lidar_synced_to_camera/
```

Ghi chu: `.pkl` khong phai GT. No la danh sach sample: 12 anh, calib, pose, lidar path, timestamp. Train va inference deu can `.pkl`.

### 2. Tao config train

```text
07_make_phenikaa_train_config.py
```

Tao config MapTR 12 camera, tro toi:

```text
normal_main.osm
phenikaa_maptr_infos_train_osm.pkl
```

Output:

```text
config/maptr_tiny_r50_phenikaa_12cam_train_osm.py
```

### 3. Tao checkpoint khoi tao 12 camera

```text
08_make_12cam_init_checkpoint.py
```

Dung khi train tu dau voi 12 camera. File nay tao checkpoint init phu hop shape 12 camera/class hien tai.

### 4. Kiem tra GT OSM

```text
09_check_train_dataset_osm.py
```

Dung de xem MapTR co doc duoc `normal_main.osm` thanh vector GT khong.

### 5. Train

```text
10_train_maptr_phenikaa_osm.sh
```

Output checkpoint:

```text
outputs/benchmark_maptr/Normal/work_dirs/maptr_12cam_osm/latest.pth
```

### 6. Inference tung frame

```text
11_infer_visualize_maptr_phenikaa.py
```

File nay chay model tren `.pkl`, xuat:

```text
predictions.json
predictions.pkl
pred_bev.png / gt_bev.png / overlay_bev.png neu co GT
```

Thuong khong can chay rieng vi `20_run_after_train_clean.sh` se goi no.

### 7. Metric khi co GT

```text
13_evaluate_maptr_predictions.py
```

Chi can khi chay `MODE=metric`. Neu khong co GT thi bo qua.

### 8. Visualize kieu paper

```text
16_visualize_maptr_paper_style.py
```

Tao anh de xem bang mat:

```text
bev_map/
camera_context/
```

`camera_context` co 12 anh camera va vector prediction ve lai len anh.

### 9. Script chinh sau train

```text
20_run_after_train_clean.sh
```

Day la file nen chay sau khi da co checkpoint. No gom cac buoc:

```text
inference -> export predictions.json -> export OSM -> visualize -> metric neu bat MODE=metric
```

### 10. Export OSM

```text
21_postprocess_predictions_to_thin_osm.py
22_export_prediction_osm_graph.py
```

`21` tao ban rat gon:

```text
predicted_vector_map_thin.osm
```

`22` giu hinh hoc prediction ro hon, nen uu tien xem file nay:

```text
predicted_vector_map_graph.osm
```

## Thu tu chay khuyen nghi

### Cach moi: chay bang 1 file YAML

Neu chi muon doi duong dan data, scenario, checkpoint, so frame, score threshold,
output dir, hay tham so export/visualize thi sua file chung:

```text
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/config/pipeline.yaml
```

Sau do neu muon chay TOAN BO pipeline:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/30_run_pipeline_from_config.py
```

File tren doc `config/pipeline.yaml` va chay lan luot:

```text
03_build_phenikaa_infos.py
11_infer_visualize_maptr_phenikaa.py
22_export_prediction_osm_graph.py
21_postprocess_predictions_to_thin_osm.py
16_visualize_maptr_paper_style.py
```

Neu muon chay tung buoc rieng thi chay truc tiep file goc. Cac file nay cung
tu doc `config/pipeline.yaml`, nen binh thuong khong can truyen tham so dai:

```bash
python3 /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/03_build_phenikaa_infos.py
python3 /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/11_infer_visualize_maptr_phenikaa.py
python3 /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/22_export_prediction_osm_graph.py
python3 /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/21_postprocess_predictions_to_thin_osm.py
python3 /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/16_visualize_maptr_paper_style.py
```

CLI args chi can dung khi muon override tam thoi gia tri trong YAML.

Kiem tra lenh se chay ma khong thuc thi:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/30_run_pipeline_from_config.py \
--steps all \
--dry-run
```

`--steps` chi la tuy chon nang cao neu muon bat file tong chi chay mot vai buoc.
Binh thuong khong can dung.

Output chinh theo YAML mac dinh:

```text
/home/khanh247/Documents/Survey/phd/phenikaa/outputs/benchmark_maptr/road_work_traffic/final_run/no_metric_full_data/
```

### A. Sau khi sua undistort balance

Chay lai infos de ghi de anh undistorted bang balance moi:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/03_build_phenikaa_infos.py \
--scenario Normal \
--split train_osm \
--max-samples 0 \
--out-pkl /home/khanh247/Documents/Survey/phd/phenikaa/outputs/benchmark_maptr/Normal/phenikaa_maptr_infos_train_osm.pkl
```

Neu muon tao full data cho inference:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/03_build_phenikaa_infos.py \
--scenario Normal \
--split full \
--max-samples 0 \
--out-pkl /home/khanh247/Documents/Survey/phd/phenikaa/outputs/benchmark_maptr/Normal/phenikaa_maptr_infos_full_normal.pkl
```

### B. Train tu dau

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/07_make_phenikaa_train_config.py
/home/khanh247/miniconda3/envs/maptr/bin/python /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/08_make_12cam_init_checkpoint.py
/home/khanh247/miniconda3/envs/maptr/bin/python /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/09_check_train_dataset_osm.py
bash /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/10_train_maptr_phenikaa_osm.sh
```

### C. Chay inference khong metric, xem bang mat

Chay nhanh 200 frame:

```bash
bash /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/20_run_after_train_clean.sh
```

Ep tao lai infos/anh undistorted truoc khi inference:

```bash
REBUILD_INFOS=1 bash /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/20_run_after_train_clean.sh
```

Chay full Normal:

```bash
NUM_SAMPLES=0 bash /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/20_run_after_train_clean.sh
```

Chay thu scenario khac bang checkpoint da train tren `Normal`, vi du `road_work_traffic`:

```bash
SCENARIO=road_work_traffic \
CHECKPOINT_SCENARIO=Normal \
REBUILD_INFOS=1 \
NUM_SAMPLES=200 \
bash /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/20_run_after_train_clean.sh
```

Neu muon chay het `road_work_traffic`:

```bash
SCENARIO=road_work_traffic \
CHECKPOINT_SCENARIO=Normal \
REBUILD_INFOS=1 \
NUM_SAMPLES=0 \
bash /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/20_run_after_train_clean.sh
```

### D. Chay metric voi GT

```bash
MODE=metric REBUILD_INFOS=1 bash /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/20_run_after_train_clean.sh
```

## Output can xem

Thu muc chinh:

```text
/home/khanh247/Documents/Survey/phd/phenikaa/outputs/benchmark_maptr/Normal/final_run/
```

Neu chay mac dinh 200 frame:

```text
final_run/no_metric_start_0_n_200/
```

File quan trong:

```text
predictions.json
predicted_vector_map_graph.osm
predicted_vector_map_thin.osm
bev_map/
camera_context/
README_RESULT.txt
```

Uu tien xem:

```text
camera_context/
predicted_vector_map_graph.osm
```

`_intermediate/` chi la thu muc phu de cac script doc lai, khong phai output chinh.

## File khong can dung hang ngay

```text
config/maptr_tiny_r50_phenikaa_12cam.py
```

Config cu/khong GT, giu lai de tham khao.

```text
_archive/legacy_scripts/
```

Script cu/debug, khong nam trong pipeline chinh.

```text
_archive/lidar_geometry_pipeline/
```

Pipeline LiDAR/geometry cu tu buoc 01-10. Da dua vao archive de tranh nham voi pipeline MapTR hien tai.
