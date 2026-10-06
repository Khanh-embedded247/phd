# Phenikaa MapTR Step-by-Step Pipeline

Tai lieu nay ghi cach chay tung file bang lenh:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python <file.py>
```

Moi file tu doc `config/pipeline.yaml`, nen binh thuong khong can truyen tham so dai.

## 1. Chuan du lieu vao

Moi scenario nen co cau truc:

```text
data/<SCENARIO>/
  CAMERA/
    CAM_P_F/*.jpg
    ...
    CAM_F_B/*.jpg
  dump/
    frames/laz/*.laz
    traj_lidar.txt
  <scenario>.osm
```

GT OSM phai ve theo:

```text
GT_OSM_DRAWING_GUIDE.md
```

MapTR chi train 6 class nhin thay duoc:

```text
lane_divider
road_edge_marking
stop_line
ped_crossing
boundary
speed_bump
```

Nhung class topology nhu `virtual_*`, `centerline`, `virtual_lanelet`,
`drivable_lane` van giu trong OSM nhung khong train perception.

## 2. Sua config chung

File:

```text
config/pipeline.yaml
```

Can chu y cac block:

```text
scenario: scenario dang inference/check hien tai
paths: config, infos, gt_osm, checkpoint
train_data.datasets: danh sach scenario dung de train chung
train: tham so train
```

Multi-dataset train khong gop OSM. Moi dataset giu cap:

```text
infos.pkl + gt.osm
```

vi moi scenario co he toa do map/trajectory rieng.

## 3. Thu tu chay tung file

### 3.1 Check raw camera/LiDAR/trajectory

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/01_check_raw_scenario.py
```

Buoc nay kiem tra:

```text
12 camera co du anh khong
timestamp camera lech bao nhieu
anh co doc duoc khong
LiDAR laz co ton tai khong
traj_lidar.txt co ton tai khong
```

Output:

```text
${FINAL_DIR}/qa/raw_scenario_report.json
```

### 3.2 Check pose quality

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/02_check_pose_quality.py
```

Buoc nay kiem tra:

```text
dt giua pose
speed bat thuong
yaw-rate bat thuong
pose jump
```

Output:

```text
${FINAL_DIR}/qa/pose_quality_report.json
${FINAL_DIR}/qa/pose_quality_steps.csv
```

### 3.3 Check GT OSM semantic

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/04_check_gt_osm_semantic.py
```

Buoc nay dem 6 class train, dem class topology bi ignore, va canh bao:

```text
missing semantic_class
unknown semantic_class
legacy semantic class
bad node refs
```

Output:

```text
${FINAL_DIR}/qa/gt_osm_semantic_report.json
```

### 3.4 Build infos, undistort camera, sync LiDAR

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/03_build_phenikaa_infos.py
```

Buoc nay lam:

```text
chon CAM_P_F lam reference time
sync 12 camera theo timestamp
undistort anh va ghi K moi vao pkl
warp LiDAR ve timestamp camera
tao infos.pkl cho MapTR
```

Output:

```text
outputs/benchmark_maptr/<SCENARIO>/phenikaa_maptr_infos_full.pkl
outputs/benchmark_maptr/<SCENARIO>/images_undistorted/
outputs/benchmark_maptr/<SCENARIO>/lidar_synced_to_camera/
```

### 3.5 Tao config train multi-dataset 6 class

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/07_make_phenikaa_train_config.py
```

Buoc nay tao:

```text
config/maptr_tiny_r50_phenikaa_12cam_train_osm.py
```

Neu `train_data.datasets` co nhieu scenario, config se dung:

```text
ConcatDataset
RepeatDataset neu repeat > 1
```

### 3.6 Tao init checkpoint 12 camera / 6 class

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/08_make_12cam_init_checkpoint.py
```

Buoc nay doc checkpoint MapTR goc, giu weight nao khop shape, bo head/class
cu khong khop 6 class.

Output:

```text
${INIT_CHECKPOINT}
```

### 3.7 Check train dataset

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/09_check_train_dataset_osm.py
```

Buoc nay build dataset tu config va in so GT vector tung leaf dataset.

### 3.8 Train

Truoc khi train that, bat trong YAML:

```yaml
train:
  enabled: true
```

Chay:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/10_train_maptr_phenikaa_osm.py
```

Output:

```text
outputs/benchmark_maptr/<SCENARIO>/work_dirs/maptr_12cam_osm/latest.pth
```

### 3.9 Inference

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/11_infer_visualize_maptr_phenikaa.py
```

Output:

```text
${INTERMEDIATE_DIR}/predictions.json
${INTERMEDIATE_DIR}/predictions.pkl
```

### 3.10 Export OSM

Graph OSM, nen xem truoc:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/22_export_prediction_osm_graph.py
```

Thin OSM, gom mong hon:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/21_postprocess_predictions_to_thin_osm.py
```

### 3.11 Visualize

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/16_visualize_maptr_paper_style.py
```

## 4. Lenh tong thay cho chay tung file

Dry-run de xem lenh:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/30_run_pipeline_from_config.py \
--steps train_all --dry-run
```

Train pipeline:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/30_run_pipeline_from_config.py \
--steps train_all
```

Deploy/inference pipeline:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/30_run_pipeline_from_config.py \
--steps deploy
```
