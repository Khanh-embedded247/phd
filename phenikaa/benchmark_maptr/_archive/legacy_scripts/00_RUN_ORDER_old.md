# Phenikaa MapTR - Thu Tu Chay De Khoi Roi

File nay la ban rut gon. Neu chi muon train/test MapTR voi GT OSM va chay thu tren doan Normal khong GT, hay chay theo dung thu tu duoi day.

## 0. Y nghia cac loai chay

- **Train**: hoc tu cac frame co GT trong `normal_main.osm`.
- **Test co GT**: chay tren frame co GT de tinh metric. Neu dung lai dung frame train thi chi la sanity check, chua phai test doc lap.
- **Inference khong GT**: chay tren frame Normal ngoai vung OSM, khong tinh metric duoc, chi xem bang mat va xuat predicted map `.osm`.

## 1. Kiem tra frame nao co GT

Chay:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
  /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/18_audit_gt_coverage.py
```

Output:

```text
phd/phenikaa/outputs/benchmark_maptr/Normal/gt_coverage.csv
```

Doc cot:

- `has_gt = 1`: frame nam trong vung da ve OSM, co the dung train/test metric.
- `has_gt = 0`: frame khong co GT, chi dung inference bang mat.
- `gt_count`: so vector GT sinh ra tai frame do.

## 2. Tao train infos tu doan co GT

Da co san file:

```text
phd/phenikaa/outputs/benchmark_maptr/Normal/phenikaa_maptr_infos_train_osm.pkl
```

Neu muon tao lai:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
  /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/03_build_phenikaa_infos.py \
  --split train \
  --start-index 220 \
  --max-samples 300 \
  --out-pkl /home/khanh247/Documents/Survey/phd/phenikaa/outputs/benchmark_maptr/Normal/phenikaa_maptr_infos_train_osm.pkl
```

## 3. Tao config train

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
  /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/07_make_phenikaa_train_config.py
```

Output:

```text
phd/phenikaa/benchmark_maptr/config/maptr_tiny_r50_phenikaa_12cam_train_osm.py
```

## 4. Kiem tra train dataset co GT

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
  /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/09_check_train_dataset_osm.py \
  --config /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/config/maptr_tiny_r50_phenikaa_12cam_train_osm.py
```

Neu nhieu frame `gt_vectors > 0` la OK.

## 5. Train

```bash
bash /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/10_train_maptr_phenikaa_osm.sh
```

Checkpoint sau train:

```text
phd/phenikaa/outputs/benchmark_maptr/Normal/work_dirs/maptr_12cam_osm/latest.pth
```

## 6. Cach chay moi sau khi train: chi dung file 20

Sau khi da train xong bang file `10_train_maptr_phenikaa_osm.sh`, hay uu tien dung script nay:

```bash
bash /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/20_run_after_train_clean.sh
```

Mac dinh script chay:

```text
MODE=no_metric
NUM_SAMPLES=200
```

Nghia la: chay model tren 200 frame Normal, khong can GT, xuat vector map `.osm` va anh visualize de kiem tra nhanh.

Neu muon chay full Normal, go ro:

```bash
NUM_SAMPLES=0 bash /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/20_run_after_train_clean.sh
```

Full Normal hien co khoang 8704 frame, co the mat vai gio tren RTX 4060.

Output sach:

```text
phd/phenikaa/outputs/benchmark_maptr/Normal/final_run/no_metric_start_0_n_200/
  predictions.json
  predictions.pkl
  predicted_vector_map.osm
  visualizations/
  README_RESULT.txt
```

Neu muon chay co metric tren vung co GT:

```bash
MODE=metric bash /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/20_run_after_train_clean.sh
```

Output:

```text
phd/phenikaa/outputs/benchmark_maptr/Normal/final_run/metric_with_gt/
  predictions.json
  predicted_vector_map.osm
  visualizations/
  evaluation_metrics.csv
  evaluation_metrics.summary.json
```

Tu day tro di, cac file `11`, `13`, `15`, `16` chi la file con duoc script `20` goi noi bo. Neu khong debug thi khong can mo.

## 7. Test nhanh tren frame co GT theo cach cu

Day la sanity check. Neu config dang tro vao `phenikaa_maptr_infos_train_osm.pkl` thi no la test tren train.

```bash
bash /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/14_scan_inference_segments.sh
```

Output:

```text
phd/phenikaa/outputs/benchmark_maptr/Normal/inference_scan
```

Metric:

```text
evaluation_metrics.csv
evaluation_metrics.summary.json
```

## 8. Chay thuc te tren doan Normal khong GT theo cach cu

Day la buoc ban dang can: train bang doan co GT, roi dua model ra doan Normal khong GT xem no ve map ra sao.

```bash
bash /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/17_infer_normal_unlabeled.sh
```

Output:

```text
phd/phenikaa/outputs/benchmark_maptr/Normal/inference_unlabeled_val
phd/phenikaa/outputs/benchmark_maptr/Normal/paper_style_unlabeled_val
phd/phenikaa/outputs/benchmark_maptr/Normal/predicted_vector_map_unlabeled_val.osm
```

Y nghia:

- `inference_unlabeled_val/predictions.json`: vector prediction tung frame.
- `paper_style_unlabeled_val/*.jpg`: anh 12 camera + BEV prediction de xem bang mat.
- `predicted_vector_map_unlabeled_val.osm`: vector map raw trong he map, chua phai Lanelet2 hoan chinh.

## 9. Xuat predicted OSM tu ket qua co san

Neu da co `predictions.json` va chi muon xuat lai OSM:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python \
  /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/15_export_prediction_osm.py \
  --pred-dir /home/khanh247/Documents/Survey/phd/phenikaa/outputs/benchmark_maptr/Normal/inference_unlabeled_val \
  --infos /home/khanh247/Documents/Survey/phd/phenikaa/outputs/benchmark_maptr/Normal/phenikaa_maptr_infos_val.pkl \
  --out-osm /home/khanh247/Documents/Survey/phd/phenikaa/outputs/benchmark_maptr/Normal/predicted_vector_map_unlabeled_val.osm
```

## 10. Cac file nao can quan tam

File chinh:

- `03_build_phenikaa_infos.py`: tao input `.pkl` cho MapTR.
- `07_make_phenikaa_train_config.py`: tao config train.
- `10_train_maptr_phenikaa_osm.sh`: train model.
- `11_infer_visualize_maptr_phenikaa.py`: inference chung.
- `13_evaluate_maptr_predictions.py`: tinh metric khi co GT.
- `15_export_prediction_osm.py`: xuat prediction sang `.osm`.
- `16_visualize_maptr_paper_style.py`: tao anh visualize.
- `17_infer_normal_unlabeled.sh`: chay thuc te ngoai GT.
- `18_audit_gt_coverage.py`: kiem tra frame nao co GT.
- `20_run_after_train_clean.sh`: script chinh sau train, nen dung file nay de bot roi.

File phu/debug:

- `02_check_infos.py`
- `04_check_maptr_dataset_load.py`
- `05_make_phenikaa_maptr_config.py`
- `06_check_maptr_model_build.py`
- `08_make_12cam_init_checkpoint.py`
- `09_check_train_dataset_osm.py`
- `12_infer_visualize_maptr_phenikaa.sh`
- `14_scan_inference_segments.sh`

## 11. Ket luan hien tai

Hien tai ban da:

1. Co GT OSM: `data/Normal/normal_main.osm`.
2. Train duoc checkpoint 12 camera.
3. Test/sanity check tren frame co GT.
4. Chay inference tren doan Normal khong GT.
5. Xuat duoc predicted vector map raw `.osm`.

Viec tiep theo moi la hau xu ly thanh Lanelet2 day du:

- noi line trung nhau,
- lam muot,
- tach boundary trai/phai,
- tao centerline,
- tao relation lanelet,
- xuat Lanelet2 `.osm` dung chuan.
