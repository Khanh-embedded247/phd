# Chạy B3 trên Phenikaa

B3 theo `docs/LO_TRINH.md` nghĩa là: đưa dữ liệu Phenikaa RESIDENTIAL vào pipeline MapTR đã kiểm tra ở B1/B2.

- `b1_camera`: MapTRv2 camera-only, dùng để smoke test/so sánh.
- `b2_fusion`: MapTR camera+LiDAR fusion từ B2, là nhánh chính cho B3 theo lộ trình hiện tại.

Lưu ý: checkpoint B2 fusion có sẵn là checkpoint nuScenes 6-camera. Vì vậy zero-shot trên Phenikaa nên chạy `phenikaa6`. Muốn dùng đủ 10 camera đúng nghĩa cần train/fine-tune checkpoint với cấu hình 10 camera Phenikaa.

## 0. Kiểm tra B0

```bash
cd /home/khanh247/Documents/Survey/phenikaa
conda activate phenikaa
python tutorials/05_load_pointcloud_information.py
python tutorials/11_all_things.py
```

Kỳ vọng: LiDAR/box project khớp camera. B0 chỉ kiểm tra calib, chưa dự đoán lane.

## 1. Chạy B1 trên nuScenes

```bash
cd /home/khanh247/Documents/Survey/phenikaa
conda activate maptr
python scripts/run_b1_eval.py
```

Kết quả:

```text
outputs/nusc_eval/e1_camera/
```

B1 dùng:

```text
projects/configs/maptrv2/maptrv2_nusc_r50_24ep_eval_8g.py
ckpts/maptrv2_nusc_r50_24ep.pth
```

Đây là camera-only.

## 2. Chạy B2 trên nuScenes

```bash
cd /home/khanh247/Documents/Survey/phenikaa
conda activate maptr
python scripts/run_b2_eval.py
```

Kết quả:

```text
outputs/nusc_eval/e2_fusion/
```

B2 dùng:

```text
projects/configs/maptr/maptr_tiny_fusion_eval_8g.py
ckpts/maptr_tiny_fusion_24e.pth
```

Đây mới là camera+LiDAR fusion: config có `use_lidar=True`, `use_camera=True`, `modality='fusion'`, `lidar_encoder`, và `ConvFuser`.

## 3. Chuẩn bị Phenikaa infos

```bash
cd /home/khanh247/Documents/Survey/phenikaa
conda activate phenikaa
python -m phenikaa_hdmap.converters.build_phenikaa_infos
```

Kết quả:

```text
data/phenikaa/infos_residential.pkl
outputs/phenikaa_vis/infos_summary.txt
```

## 4. Kiểm tra môi trường B3 fusion

B3 fusion cần CUDA và cần đọc được `.laz`.

```bash
cd /home/khanh247/Documents/Survey/phenikaa
conda activate maptr
python - <<'PY'
import torch, laspy
print("cuda:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
print("laspy:", laspy.__version__)
PY
```

Nếu đọc `.laz` báo `No LazBackend selected`, cài backend:

```bash
conda activate maptr
python -m pip install laszip
```

Hoặc convert trước LAZ sang BIN:

```bash
cd /home/khanh247/Documents/Survey/phenikaa
conda activate maptr
python -m phenikaa_hdmap.converters.laz_to_bin \
  data/phenikaa/sequences/RESIDENTIAL_AREA/Lidar \
  -o data/phenikaa/sequences/RESIDENTIAL_AREA/LidarBin
```

Step 1 sẽ tự ưu tiên `.bin` nếu có `LidarBin/<timestamp>.bin`.

## 5. B3 chính: Phenikaa ảnh + LiDAR + calib

Smoke test 1 frame:

```bash
cd /home/khanh247/Documents/Survey/phenikaa
conda activate maptr
python scripts/run_b3_phenikaa.py \
  --baseline b2_fusion \
  --max-frames 1 \
  --force-step1 \
  --preview-frames 1
```

Chạy 200 frame:

```bash
cd /home/khanh247/Documents/Survey/phenikaa
conda activate maptr
python scripts/run_b3_phenikaa.py \
  --baseline b2_fusion \
  --max-frames 200 \
  --force-step1 \
  --preview-frames 20
```

Mặc định `b2_fusion` dùng `phenikaa6`, vì checkpoint B2 được train trên 6 camera nuScenes. Không dùng `all10` cho checkpoint này nếu chưa fine-tune 10 camera.

Kết quả:

```text
outputs/phenikaa_vis/phenikaa_b3/maptr_results/nuscmap_results.json
outputs/phenikaa_vis/phenikaa_b3/lanes_local/
outputs/phenikaa_vis/phenikaa_b3/lanes_local_manifest.json
outputs/phenikaa_vis/phenikaa_b3/preview/
outputs/phenikaa_vis/phenikaa_b3/global_map/global_vectors.json
outputs/phenikaa_vis/phenikaa_b3/global_map/global_vectors.geojson
outputs/phenikaa_vis/phenikaa_b3/global_map/global_vectors.osm
outputs/phenikaa_vis/phenikaa_b3/global_map/ego_poses.geojson
```

## 6. Chạy so sánh camera-only

Nhánh này không dùng LiDAR feature, chỉ dùng camera + calib metadata.

```bash
cd /home/khanh247/Documents/Survey/phenikaa
conda activate maptr
python scripts/run_b3_phenikaa.py \
  --baseline b1_camera \
  --camera-set all10 \
  --max-frames 20 \
  --force-step1 \
  --preview-frames 20
```

## 7. Về `residential_map.pcd`

`data/phenikaa/maps/residential_map.pcd` không phải GT lanelet. Nó là bản đồ điểm tĩnh/point cloud map, phù hợp cho B6 map prior hoặc hỗ trợ annotate. Nó chưa thể dùng để chấm mAP MapTR nếu chưa có lane vector ground truth.

Muốn train/fine-tune Phenikaa 10 camera cần có GT lane vector:

- `divider`
- `boundary`
- `ped_crossing` nếu có

Sau đó mới tạo `infos_residential_train.pkl`, `infos_residential_val.pkl` và config train 10 camera.
