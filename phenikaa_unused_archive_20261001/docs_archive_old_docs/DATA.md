# Phenikaa HD Map — Data

Mọi data dùng cho đồ án nằm **trong** `phenikaa/data/` (không phụ thuộc `Survey/test/`).

## Phenikaa sequence

| Mục | Đường dẫn |
|-----|-----------|
| Sequence | `data/phenikaa/sequences/RESIDENTIAL_AREA/` |
| Images | `.../Image/CAM_*/*.jpg` (10 cameras) |
| LiDAR | `.../Lidar/*.laz` (200 frames) |
| Pose | `.../Pose/*.txt` (3×4, map ← sensor/ego) |
| Labels | `.../Label/*.txt` (Car / Rider / Pedestrian — **không phải lane GT**) |
| Calib | `data/phenikaa/calib/Camera_Intrinsics.json` |
| Calib | `data/phenikaa/calib/Sensor_Extrinsics.json` |

Cameras:

- Pinhole: `CAM_P_F`, `CAM_P_FL`, `CAM_P_FR`, `CAM_P_L`, `CAM_P_R`, `CAM_P_B`
- Fisheye: `CAM_F_F`, `CAM_F_L`, `CAM_F_R`, `CAM_F_B`

## Maps (ứng viên — cần verify)

| File | Trạng thái |
|------|------------|
| `data/phenikaa/maps/*_candidate*` | **không dùng** (sai khu) |
| `residential.pcd` | Chưa có — chỉ cần cho B6 |

## nuScenes (train / metric)

| Mục | Path |
|-----|------|
| Raw (mini) | `data/nuscenes/raw/` |
| can_bus | `data/nuscenes/can_bus/` |
| MapTR nhìn thấy | `third_party/MapTR/data/nuscenes` → `../../../data/nuscenes/raw` |

## Upstream MapTRv2

`third_party/MapTR/` — clone branch **maptrv2**, nằm trong dự án Phenikaa.

Tutorials calib: `tutorials/` — **của Phenikaa**, không phải upstream.
