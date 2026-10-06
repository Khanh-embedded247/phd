# Phenikaa HD Map — System Design

## 1. Bài toán

**Input (1 frame / cửa sổ ngắn):**

- Multi-camera Phenikaa (10 cam: 6 pinhole + 4 fisheye)
- Multi-LiDAR đã gộp → 1 cloud `.laz` / frame
- (Tuỳ chọn) Static pointcloud map `.pcd` crop theo ego `Pose/`

**Output:**

- Vector map local: lane divider, road boundary, centerline (+ confidence)
- (Prototype) Aggregate nhiều frame → vector / lanelet thô trên map frame

## 2. Pipeline 2 tầng

```text
[Cameras] ──► Camera BEV ──┐
[LiDAR]   ──► LiDAR BEV  ──┼──► Fusion (+ gating) ──► MapTRv2 decoder
[map.pcd] ──► Prior BEV* ──┘                              │
                                                     vector polylines
                                                            │ Pose
                                                            ▼
                                              Multi-frame aggregation
                                              (filter / merge / gap-fill)
```

\* Prior chỉ bật khi `verify_map_pose.py` xác nhận PCD khớp `Pose/`.

## 3. Vai trò từng kho dữ liệu

| Nguồn | Vai trò |
|-------|---------|
| nuScenes | Train + metric định lượng (có HD map GT) |
| Phenikaa RESIDENTIAL_AREA | Demo / qualitative (chưa có lane GT) |
| Label Phenikaa (Car/Rider/…) | Chỉ phục vụ check calib, **không** train map |
| map.pcd | Static prior geometry (density / height / intensity) |

## 4. Module phần mềm

| Module | Thư mục | Chức năng |
|--------|---------|-----------|
| Paths | `phenikaa_hdmap/paths.py` | Mọi đường dẫn Phenikaa |
| Calib utils | `phenikaa_hdmap/utils/calib.py` | Load intrinsics/extrinsics |
| Pose utils | `phenikaa_hdmap/utils/pose.py` | Đọc Pose 3×4 map←ego |
| Converters | `phenikaa_hdmap/converters/` | laz→bin, infos.pkl, prior raster |
| Dataset | `phenikaa_hdmap/datasets/` | Dataset inference Phenikaa |
| Models | `phenikaa_hdmap/models/` | Wrapper fusion / prior (thin) |
| Tools | `phenikaa_hdmap/tools/` | infer, aggregate, viz |
| Upstream | `third_party/MapTR` | MapTRv2 backbone |

## 5. Thí nghiệm (rút gọn)

| ID | Nội dung | Metric |
|----|----------|--------|
| E1 | MapTRv2 camera — nuScenes | mAP |
| E2 | + LiDAR BEV fusion — nuScenes | mAP vs E1 |
| E3 | + static prior (nếu có map) | qualitative / drop khi corruption |
| E4 | Infer Phenikaa Cam+LiDAR | video / BEV overlay |
| E5 | Multi-frame aggregate | demo gap-fill |

## 6. Phạm vi không làm (deadline ngắn)

- Annotate full lanelet Phenikaa
- Train map model chỉ trên 200 frame (không đủ GT)
- Production Lanelet2 hoàn chỉnh
