# Phenikaa HD Map — Roadmap

Checklist làm việc. Đánh dấu `[x]` khi xong.

## Tuần 1 — Setup & calib

- [ ] Chạy `python scripts/check_setup.py` — mọi symlink OK
- [ ] Sửa `tutorials/phenikaa_paths.py` trỏ về `data/phenikaa/...` (hoặc dùng `phenikaa_hdmap.paths`)
- [ ] Chạy tutorials `04`, `05`, `11` — LiDAR + box đúng trên ảnh
- [ ] Chạy `python scripts/verify_map_pose.py` — chọn `residential.pcd` chính thức
- [ ] Copy/symlink map đã verify → `data/phenikaa/maps/residential.pcd`

## Tuần 2 — Baseline MapTRv2 (nuScenes)

- [ ] Eval camera MapTRv2 mini (đã có trong `third_party/MapTR`)
- [ ] Lưu số mAP + vài hình vào `outputs/nusc_eval/e1_camera/`
- [ ] Ghi baseline vào log thí nghiệm

## Tuần 3–4 — Camera + LiDAR fusion

- [x] Đọc config `third_party/MapTR/projects/configs/maptr/maptr_tiny_fusion_24e.py`
- [x] Eval MapTR-tiny fusion official ckpt (MapTR v1) trên mini → `outputs/nusc_eval/e2_fusion/`
- [ ] (Tuỳ chọn) Port/train MapTRv2 fusion riêng nếu cần so sánh fair hơn
- [x] So sánh E2 vs E1 → `outputs/nusc_eval/e2_fusion/metrics.md`

## Tuần 4–5 — Adapter Phenikaa

- [ ] `laz_to_bin.py` / đọc laz trong pipeline
- [ ] `build_phenikaa_infos.py` → `data/phenikaa/infos_residential.pkl`
- [ ] `infer_phenikaa.py` với checkpoint nuScenes
- [ ] Visualization → `outputs/phenikaa_vis/`

## Tuần 6 — Static prior (nếu có map)

- [ ] `rasterize_map_prior.py` (density, height, intensity)
- [ ] Fuse prior + gating trong model wrapper
- [ ] Ablation có/không prior trên frame bị che

## Tuần 7 — Aggregation prototype

- [ ] Transform vector theo Pose → map frame
- [ ] Filter + merge + gap-fill đơn giản
- [ ] Export geojson / viz → `outputs/aggregated_maps/`

## Tuần 8 — Viết luận & freeze

- [ ] Bảng số nuScenes + hình Phenikaa
- [ ] Freeze phạm vi; không mở thêm module lớn

## Nguyên tắc

1. Train định lượng trên **nuScenes**.
2. Phenikaa = **qualitative / hệ thống**.
3. Code mới chỉ trong `phenikaa_hdmap/`; upstream MapTR sửa tối thiểu.
