# Phenikaa HD Map — Online Lanelet / Vector HD Map

Dự án luận văn: **Robust Online Lanelet-Level HD Map Construction** từ multi-camera, multi-LiDAR và (tuỳ chọn) static pointcloud map prior trên dữ liệu **Phenikaa**.

Phạm vi ngắn hạn (MVP):

1. Baseline MapTRv2 camera / camera+LiDAR trên **nuScenes** (có GT map).
2. Adapter + inference qualitative trên sequence **RESIDENTIAL_AREA** Phenikaa.
3. (Nếu map.pcd khớp Pose) thêm static-map prior BEV.

## Cấu trúc (tất cả trong `phenikaa/`)

```text
phenikaa/
├── README.md
├── requirements.txt          # env phenikaa (B0 tooling)
├── docs/                     # lộ trình, khái niệm, hình mẫu
├── configs/                  # ghi chú thí nghiệm E1/E2…
├── phenikaa_hdmap/           # code chính đồ án
├── scripts/
├── tutorials/                # của Phenikaa (calib / viz) — không phải upstream
├── third_party/MapTR/        # MapTRv2 opensource (branch maptrv2) nằm trong dự án
│   ├── docs/install.md
│   ├── requirement.txt       # extras nhỏ (shapely/av2)
│   ├── ckpts/                # trọng số ResNet + MapTRv2
│   └── data/ → ../../data/nuscenes/…
├── data/
│   ├── phenikaa/             # calib, sequences, maps
│   └── nuscenes/             # raw mini + can_bus (thật, trong phenikaa)
└── outputs/
```

`Survey/test/` là **quá khứ** — không dùng cho đồ án.

## Hai môi trường

| Env conda | File deps | Dùng cho |
|-----------|-----------|----------|
| `phenikaa` | [`requirements.txt`](requirements.txt) | B0 tutorials, data tooling |
| `maptr` | [`requirements-maptr.txt`](requirements-maptr.txt) (+ [`environment-maptr.yml`](environment-maptr.yml)) | B1+ MapTRv2 |

## Bắt đầu nhanh

```bash
cd /home/khanh247/Documents/Survey/phenikaa

# B0 — env phenikaa
conda activate phenikaa
python scripts/check_setup.py
cd tutorials && python 05_load_pointcloud_information.py && python 11_all_things.py

# B1 — env maptr
conda activate maptr
cd third_party/MapTR
# xem configs/e1_nusc_camera.md
```

## Tài liệu

| File | Nội dung |
|------|----------|
| **[docs/HIEU_RO.md](docs/HIEU_RO.md)** | Hiểu khái niệm |
| **[docs/LO_TRINH.md](docs/LO_TRINH.md)** | Lộ trình từng bước |
| **[docs/mau/](docs/mau/)** | Hình mẫu |
| [TASKS.md](TASKS.md) | Checklist |
| [docs/DATA.md](docs/DATA.md) | Dữ liệu |
| [docs/CHAY_B3_PHENIKAA.md](docs/CHAY_B3_PHENIKAA.md) | Lệnh chạy B3 Phenikaa |
| [configs/e1_nusc_camera.md](configs/e1_nusc_camera.md) | Lệnh eval B1 |

## Quy ước

- Logic đồ án → `phenikaa_hdmap/`
- Upstream MapTRv2 → `third_party/MapTR/` (patch tối thiểu)
- Tutorials calib → `tutorials/` (thuộc Phenikaa)
