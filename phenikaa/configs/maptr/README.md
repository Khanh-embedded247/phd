# MapTR Configs

Thu muc nay la source of truth cho cau hinh MapTR Phenikaa.

- `pipeline.yaml`: cau hinh tong de chay pipeline tu raw data den infer/export/visualize.
- `maptr_tiny_r50_phenikaa_12cam_train_osm.py`: config train hien tai cho 6 class OSM.
- `maptr_tiny_r50_phenikaa_12cam.py`: config infer/legacy tu giai doan dau, giu de doi chieu.

Quy tac:
- Runtime code MapTR dung `${PHENIKAA_ROOT}/third_party/MapTR_phenikaa`.
- Checkpoint goc/pretrained dung `${PHENIKAA_ROOT}/third_party/MapTR/ckpts` vi file `.pth` dang nam o do.
- Output train/infer van nam trong `outputs/only_camera/...`; day la du lieu ket qua, khong phai source code.
