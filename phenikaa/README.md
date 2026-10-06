# Phenikaa Vector HD Map Project

Du an nay train va chay MapTR cho bai toan tao vector HD map tu camera Phenikaa, voi GT OSM duoc ve theo quy tac rieng cho 6 class:
`lane_divider`, `road_edge_marking`, `stop_line`, `ped_crossing`, `boundary`, `speed_bump`.

## Cau Truc Chinh

```text
phenikaa/
├── data/                 # data/raw, data/processed, data/external; khong push git
├── outputs/              # ket qua build infos, train, infer, export, visualize
├── configs/maptr/        # tat ca config MapTR dang dung
├── scripts/maptr/        # pipeline script chay truc tiep bang Python
├── src/phenikaa_maptr/   # helper noi bo cho pipeline
├── third_party/          # MapTR runtime va checkpoint goc
├── docs/                 # tai lieu nghien cuu, GT rule, pipeline, cleanup
└── thesis_papers/        # paper, latex, figures cho luan van
```

`Pipeline moi chay tu `scripts/maptr` va `configs/maptr`.

## Data Layout

Du lieu goc cua cac scenario nam o:

```text
data/raw/<scenario>/
```

Vi du:

```text
data/raw/Normal/
data/raw/road_work_traffic/
data/raw/rain_low_light/
data/raw/nuscenes/
```

Cac duong dan cu nhu `data/Normal` va `data/road_work_traffic` da duoc bo. Config va pipeline moi chi doc canonical path trong `data/raw`. Neu dung output `.pkl` cu duoc build truoc khi doi layout, nen rebuild lai infos tu raw.

## Chay Pipeline

Chay tung buoc bang runner tong:

```bash
cd /home/khanh247/Documents/Survey/phd/phenikaa
/home/khanh247/miniconda3/envs/maptr/bin/python scripts/maptr/30_run_pipeline_from_config.py --steps infer,export,visualize
```

Chay tu dau den cuoi khi can rebuild/train:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python scripts/maptr/30_run_pipeline_from_config.py --steps raw_qa,pose_qa,gt_qa,build_infos,filter_infos,make_config,make_init,check_dataset,train,infer,export,visualize
```

Dry-run de xem lenh se chay:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python scripts/maptr/30_run_pipeline_from_config.py --steps infer,export,visualize --dry-run
```

## File Quan Trong Can Giu

- Config: `configs/maptr/pipeline.yaml`, `configs/maptr/maptr_tiny_r50_phenikaa_12cam_train_osm.py`.
- Source runtime: `third_party/MapTR_phenikaa/`, `scripts/maptr/`, `src/phenikaa_maptr/`.
- Checkpoint da hoc: `checkpoints/road_work_traffic_maptr_epoch_4.pth`.
- Checkpoint pretrained goc: `third_party/MapTR/ckpts/maptrv2_nusc_r50_24ep.pth`, `resnet50-19c8e357.pth`.
- GT rule: `docs/03_GT_OSM_DRAWING_GUIDE_FULL.md`.

## Sau Train

Model hoc duoc nam trong checkpoint `.pth`, khong nam trong file `.log` hay `.pkl`.
Dung checkpoint do de infer tren scenario khac bang cach sua `configs/maptr/pipeline.yaml`:

- `scenario.name`: scenario can infer/export.
- `scenario.checkpoint_scenario`: scenario chua checkpoint da train.
- `paths.checkpoint`: mac dinh tro den `latest.pth` cua checkpoint_scenario.

## Tai Lieu

- `docs/01_REPOSITORY_STRUCTURE.md`: cau truc repo.
- `docs/02_PIPELINE_FROM_RAW_TO_VECTOR_MAP.md`: pipeline tu raw den vector map.
- `docs/03_GT_OSM_DRAWING_RULES.md`: rule ve GT ban ngan.
- `docs/03_GT_OSM_DRAWING_GUIDE_FULL.md`: rule ve GT day du.
- `docs/04_OUTPUTS_AND_CLEANUP.md`: nen giu/xoa output nao.
- `docs/05_EXPERIMENTS_AND_CHECKPOINTS.md`: quan ly thuc nghiem/checkpoint.
- `docs/06_THESIS_QA_CAMERA_ONLY_MAPTR.md`: cau hoi bao ve/hoi dap ky thuat.
