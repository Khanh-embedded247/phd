# Experiments Va Checkpoint

## File Mang Tri Thuc Model

Tri thuc model da hoc nam trong checkpoint:

```text
checkpoints/road_work_traffic_maptr_epoch_4.pth
```

Khi mang sang may khac de infer, can toi thieu:

- `configs/maptr/maptr_tiny_r50_phenikaa_12cam_train_osm.py`
- checkpoint `.pth` da train
- `third_party/MapTR_phenikaa/`
- `src/phenikaa_maptr/` va `scripts/maptr/`
- data/inference input tuong ung, gom image/pose/infos pkl neu khong rebuild

## File Co The Xoa Khi Thieu Dung Luong

- `.log`, `.log.json`: chi can giu log cua experiment tot nhat.
- visualization trung gian: co the tao lai tu `predictions.json` va infos.
- checkpoint epoch cu: giu `latest.pth` hoac epoch tot nhat, xoa cac epoch khong dung.
- `_intermediate/`: co the tao lai khi chay export/visualize.

## Config Source Of Truth

Sua config tai:

```text
configs/maptr/pipeline.yaml
configs/maptr/maptr_tiny_r50_phenikaa_12cam_train_osm.py
```

Config copy trong `outputs/.../work_dirs/` chi la snapshot luc train, khong nen sua lam source chinh.
