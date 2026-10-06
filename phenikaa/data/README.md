# Data Directory

Theo tinh than Cookiecutter Data Science, thu muc `data/` khong dua len Git.

Quy uoc mong muon:

```text
data/
  raw/         # du lieu goc, khong sua truc tiep
  processed/   # du lieu da tien xu ly/gan nhan/chuan hoa
  external/    # dataset ben ngoai: nuScenes, KITTI, pretrained dataset...
```

Hien tai repo van con cac scenario cu nam truc tiep trong `data/` nhu `Normal/`, `road_work_traffic/`. Chua di chuyen de tranh lam hong pipeline dang chay. Khi migrate dan, moi scenario nen ve dang:

```text
data/raw/<scenario>/CAMERA/...
data/raw/<scenario>/dump/...
data/processed/<scenario>/<scenario>.osm
```
