# Cau Truc Repository Phenikaa

Du an duoc sap xep theo huong Cookiecutter Data Science ket hop cach lam lab: data/output de rieng, config/source/script de rieng, third-party code tach rieng.

```text
phenikaa/
├── data/                 # data/raw, data/processed, data/external; khong push git
├── outputs/              # ket qua sinh ra tu pipeline, khong push git
├── configs/maptr/        # cau hinh MapTR Phenikaa
├── scripts/maptr/        # script pipeline chay bang python + ten file
├── src/phenikaa_maptr/   # helper noi bo cua pipeline
├── third_party/          # MapTR runtime va checkpoint goc
├── docs/                 # tai lieu nghien cuu/pipeline/rule GT
└── thesis_papers/        # paper, latex thesis, figure bao cao
```


## Data Layout

- `data/raw/`: du lieu goc, gom cac scenario Phenikaa va dataset external raw nhu NuScenes. Khong sua truc tiep noi dung raw neu khong co ly do ro rang.
- `data/processed/`: du lieu sinh ra sau tien xu ly neu can luu trong data thay vi `outputs/`.
- `data/external/`: du lieu ben thu ba hoac tai nguyen ngoai.
Canonical path tu nay la `data/raw/<scenario>`. Khong giu symlink `data/<scenario>`; neu pkl cu chua path cu thi rebuild lai infos tu raw.

## Vai Tro Tung Thu Muc

- `configs/maptr/`: noi duy nhat can sua config thuc nghiem MapTR.
- `scripts/maptr/`: cac buoc pipeline co the chay rieng, hoac qua `30_run_pipeline_from_config.py`.
- `src/phenikaa_maptr/`: code helper dung chung, khong chay truc tiep nhu script.
- `third_party/MapTR_phenikaa/`: ban MapTR da sua de doc OSM/6 class Phenikaa.
- `third_party/MapTR/ckpts/`: luu checkpoint pretrained goc.
- `outputs/only_camera/`: ket qua sinh ra tu pipeline; ten thu muc giu lai de khong lam hong output cu.

## Thu Muc Cu

