# Project Structure

Repo `phenikaa/` duoc chuan hoa theo Cookiecutter Data Science ket hop thuc te lab autonomous driving.

```text
phenikaa/
  README.md
  .gitignore
  requirements.txt
  environment-maptr.yml

  data/
    raw/
    processed/
    external/

  checkpoints/
  configs/
  src/
    dataset/
    models/
    pipelines/
    utils/

  scripts/
  notebooks/
  experiments/
  docs/
  thesis_papers/
    literature_review/
    latex_thesis/
    figures/

  benchmark_maptr/      # pipeline MapTR dang chay thuc te
  phenikaa_hdmap/       # package nghien cuu noi bo co san
  third_party/          # code ben thu ba, vi du MapTR
  outputs/              # output/cache/log, khong push Git
```

## Nguyen tac

- `data/`, `outputs/`, `checkpoints/`, `experiments/` khong push Git.
- `benchmark_maptr/` hien la pipeline san xuat ket qua MapTR; chua di chuyen de tranh hong duong dan.
- `configs/` la noi nhin nhanh tat ca cau hinh thuc nghiem quan trong.
- `src/` la muc tieu refactor lau dai; code on dinh se duoc tach dan tu `benchmark_maptr/` va `phenikaa_hdmap/`.
- `docs/` giai thich cach chay va quy tac du an; `thesis_papers/` phuc vu luan van/paper.
