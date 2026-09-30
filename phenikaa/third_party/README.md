# MapTR / MapTRv2 — ghi chú env trong dự án Phenikaa

## Dùng nhánh nào?

- Repo opensource: [hustvl/MapTR](https://github.com/hustvl/MapTR)
- Đồ án dùng branch **`maptrv2`**
- Code: `phenikaa/third_party/MapTR/` (một chỗ)

## Conda env `maptr`

| File | Vai trò |
|------|---------|
| [`../requirements-maptr.txt`](../requirements-maptr.txt) | **Chính** — pip deps + hướng dẫn cài torch/mmcv/mmdet3d |
| [`../environment-maptr.yml`](../environment-maptr.yml) | Tạo env `python=3.8` rồi làm tiếp theo file trên |
| `MapTR/docs/install.md` | Upstream gốc |
| `MapTR/requirement.txt` | Extras nhỏ upstream (shapely/av2) |

```bash
conda activate maptr
cd /home/khanh247/Documents/Survey/phenikaa/third_party/MapTR
```

Env `phenikaa` → [`../requirements.txt`](../requirements.txt) — chỉ B0 / tutorials.
