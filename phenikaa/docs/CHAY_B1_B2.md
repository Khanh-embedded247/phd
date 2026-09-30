# Hướng dẫn chạy B1 và B2 (chỉ lệnh Python)

Mục tiêu: **bạn tự chạy**, hiểu từng bước. Mỗi bước = **một file `.py`**.  
Sửa đường dẫn **trong file Python** (khối `CONFIG`), không cần nhớ lệnh dài.

Sau mỗi lần chấm điểm (eval), số tự ghi vào **`docs/BAO_CAO_QUA_TRINH.docx`** (file chung B0→B3…).

---

## Chuẩn bị (một lần)

```bash
conda activate maptr
cd /home/khanh247/Documents/Survey/phenikaa
python scripts/check_setup.py
```

Nếu thiếu checkpoint, tải vào `third_party/MapTR/ckpts/` (xem `configs/e1_nusc_camera.md`).

---

## B1 — Chỉ camera (MapTRv2)

### Bước 1: Chấm điểm mAP (~ vài chục phút)

```bash
conda activate maptr
cd /home/khanh247/Documents/Survey/phenikaa
python scripts/run_b1_eval.py
```

**Sửa tham số:** mở `scripts/run_b1_eval.py`, khối `CONFIG`:

| Biến | Ý nghĩa |
|------|---------|
| `PROJECT_ROOT` | Thư mục gốc `phenikaa/` trên máy bạn |
| `MAPTR_CONFIG` | File config MapTR (tương đối từ `third_party/MapTR/`) |
| `CHECKPOINT` | File `.pth` trọng số |
| `OUTPUT_DIR` | Nơi lưu `eval_log.txt` |
| `GPU_COUNT` | Số GPU (4060 8GB → để `1`) |
| `GHI_BAO_CAO` | `True` = tự ghi Word sau khi xong |

**Đầu ra:**

- Log: `outputs/nusc_eval/e1_camera/eval_log.txt`
- Bảng số: `outputs/nusc_eval/e1_camera/metrics.md`
- Word: `docs/BAO_CAO_QUA_TRINH.docx` (mục B1)

### Bước 2: Xem lại 4 số (không chạy model)

```bash
python scripts/xem_ket_qua.py b1
```

### Bước 3: Vẽ hình (tuỳ chọn)

```bash
python scripts/run_b1_vis.py
```

**Sửa tham số:** `scripts/run_b1_vis.py` → `SHOW_DIR`, `SCORE_THRESH`, `MAX_SAMPLES`.

Hình nằm ở `outputs/nusc_eval/e1_camera/vis_pred/<tên_frame>/`.

---

## B2 — Camera + LiDAR (fusion)

### Bước 1: Chấm điểm

```bash
conda activate maptr
cd /home/khanh247/Documents/Survey/phenikaa
python scripts/run_b2_eval.py
```

**Sửa tham số:** `scripts/run_b2_eval.py` (cùng cách như B1).

**Đầu ra:**

- `outputs/nusc_eval/e2_fusion/eval_log.txt`
- `docs/BAO_CAO_QUA_TRINH.docx` (mục B2)

### Bước 2: Xem số

```bash
python scripts/xem_ket_qua.py b2
```

### Bước 3: Vẽ hình (tuỳ chọn)

```bash
python scripts/run_b2_vis.py
```

---

## Ghi báo cáo Word (nếu cần chạy lại tay)

Eval thường đã ghi tự động. Nếu muốn ghi lại từ log cũ:

```bash
conda activate phenikaa
python scripts/ghi_bao_cao.py b1
python scripts/ghi_bao_cao.py b2
```

**Sửa đường dẫn Word:** `scripts/ghi_bao_cao.py` → `BAO_CAO_DOCX`.

---

## Hiểu nhanh B1 vs B2

| | B1 | B2 |
|--|----|----|
| Đầu vào | 6 camera | 6 camera + LiDAR |
| Model | MapTR**v2** camera | MapTR **v1** fusion (ckpt official) |
| Dataset chấm điểm | nuScenes **mini** (81 frame) | Cùng mini |
| Ý nghĩa | Baseline “chỉ nhìn ảnh” | “Thêm LiDAR có giúp không?” |
| **Không phải** | Điểm trên đường Phenikaa VN | |

So sánh hợp lệ về **hướng đề tài**, nhưng hai model hơi khác kiến trúc — ghi đúng trong báo cáo (mục 6.2 Word).

---

## Thứ tự làm lại từ đầu

1. `python scripts/run_b1_eval.py`
2. `python scripts/run_b1_vis.py` (tuỳ chọn)
3. `python scripts/run_b2_eval.py`
4. `python scripts/run_b2_vis.py` (tuỳ chọn)
5. Mở `docs/BAO_CAO_QUA_TRINH.docx` → điền **kết luận** bằng lời (mục 5.4, 6.4)

---

## Lỗi thường gặp

| Lỗi | Cách xử lý |
|-----|------------|
| `conda: command not found` | Mở terminal mới, `conda activate maptr` |
| `No module named torch` | Đang ở sai env → `conda activate maptr` |
| `No such file ... pkl` | Chạy lại eval (script tự sửa symlink data) |
| Không ghi được Word | `conda activate phenikaa` rồi `pip install python-docx` |
