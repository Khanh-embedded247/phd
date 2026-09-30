# Đọc hết file này trước khi làm tiếp

Viết bằng tiếng Việt dễ hiểu. Mục tiêu: biết mình **cần gì / không cần gì**, đầu vào → đầu ra.

---

## 1. Hai kiểu việc (đừng trộn)

### Kiểu A — Bộ dữ liệu công khai (nuScenes)

Có **đáp án làn đường sẵn**.  
Dùng để **dạy** mạng và **chấm điểm số**.

### Kiểu B — Dữ liệu Phenikaa của bạn (khu dân cư)

Có camera + LiDAR + chỉnh cảm biến + vị trí xe.  
**Chưa có đáp án làn đường.**  
Dùng để **chạy thử xem hình**, chứng minh trên đường Việt Nam.

Luận văn của bạn = làm **cả hai**: số liệu trên A, hình minh họa trên B.

---

## 2. Với bài toán Phenikaa: có BẮT BUỘC đáp án làn đường không?

### Trả lời ngắn

| Việc bạn muốn | Có cần đáp án làn Phenikaa? |
|---------------|------------------------------|
| Chạy thử model, xem vẽ được vạch/lề không | **Không** |
| Ra số “đúng bao nhiêu %” trên Phenikaa | **Có** |
| Đúng Survey: chấm điểm chính | Chấm trên **nuScenes** (đã có đáp án) |
| Đúng Survey: Phenikaa | Chủ yếu **xem hình**; đáp án Phenikaa chỉ cần nếu muốn thêm số liệu nhỏ |

**Kết luận đồ án:**  
**Không bắt buộc** phải có đáp án làn Phenikaa mới làm được đề tài.  
Bắt buộc có đáp án là khi train/chấm trên nuScenes (bộ đó **đã có sẵn**).

---

## 3. “Đáp án làn đường” là gì? Tạo thế nào?

### Nó là cái gì?

Không phải nhãn xe/người trong file `Label/`.

Nó là: **các đường cong mô tả vạch kẻ / mép đường / lối đi bộ quanh xe**, tính bằng mét.

Ví dụ thật lấy từ file pkl nuScenes của bạn (1 khung hình):

- Vạch phân làn (`divider`): đoạn đầu có các điểm  
  `(10.1, -30)`, `(10.3, -13)`, `(10.9, 3.9)` … (mét quanh xe)
- Mép đường (`boundary`): nhiều điểm tương tự
- Tim đường (`centerline`): tương tự

Nhìn hình mẫu trong thư mục `docs/mau/`:

| File | Ý nghĩa |
|------|---------|
| `01_nuscenes_anh_camera.jpg` | Ảnh camera nuScenes |
| `02_nuscenes_ban_do_nen.png` | Bản đồ nền khu vực (nhìn từ trên) |
| `04_dap_an_lan_duong_nhin_tu_tren.png` | **Đáp án làn** vẽ trên mặt nhìn từ trên |
| `03_phenikaa_anh_camera.jpg` | Ảnh camera Phenikaa của bạn |
| `05_anh_...` | Ảnh trong bộ kết quả model (nếu có) |

### File `boston-seaport.json` dùng để làm gì?

Đó là **bản đồ thành phố đã vẽ sẵn** (đường, vạch…).  
Phần mềm cắt quanh vị trí xe → thành đáp án từng khung hình.  
**Không phải** nhãn phát hiện xe.

### Phenikaa muốn có đáp án thì tạo thế nào?

Với dữ liệu bạn đang có (chỉ ảnh + LiDAR + nhãn xe):

1. Chọn vài chục khung hình.  
2. Người (hoặc công cụ vẽ) **vẽ tay** vạch/mép đường trên ảnh hoặc trên mặt nhìn từ trên.  
3. Lưu thành danh sách điểm (hoặc file bản đồ làn).  

**Không** tự sinh ra từ `Label/` xe được.  
**Không** bắt buộc làm bước này ngay để chạy thử.

Với **một khu bất kỳ** cũng vậy: muốn chấm điểm thì phải có bản đồ làn hoặc vẽ tay; muốn chỉ demo thì chỉ cần cảm biến + file chỉnh cảm biến + mạng đã dạy trên bộ có đáp án.

---

## 4. Từng thứ trong opensource — cần hay không?

### 4.1 nuScenes mini + map expansion

- **Ảnh + LiDAR:** đầu vào để dạy mạng.  
- **map expansion (json):** bản đồ để cắt ra đáp án làn.  
- **Cần cho đồ án:** Có (để dạy và chấm điểm).  
- Xem mẫu: `docs/mau/01_...`, `02_...`, `04_...`

### 4.2 can_bus / ego (nuScenes)

- Là thông tin **xe đang ở đâu / hướng nào** trong bộ nuScenes.  
- MapTR dùng khi **đóng gói** dữ liệu nuScenes và khi dạy.  
- **Phenikaa không cần file can_bus.**  
- Phenikaa đã có thư mục `Pose/` (vai trò tương tự: biết xe ở đâu khi ghép nhiều khung hình).  
- **Có thật sự cần?** Chỉ khi làm nhánh nuScenes. Không copy sang Phenikaa.

### 4.3 File pkl infos nuScenes

- Là **mục lục đã đóng gói sẵn**: mỗi dòng = 1 khung hình, trỏ tới ảnh, LiDAR, đáp án làn, vị trí xe…  
- **Không dùng chung mọi khu.**  
  - nuScenes → một (vài) file pkl nuScenes.  
  - Phenikaa → file mục lục riêng (`infos_residential.pkl` đã tạo).  
- Đổi khu / đổi bộ dữ liệu → tạo mục lục mới.

### 4.4 Backbone ResNet (file `.pth` nhỏ)

- Là phần mạng **nhìn ảnh** đã học sẵn trên ảnh thường (ImageNet), chưa phải “biết làn đường”.  
- MapTR lấy làm điểm khởi đầu rồi dạy tiếp trên nuScenes.  
- **Dùng được cho mọi khu** khi chạy cùng kiến trúc MapTR (không gắn với Boston hay Phenikaa).

### 4.5 Checkpoint MapTR sau khi dạy trên nuScenes

- Đây mới là bộ não **biết đoán làn**.  
- Dùng được để **chạy thử Phenikaa** (chất lượng có thể kém hơn vì khác camera/đường).  
- Không chứa ảnh Boston bên trong.

---

## 5. Lấy gì từ opensource? Cần thêm gì?

### Lấy từ opensource MapTR + nuScenes

1. Code kiến trúc (cách xử lý ảnh/LiDAR → đoán đường).  
2. Cách dạy và chấm trên nuScenes.  
3. Trọng số khởi đầu ResNet.  
4. (Nên có) trọng số MapTR đã dạy xong trên nuScenes — tự dạy hoặc tải.  

### Lấy từ Phenikaa (đã có)

1. Ảnh nhiều camera.  
2. LiDAR.  
3. File chỉnh cảm biến (`calib/`).  
4. Vị trí xe (`Pose/`) — dùng khi ghép nhiều khung hình.  

### Cần thêm / làm tiếp (chưa xong)

1. Chạy xong bước dạy/chấm **camera** trên nuScenes (baseline 1).  
2. Thêm nhánh **camera + LiDAR**, dạy lại, so sánh (baseline 2).  
3. Viết cầu nối: đọc data Phenikaa → chạy model → ra hình.  
4. (Sau) ghép nhiều khung hình bằng `Pose/`.  
5. (Sau, nếu có) bản đồ điểm tĩnh đúng khu → nhánh prior.  
6. (Tuỳ chọn) vẽ tay vài đoạn làn Phenikaa nếu muốn thêm số liệu VN.

### Không cần / có thể bỏ khỏi đầu

- Coi `Label/` xe là đáp án làn.  
- File bản đồ điểm `map_candidate` / `pnkx` (sai khu / sai hệ tọa độ).  
- Radar nuScenes.  
- Tự tìm lại ma trận chỉnh cảm biến nếu `calib/` đã khớp (chỉ cần kiểm tra bằng tutorials).  
- Bắt buộc có đáp án làn Phenikaa trước khi demo.

---

## 6. Đường ống gọn (đầu vào → đầu ra)

### Nhánh 1 — Dạy và chấm điểm (nuScenes)

```text
ĐẦU VÀO:
  ảnh + LiDAR nuScenes
  bản đồ map expansion
  can_bus / vị trí xe (trong bộ nuScenes)
  mục lục .pkl

XỬ LÝ:
  MapTR học: đoán đường ↔ so với đáp án làn

ĐẦU RA:
  file trọng số MapTR (.pth)
  bảng điểm (mAP…)
  hình nhìn từ trên có đường vẽ
```

### Nhánh 2 — Chạy thử Phenikaa (đúng ý bạn)

```text
ĐẦU VÀO:
  ảnh Phenikaa
  LiDAR Phenikaa
  calib Phenikaa          ← dùng file có sẵn, không lấy của nuScenes
  trọng số MapTR (nhánh 1)
  (tuỳ chọn) Pose/

XỬ LÝ:
  Cùng kiến trúc MapTR, một lần chạy ra đường đoán

ĐẦU RA:
  đường vẽ trên hình / nhìn từ trên
  (sau) ghép nhiều khung → file bản đồ làn thô
```

Không có bước “tự xuất file đặc trưng BEV bằng tay”. Model làm bên trong.

---

## 7. Thứ tự làm việc (theo Survey, đã rút gọn)

1. Kiểm tra chỉnh cảm biến Phenikaa (tutorials) — dùng `calib/` có sẵn.  
2. Baseline 1: camera trên nuScenes → trọng số + điểm.  
3. Baseline 2: camera + LiDAR trên nuScenes → so với bước 2.  
4. Chạy thử lên RESIDENTIAL → hình.  
5. Ghép nhiều khung bằng Pose (chưa cần bản đồ điểm tĩnh).  
6. Khi có bản đồ điểm đúng khu → thêm nhánh prior.  
7. (Tuỳ chọn) vẽ tay đáp án làn Phenikaa → thêm vài số liệu VN.

---

## 8. Một câu nhớ

> Opensource dạy mạng trên chỗ **đã có đáp án làn**.  
> Phenikaa của bạn **đưa cảm biến + chỉnh cảm biến thật** vào mạng đó để xem đường Việt Nam.  
> Đáp án làn Phenikaa **không bắt buộc** để chạy thử; chỉ cần nếu muốn chấm điểm trên đúng data mình.
