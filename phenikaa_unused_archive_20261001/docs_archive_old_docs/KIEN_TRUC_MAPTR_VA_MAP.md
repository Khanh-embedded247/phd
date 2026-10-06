# Kiến trúc MapTRv2, trọng số, mAP — và khoảng cách với Phenikaa

Tài liệu phục vụ viết báo cáo / luận văn. Đồng bộ với code trong `third_party/MapTR` (branch maptrv2) và thí nghiệm B1 đã chạy trên **nuScenes mini**.

---

## 1. “Mạng nơ-ron” ở đây là gì? Thể hiện qua file nào?

### 1.1. Không phải một file duy nhất

| Thành phần | Vai trò | File / chỗ trong repo |
|------------|---------|------------------------|
| **Định nghĩa kiến trúc** (lớp nào, kích thước) | “Bản thiết kế” mạng | Config: `projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py` |
| **Code xây dựng mạng** | Implement các khối | `projects/mmdet3d_plugin/maptr/detectors/maptrv2.py` (lớp `MapTRv2`) |
| | | `.../dense_heads/maptrv2_head.py` (đầu ra đường vector) |
| | | `.../modules/transformer.py`, `encoder.py` (BEV / attention) |
| **Trọng số đã học** | Các số \(W\) trong mạng | `ckpts/maptrv2_nusc_r50_24ep.pth` (checkpoint B1) |
| **Backbone khởi tạo** | ResNet nhìn ảnh (ImageNet) | `ckpts/resnet50-19c8e357.pth` |
| **Đánh giá mAP** | Công thức chamfer / AP | `projects/mmdet3d_plugin/datasets/map_utils/tpfp*.py`, `mean_ap.py` |

Khi load `.pth`, code đọc `state_dict` (tên từng lớp → tensor) và gán vào đúng module trong `MapTRv2`.  
**Kiến trúc** nằm ở config + `.py`; **“bộ não đã học”** nằm ở `.pth`.

### 1.2. Luồng xử lý (camera-only — đúng B1)

```text
Ảnh nhiều camera (nuScenes: 6 cam quanh xe)
        │
        ▼
  ResNet-50  →  FPN          # trích đặc trưng ảnh 2D
        │
        ▼
  LSS / BEV encoder          # “rải” đặc trưng lên lưới nhìn từ trên (BEV)
        │                      kích thước cấu hình: bev_h=200, bev_w=100
        ▼                      vùng quanh xe: point_cloud_range ≈ ±15m × ±30m
  Transformer decoder
  (MapTRv2Head + queries)
        │
        ▼
  Nhiều đường vector (polyline)
  mỗi đường ≈ 20 điểm (x,y) trong hệ tọa độ xe
  + nhãn lớp: divider | ped_crossing | boundary
  + điểm tin cậy (score)
```

Đây là **online vectorized HD map**: đầu ra là **đường cong hình học**, không phải ảnh segmentation thuần.

---

## 2. Bộ não học cái gì? (không phải “học hướng đi” theo nghĩa planning)

### 2.1. Bài toán học có giám sát

Trên nuScenes, mỗi khung hình có **đáp án làn** (cắt từ map expansion): các polyline mét quanh xe.

Mạng học ánh xạ:

\[
f_\theta(\text{ảnh}, \text{calib}) \;\rightarrow\;
\{\text{polyline}_i,\; \text{class}_i,\; \text{score}_i\}
\]

với \(\theta\) = toàn bộ trọng số trong `.pth`.

### 2.2. Học “thế nào là làn đường” nghĩa là gì?

Học **mẫu hình thống kê** từ hàng nghìn khung có đáp án:

- Vạch kẻ (`divider`) thường là đoạn gần thẳng / cong nhẹ theo hướng đường.
- Mép đường (`boundary`) thường là biên hình học hai bên.
- Lối đi bộ (`ped_crossing`) là đoạn ngang đặc trưng.

Không học một định nghĩa hình học cứng “làn = 3.5 m”.  
Học: **nhìn ảnh đa camera + biết góc camera → dự đoán tọa độ điểm trên mặt đất quanh xe**.

**Không** phải học “xe nên rẽ trái/phải” (planning).  
Có thể học **hướng hình học của đoạn đường** (điểm đầu → điểm cuối polyline) như một phần của hình dạng vector — đó là hình học map, không phải lệnh điều khiển.

### 2.3. Hàm mất mát (ý tưởng — khi train)

Khi train (upstream đã làm; B1 của bạn chủ yếu eval), loss gồm các thành phần kiểu:

- Phân loại lớp đường (classification).
- Hồi quy tọa độ điểm trên polyline (regression / point set matching).
- Matching dự đoán ↔ đáp án (Hungarian / permutation-equivalent như paper MapTR).
- (MapTRv2) phụ trợ segmentation BEV / PV nếu bật `aux_seg`.

Chi tiết công thức đầy đủ: paper MapTR / MapTRv2 (arXiv). Báo cáo nên trích paper + trích config loss trong `maptrv2_nusc_r50_24ep.py`.

---

## 3. Cách tính các số “điểm” (AP / mAP) — không nhầm với trọng số mạng

### 3.1. Phân biệt

| Thuật ngữ | Là gì |
|-----------|--------|
| **Trọng số mạng** \(\theta\) | Tensor trong `.pth` |
| **Score** của một đường | Độ tin cậy model gán cho đường đó (0–1) |
| **AP / mAP** | Chỉ số **chấm điểm** sau khi so pred với GT |

### 3.2. Chamfer giữa hai polyline

Cho đường dự đoán \(P=\{p_j\}\) và đáp án \(G=\{g_k\}\) (điểm trên mặt đất, mét):

\[
d_{\mathrm{CD}}(P,G)
=
\frac{1}{|P|}\sum_{p\in P}\min_{g\in G}\|p-g\|
+
\frac{1}{|G|}\sum_{g\in G}\min_{p\in P}\|g-p\|
\]

(Implementation MapTR dùng biến thể qua buffer Shapely / điểm lấy mẫu — xem `tpfp_chamfer.py`.)

### 3.3. True Positive / False Positive

Với mỗi lớp (divider, …), sắp xếp pred theo score giảm dần.  
Pred khớp GT nếu khoảng cách Chamfer **nhỏ hơn ngưỡng** \(\tau\) (trong log B1: 0.5 m, 1.0 m, 1.5 m).  
Mỗi GT chỉ khớp một pred (greedy / matching chuẩn detection).

- **TP**: khớp được GT chưa dùng.  
- **FP**: không khớp.  
- **FN**: GT không ai khớp.

### 3.4. AP và mAP

Với một lớp \(c\) và một \(\tau\):

1. Quét pred theo score → tích lũy precision–recall.  
2. \(\mathrm{AP}_{c,\tau}\) = diện tích dưới đường PR (theo cách COCO-style / code `mean_ap.py`).

Thường báo:

\[
\mathrm{AP}_c = \mathrm{mean}_{\tau\in\{0.5,1.0,1.5\}} \mathrm{AP}_{c,\tau}
\]

\[
\mathrm{mAP} = \mathrm{mean}_{c\in\{\text{divider},\,\text{ped},\,\text{boundary}\}} \mathrm{AP}_c
\]

**Số B1 của bạn (nuScenes mini val):** mAP ≈ **0.803**  
(divider 0.883, ped_crossing 0.927, boundary 0.599).  
Đây là **điểm trên mini opensource**, không phải Phenikaa.

---

## 4. Opensource đã giải quyết “camera bị che / vạch mờ” chưa? Áp sang Phenikaa có đủ không?

### 4.1. Trả lời thẳng

**Chưa.** MapTRv2 camera-only **không** được thiết kế như “giải pháp đầy đủ” cho occlusion / vạch yếu trên đường Việt Nam.

| Vấn đề đề tài | Camera-only MapTR (B1) | Hướng trong lộ trình đồ án |
|---------------|-------------------------|----------------------------|
| Vạch bị che / mờ | Thường **tụt** chất lượng | B2: +LiDAR; B4: đo độ tụt; B6: prior bản đồ điểm |
| Chỉ nhìn một khung | Thấy đoạn ngắn | B5: ghép nhiều khung bằng `Pose/` |
| Đường VN khác nuScenes | Domain gap | B3: chạy thử Phenikaa; B7: annotate nhỏ nếu cần số |

B1 chỉ chứng minh: **pipeline + baseline camera có số**.  
Phần “robust khi camera yếu” là **đóng góp đề tài** (fusion / multi-frame / prior), không có sẵn chỉ vì tải `.pth` opensource.

### 4.2. Camera nuScenes vs Phenikaa — có ảnh hưởng?

| | nuScenes (B1) | Phenikaa |
|---|---------------|----------|
| Số camera | 6 pinhole quanh xe | 10 cam: 6 pinhole + **4 fisheye** |
| Mô hình chiếu | Pinhole + extrinsics chuẩn nuScenes | `Camera_Intrinsics.json` / `Sensor_Extrinsics.json` riêng |
| Độ méo ảnh | Ít (pinhole) | Fisheye **cong mạnh** |
| Phân bố ảnh | Đường Mỹ / Singapore, điều kiện paper | Đường VN, vạch/giao thông khác |

**Hệ quả:**

1. **Phải** đưa đúng calib Phenikaa vào cầu nối B3 — không dùng calib nuScenes.  
2. Fisheye nếu đưa thẳng vào mạng “quen” pinhole → lệch hình học BEV, làn dễ sai. Cần: undistort / model chiếu đúng / hoặc chỉ dùng cam pinhole trước khi mở rộng fisheye.  
3. Kỳ vọng chất lượng B3 **thấp hơn** số mAP trên nuScenes — đó là bình thường (domain shift), không phải lỗi logic đề tài.  
4. B0 (chiếu LiDAR lên ảnh khớp) chứng minh calib Phenikaa dùng được — điều kiện cần trước B3.

---

## 5. Công thức / sơ đồ nên đưa vào báo cáo (checklist)

1. **Sơ đồ khối** MapTRv2 (mục 1.2) + trích config `bev_h/w`, `map_classes`, `point_cloud_range`.  
2. **Bảng file** kiến trúc vs `.pth` (mục 1.1).  
3. **Định nghĩa học có giám sát** \(f_\theta\) (mục 2.1).  
4. **Chamfer + AP + mAP** (mục 3) + bảng số B1 + ghi rõ *nuScenes mini*.  
5. **Giới hạn**: chưa giải occlusion; camera khác loại → domain gap (mục 4).  
6. **Liên kết đề tài**: B2/B3/B5/B6 là các bước giải quyết khoảng trống của opensource camera-only.

Paper nên cite:

- MapTR (ICLR 2023): arXiv 2208.14437  
- MapTRv2: arXiv 2308.05736  

---

## 6. Một đoạn có thể chép gần nguyên vào báo cáo

> Trong thí nghiệm baseline E1, chúng tôi sử dụng kiến trúc MapTRv2 (ResNet-50 + mã hóa BEV kiểu LSS + bộ giải mã Transformer) với trọng số chính thức đã huấn luyện trên nuScenes. Mạng học ánh xạ từ ảnh đa camera và thông số hiệu chỉnh sang các đường vector làn (`divider`, `ped_crossing`, `boundary`) trong hệ tọa độ xe. Chất lượng được đo bằng mAP theo khoảng cách Chamfer giữa polyline dự đoán và đáp án bản đồ. Kết quả trên tập nuScenes mini val đạt mAP ≈ 0.80; đây là chỉ số trên dữ liệu công khai có đáp án, không phải trên Phenikaa. Baseline camera-only chưa giải quyết triệt để tình huống camera bị che khuất hay vạch mờ — các hạn chế này được xử lý ở các thí nghiệm tiếp theo (fusion LiDAR, tích lũy theo Pose, và tùy chọn prior bản đồ điểm). Khi triển khai sang Phenikaa, sự khác biệt về số lượng camera và đặc biệt nhánh fisheye tạo domain gap về mô hình chiếu; do đó bắt buộc dùng bộ calib Phenikaa và chấp nhận đánh giá định tính trên dữ liệu Việt Nam cho đến khi có đáp án làn cục bộ.
