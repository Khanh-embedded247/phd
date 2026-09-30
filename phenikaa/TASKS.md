# Phenikaa — việc cần làm (gọn)

Đọc trước (theo thứ tự):

1. [docs/HIEU_RO.md](docs/HIEU_RO.md) — hiểu khái niệm  
2. **[docs/LO_TRINH.md](docs/LO_TRINH.md) — lộ trình từng bước + sơ đồ (sửa ở đây)**  
3. [docs/mau/](docs/mau/) — hình mẫu  

## Làm lần lượt

1. [x] **B0** Kiểm tra chỉnh cảm biến (`tutorials` 05 → 11)
2. [x] **B1** Baseline camera trên nuScenes → trọng số + điểm
3. [x] **B2** Baseline camera + LiDAR trên nuScenes → so với B1
4. [ ] **B3** Chạy thử Phenikaa → `outputs/phenikaa_vis/`
   - Hướng dẫn chạy: [docs/CHAY_B3_PHENIKAA.md](docs/CHAY_B3_PHENIKAA.md)
5. [ ] **B4** Thử vạch mờ (nuScenes)
6. [ ] **B5** Ghép nhiều khung bằng Pose/
7. [ ] **B6** (Sau) bản đồ điểm đúng khu → prior
8. [ ] **B7** (Tuỳ chọn) vẽ tay đáp án làn Phenikaa

Chi tiết từng bước: [docs/LO_TRINH.md](docs/LO_TRINH.md)

## Không làm / không dùng lúc này

- `Label/` xe làm đáp án làn  
- `data/phenikaa/maps/*_candidate*` (sai khu)  
- Radar / can_bus gắn vào Phenikaa  
- Bắt buộc có đáp án làn Phenikaa trước khi demo (B3)  
