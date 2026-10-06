import laspy
import numpy as np

# 1. Đường dẫn tới file LAZ của bạn
laz_path = "/home/khanh247/Documents/Survey/phd/phenikaa/data/Normal/LIDAR/LIDAR_TOP/1781509258-099941015.laz"

# 2. Đọc file
las = laspy.read(laz_path)

print("="*50)
print(f" THÔNG TIN TỔNG QUAN FILE: {laz_path}")
print("="*50)
# Phiên bản định dạng LAS (ví dụ: 1.2, 1.4) và Định dạng bản ghi điểm (Point Format)
print(f"Phiên bản LAS: {las.header.version}")
print(f"Point Format (Định dạng điểm): {las.header.point_format.id}")
print(f"Tổng số điểm LiDAR: {las.header.point_count:,}")

# Tọa độ bounding box (Giới hạn không gian của dữ liệu)
print(f"Tọa độ X (Min - Max): {las.header.x_min} -> {las.header.x_max}")
print(f"Tọa độ Y (Min - Max): {las.header.y_min} -> {las.header.y_max}")
print(f"Tọa độ Z / Cao độ (Min - Max): {las.header.z_min} -> {las.header.z_max}")

print("\n" + "="*50)
print(" DANH SÁCH CÁC TRƯỜNG DỮ LIỆU CÓ TRONG FILE (Dimensions)")
print("="*50)
# Liệt kê tất cả các thuộc tính mà mỗi điểm LiDAR trong file này sở hữu
for dim_name in las.point_format.dimension_names:
    print(f"- {dim_name}")

print("\n" + "="*50)
print(" ĐỌC DỮ LIỆU CHI TIẾT CỦA CÁC TRƯỜNG PHỔ BIẾN (Ví dụ 5 điểm đầu)")
print("="*50)

# Lấy tọa độ thực tế (đã áp dụng hệ số Scale và Offset từ Header)
x_coords = las.x[:5]
y_coords = las.y[:5]
z_coords = las.z[:5]

# Cường độ phản xạ tín hiệu (Intensity)
intensity = las.intensity[:5]

# Phân loại điểm (Classification: ví dụ 2 là mặt đất, 3 là thảm thực vật thấp, 5 là cây cao...)
classification = las.classification[:5]

# In thử nghiệm dữ liệu 5 điểm đầu tiên
for i in range(5):
    print(f"Điểm #{i+1}: X={x_coords[i]:.3f}, Y={y_coords[i]:.3f}, Z={z_coords[i]:.3f} | "
          f"Intensity={intensity[i]} | Class={classification[i]}")
