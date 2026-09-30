# Phenikaa Vector HD Map Benchmark

Folder này chứa pipeline benchmark từng bước cho bài toán:

```text
raw camera + raw/deskew LiDAR + trajectory
-> đồng bộ sensor
-> đưa dữ liệu về hệ map
-> fusion camera-LiDAR
-> tạo vector HD map / lanelet
-> benchmark theo scenario
```

## Quy Ước Hệ Tọa Độ

Hệ cuối cùng để benchmark/vector map:

```text
MAP frame từ dump/traj_lidar.txt
```

`traj_lidar.txt` có format:

```text
timestamp tx ty tz qx qy qz qw
```

Trong pipeline này, mỗi dòng được hiểu là:

```text
T_map_lidar
```

tức pose của LiDAR/xe trong hệ map tại timestamp tương ứng.

`lidar_at_t_cam` chỉ là hệ trung gian để đồng bộ LiDAR về đúng thời điểm camera
trước khi project lên ảnh.

`T_geo_world.txt` chỉ dùng sau này nếu cần export map sang hệ địa lý. Nó không
phải bước chính cho camera-LiDAR sync.

## Các Bước Hiện Có

### 01_build_index.py

Tạo file index JSONL cho scenario.

Mỗi dòng index là một sample benchmark:

```text
camera image path
camera timestamp
nearest deskew LiDAR path
deskew LiDAR timestamp
traj_lidar.txt path
```

Chạy:

```bash
conda run -n phenikaa python /home/khanh247/Documents/Survey/phd/phenikaa/benchmark/01_build_index.py
```

Output mặc định:

```text
/home/khanh247/Documents/Survey/phd/phenikaa/outputs/benchmark/Normal/index.jsonl
```

### 02_project_lidar_to_camera_time.py

Dùng để kiểm tra đồng bộ và calib bằng ảnh overlay.

Pipeline:

```text
ảnh raw
-> undistort ảnh
-> nội suy T_map_lidar(t_cam)
-> nội suy T_map_lidar(t_lidar)
-> đưa point LiDAR về lidar_at_t_cam
-> LiDAR -> camera
-> project lên ảnh đã undistort
```

Chạy frame bất kỳ:

```bash
conda run -n phenikaa python /home/khanh247/Documents/Survey/phd/phenikaa/benchmark/02_project_lidar_to_camera_time.py --index 2656
```

### 03_accumulate_lidar_map.py

Tích lũy nhiều frame LiDAR về hệ MAP.

Đây là bước phục vụ bài toán cuối:

```text
multi-frame LiDAR evidence trong MAP
-> detect/fusion/vectorize
-> vector HD map
```

Chạy:

```bash
conda run -n phenikaa python /home/khanh247/Documents/Survey/phd/phenikaa/benchmark/03_accumulate_lidar_map.py --index 2656 --window-sec 0.2
```

Output là `.npz` chứa `points_map`.

### 04_project_all_images.py

Chạy projection cho nhiều ảnh để tạo QA report cho scenario.

Các tham số hay dùng đã được đặt ở block `USER CONFIG` đầu file:

```text
CONFIG_SEQUENCE_DURATION_SEC
CONFIG_MAX_OVERLAYS
CONFIG_START_INDEX
CONFIG_MAX_FRAMES
```

Phương pháp sync trong file 04:

```text
1. Với ảnh tại t_cam
2. Chọn LiDAR deskew frame gần nhất t_lidar từ index.jsonl
3. Nội suy T_map_lidar(t_cam)
4. Nội suy/đọc T_map_lidar(t_lidar)
5. Warp cloud deskew từ t_lidar sang t_cam bằng ego trajectory
6. Project lên ảnh đã undistort
```

Mặc định:

```text
- xử lý toàn bộ index.jsonl
- ghi projection_report.csv cho tất cả frame
- chỉ lưu tối đa 100 overlay JPG cách đều nhau
- lưu frame overlay liên tiếp trong 20s đầu vào sequence_20.0s/frames
```

CSV dùng để xem định lượng sau này:

```text
frame nào lệch timestamp bao nhiêu
LiDAR frame nào được dùng
số point chiếu vào ảnh
tỉ lệ point lọt ảnh
đường dẫn overlay nếu có lưu
```

Chạy nhanh để test:

```bash
conda run -n phenikaa python /home/khanh247/Documents/Survey/phd/phenikaa/benchmark/04_project_all_images.py --sequence-duration-sec 2 --max-frames 70 --max-overlays 0
```

Tạo frame overlay liên tiếp trong 20 giây đầu:

```bash
conda run -n phenikaa python /home/khanh247/Documents/Survey/phd/phenikaa/benchmark/04_project_all_images.py --sequence-duration-sec 20 --max-overlays 100
```

Chạy toàn bộ scenario:

```bash
conda run -n phenikaa python /home/khanh247/Documents/Survey/phd/phenikaa/benchmark/04_project_all_images.py --max-overlays 100
```

### 05_make_projection_video.py

Ghép video từ các frame overlay liên tiếp đã tạo ở bước 04.

Các tham số video cũng nằm trong block `USER CONFIG` đầu file:

```text
CONFIG_SEQUENCE_DURATION_SEC
CONFIG_VIDEO_FPS
CONFIG_OUTPUT_NAME
```

File này không project lại LiDAR. Nó chỉ đọc:

```text
outputs/benchmark/Normal/projection_all/sequence_20.0s/frames/frame_*.jpg
```

Rồi ghép thành MP4, nên video không bị giật do skip/project trong lúc ghi.

Chạy:

```bash
conda run -n phenikaa python /home/khanh247/Documents/Survey/phd/phenikaa/benchmark/05_make_projection_video.py --fps 30
```

Output mặc định:

```text
outputs/benchmark/Normal/videos/projection_sequence_20s.mp4
```

### 06_inspect_glim_map.py

Đọc map PCD đã tạo bằng GLIM và tạo ảnh nhìn từ trên xuống để kiểm tra nền map.

Input mặc định:

```text
data/Normal/map - Cloud.pcd
```

Output:

```text
outputs/benchmark/Normal/glim_map/stats.json
outputs/benchmark/Normal/glim_map/bev_density.png
outputs/benchmark/Normal/glim_map/bev_height_max.png
outputs/benchmark/Normal/glim_map/bev_height_range.png
outputs/benchmark/Normal/glim_map/bev_intensity_mean.png
```

Chạy:

```bash
conda run -n phenikaa python /home/khanh247/Documents/Survey/phd/phenikaa/benchmark/06_inspect_glim_map.py
```

Ý nghĩa:

```text
bev_density        : chỗ nào có nhiều point
bev_height_max     : độ cao lớn nhất mỗi ô
bev_height_range   : chênh cao trong mỗi ô, hữu ích để nhìn curb/vỉa hè
bev_intensity_mean : phản xạ trung bình, có thể giúp nhìn vạch đường nếu rõ
```
