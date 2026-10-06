# Master Thesis Notes and Possible Q&A

Tai lieu nay giup giai thich ban chat viec dang lam, dong gop cua de tai,
va cac cau hoi co the gap khi bao ve.

## 1. Bai toan dang lam

Muc tieu la xay dung pipeline tao vector HD map cuc bo tu du lieu xe Phenikaa,
tap trung vao cac thanh phan co the quan sat bang camera:

```text
lane_divider
road_edge_marking
stop_line
ped_crossing
boundary
speed_bump
```

Mo hinh nen tang la MapTR: nhan anh da dong bo/undistort tu 12 camera, tao BEV
feature, va du doan cac vector map element trong he toa do local quanh xe.

## 2. Diem dong gop chinh

### 2.1 Chuyen MapTR sang du lieu Phenikaa 12 camera

Repo MapTR goc thiet ke cho dataset nhu nuScenes. Phan nay chuyen sang du lieu
Phenikaa:

```text
12 camera
calibration rieng
trajectory traj_lidar.txt
LiDAR timestamp rieng
GT OSM tu Vector Map Builder
```

Dong gop:

```text
tao infos.pkl theo schema MapTR
sync 12 camera
undistort anh
warp LiDAR ve timestamp camera
doi frame ego/base_link dung hon
```

### 2.2 Dinh nghia lai semantic class cho GT OSM Viet Nam

MapTR goc co class kha rong nhu `divider`. De phu hop duong Viet Nam va HD map,
de tai tach thanh:

```text
lane_divider
road_edge_marking
stop_line
ped_crossing
boundary
speed_bump
```

GT OSM co the van day du topology:

```text
virtual_lane_divider
virtual_boundary
virtual_transition_boundary
centerline
virtual_lanelet
drivable_lane
sidewalk
```

Nhung cac class topology/ao khong dua vao train perception giai doan dau.

### 2.3 Multi-scenario training

Thay vi train rieng tung scenario, pipeline ho tro nhieu cap:

```text
infos.pkl + gt.osm
```

Moi scenario giu OSM rieng vi moi scenario co he toa do map/trajectory rieng.
Config train dung `ConcatDataset` va `RepeatDataset` de hoc tu nhieu bo du lieu.

### 2.4 QA pipeline truoc train

Them cac buoc kiem tra:

```text
raw camera/LiDAR/trajectory QA
pose quality QA
GT OSM semantic QA
dataset GT check
```

Muc tieu la phat hien loi data/calib/GT truoc khi train ton thoi gian GPU.

## 3. Vi sao chi dung camera de train MapTR?

MapTR la bai toan vector map perception tu camera. Camera quan sat truc tiep:

```text
vach son
vach mep trang
vach dung
vach qua duong
go giam toc
mot phan mep duong/curb
```

LiDAR trong pipeline hien tai chu yeu dung cho:

```text
pose/trajectory
sync timestamp
kiem tra hinh hoc
ho tro debug/GT neu can
```

Khong dua point cloud vao model MapTR trong cau hinh hien tai. Neu dua LiDAR vao
mo hinh, bai toan se thanh multi-modal/fusion, can sua kien truc va train khac.

## 4. Gap lon: duong ao/topology khong nhin thay

Camera va LiDAR khong truc tiep nhin thay:

```text
centerline
virtual_lanelet
junction connector logic
lanelet qua giao lo
duong quay dau
duong noi nhap/tach lan
```

Do do pipeline tach thanh hai tang:

```text
Tang 1: MapTR hoc visible map elements
Tang 2: postprocess/topology reconstruction sinh duong ao va lanelet
```

Day la cach hop ly hon viec bat model hoc cac duong ao khong co bang chung sensor.

## 5. Cau hoi co the gap va cach tra loi

### Q1. Tai sao khong train model du doan truc tiep lanelet/topology?

Vi lanelet/topology nhu centerline, connector qua giao lo, lane ao khong phai
doi tuong quan sat truc tiep. Cung mot anh co the co nhieu cach noi topology
hop ly. Neu ep model hoc truc tiep, label se mang tinh quy uoc va de bi nhiem.
Vi vay de tai tach perception visible element va topology reconstruction.

### Q2. Vi sao tach `divider` thanh `lane_divider`, `road_edge_marking`, `stop_line`?

`divider` qua rong. Voi HD map Viet Nam, vach phan lan, vach mep trang va vach
dung co y nghia khac nhau cho planning. Tach class giup output map co semantic
ro hon va de tao lanelet/topology sau nay.

### Q3. Neu duong khong co vach ke thi model xu ly sao?

Model khong nen tu tao vach that neu ngoai doi khong co. Trong GT, doan nay
duoc ve bang `virtual_*` hoac `virtual_lanelet` cho topology. Giai doan
perception chi hoc cac doi tuong nhin thay. Sau prediction, module topology se
sinh bien ao/centerline dua tren rule, trajectory va lane width.

### Q4. Vi sao khong gop nhieu file OSM thanh mot file de train?

Moi scenario co he toa do map va trajectory rieng. OSM phai di cung infos/pose
cua scenario do. Gop OSM co the lam sai he toa do. Cach dung dung la multi-dataset:
moi dataset co `infos.pkl + gt.osm`, sau do ghep bang `ConcatDataset`.

### Q5. LiDAR co vai tro gi neu mo hinh chi dung camera?

LiDAR hien duoc dung de sync/kiem tra hinh hoc va co the ho tro GT. Script build
infos warp LiDAR ve timestamp camera de du lieu nhat quan. Tuy nhien input chinh
cua MapTR hien tai la anh camera. Neu muon dung LiDAR lam input model, can mo rong
kien truc sang fusion.

### Q6. Khac biet giua `road_edge_marking` va `boundary`?

`road_edge_marking` la vach mep trang son tren mat duong. `boundary` trong model
tuong ung `road_boundary`, la curb/bo via/mep vat ly cua mat duong. Hai doi tuong
co the gan nhau nhung y nghia HD map khac nhau.

### Q7. Speed bump la polygon thi MapTR hoc the nao?

MapTR hoc vector. Neu speed bump la polygon, loader can chuyen polygon/closed way
thanh closed polyline de lam vector GT. Nhung ve mat semantic, van giu:

```text
type=speed_bump
semantic_class=speed_bump
```

### Q8. Vi sao can undistort anh?

MapTR dung camera intrinsic pinhole trong projection/BEV transform. Anh fisheye
hoac anh co distortion neu khong undistort se lam sai hinh hoc giua anh, pose va
vector GT. Vi vay pipeline tao anh undistorted va luu K moi vao infos.pkl.

### Q9. Vi sao pose quality quan trong?

GT OSM nam trong he map, moi frame train se cat patch quanh xe theo pose. Neu pose
sai/yaw rung, vector GT se lech so voi anh, model hoc sai. Do do can QA pose truoc
train.

### Q10. Neu GT bi ve sai semantic_class thi hau qua gi?

Model hoc theo label GT. Neu `line_thin solid` luc thi la lane divider, luc thi la
road edge nhung thieu `semantic_class`, model se hoc nham. Vi vay GT QA bat buoc
kiem tra missing/unknown semantic truoc train.

### Q11. Dong gop khoa hoc/co ky thuat cua de tai nam o dau?

Dong gop nam o:

```text
thich nghi MapTR sang du lieu Phenikaa 12 camera
xay dung schema GT OSM phu hop HD map Viet Nam
tach perception visible element va topology ao
pipeline QA + build infos + train/infer/export OSM
multi-scenario training bang ConcatDataset/RepeatDataset
```

### Q12. Gioi han hien tai la gi?

Gioi han:

```text
chua fusion LiDAR vao model
topology reconstruction moi la huong tiep theo
du lieu GT con phu thuoc ve tay
class hiem nhu speed_bump/ped_crossing can them mau
duong khong vach can rule/trajectory de sinh virtual lanelet
```

## 6. Mot cau tom tat de bao ve

De tai khong chi train lai MapTR. Phan chinh la xay dung mot pipeline HD map
perception cho du lieu Phenikaa: tu sync/undistort camera, sync LiDAR/pose, chuan
hoa GT OSM, train multi-scenario 6 semantic class, den export vector OSM. Diem
quan trong la tach ro nhung gi sensor nhin thay de model hoc va nhung gi la
topology ao de xu ly bang rule/postprocess sau.
