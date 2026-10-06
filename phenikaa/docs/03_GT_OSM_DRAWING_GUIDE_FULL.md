# Phenikaa GT OSM Drawing Guide

File nay la quy uoc ve GT `.osm` cho MapTR/HD map Phenikaa.

Muc tieu:
- GT `.osm` van la HD map day du de dung cho lanelet/topology.
- MapTR chi hoc cac vector co the quan sat truc tiep tu camera/LiDAR.
- Cac duong ao/topology duoc giu trong GT, nhung perception loader se bo qua khi train MapTR giai doan dau.

## 1. Nguyen tac chung

Neu doi tuong co dau hieu vat ly ngoai doi:
- vach son
- mep via he/curb
- vach dung
- vach qua duong
- go giam toc

thi ve bang semantic class that.

Neu doi tuong do ban tao ra de xe hieu topology, nhung sensor khong nhin thay truc tiep:
- duong chia lan ao
- bien lan ao
- duong noi qua giao lo
- lanelet chuyen tiep
- lanelet tao tu bien ao

thi ve bang `virtual_*` hoac lanelet topology. Khong dung cac object nay de train MapTR perception o giai doan dau.

Luon uu tien gan tag:

```text
semantic_class=...
```

Khong chi dua vao `type/subtype`, vi cung `type=line_thin, subtype=solid` co the la vach phan lan lien hoac vach mep trang.

## 1.1 Bang ve nhanh de dung ngay

Neu chi can mo Vector Map Builder va ve GT nhat quan, dung bang nay truoc.

| Doi tuong can ve | Primitive | type | subtype | semantic_class | Ghi chu |
|---|---|---|---|---|---|
| Vach phan lan dut that | LineString | `line_thin` | `dashed` | `lane_divider` | Train MapTR |
| Vach phan lan lien that | LineString | `line_thin` | `solid` | `lane_divider` | Train MapTR |
| Vach mep trang that | LineString | `line_thin` | `solid` | `road_edge_marking` | Train MapTR; phan biet voi lane_divider bang `semantic_class` |
| Bo via/curb/mep mat duong that | LineString | `road_border` | `solid` | `road_boundary` | Train MapTR duoi class `boundary` |
| Vach dung | LineString | `stop_line` | `solid` | `stop_line` | Train MapTR |
| Vach qua duong | Lanelet relation | `lanelet` | `crosswalk` | `ped_crossing` | Train MapTR; relation nen co member `left/right` |
| Go giam toc | LineString/Polygon | `speed_bump` | `solid` hoac rong | `speed_bump` | Train MapTR neu parser doc duoc primitive do |
| Chia lan ao | LineString | `virtual` | `dashed` | `virtual_lane_divider` | Khong train MapTR luc dau |
| Bien ao | LineString | `virtual` | `solid` | `virtual_boundary` | Khong train MapTR luc dau |
| Bien ao vung nhap/tach/cua | LineString | `virtual` | `solid` | `virtual_transition_boundary` | Khong train MapTR luc dau |
| Lane xe chay thuong | Lanelet relation | `lanelet` | `road` | `drivable_lane` | Topology, khong train MapTR luc dau |
| Centerline dieu huong | LineString | `virtual` | `centerline` | `centerline` | Topology, khong train MapTR luc dau |
| Lanelet ao: giao lo/thang/cua/trai/phai/quay dau/nhap tach | Lanelet relation | `lanelet` | `road` | `virtual_lanelet` | Topology, khong train MapTR luc dau |

Checklist nhanh:

- Co son/vat the that ngoai doi thi ve semantic that.
- Khong co son/vat the that ma chi can duong cho planner thi ve `virtual_*`.
- Khong dung `virtual_*` de train MapTR perception giai doan dau.
- `line_thin + solid` bat buoc phai co `semantic_class` de biet la `lane_divider` hay `road_edge_marking`.
- Vach mo nhung con xac dinh duoc thi van ve semantic that, co the them `visibility=low`.
- Cac lanelet ao trong giao lo/cua/quay dau/nhap tach quy chung thanh `virtual_lanelet`; dung `centerline` de mo ta huong di ben trong.

## 2. Cac class cho MapTR hoc

Day la cac class perception. Model duoc train de du doan cac doi tuong nay.

```python
map_classes = [
    "lane_divider",
    "road_edge_marking",
    "stop_line",
    "ped_crossing",
    "boundary",
    "speed_bump",
]
```

Trong code, `boundary` tuong ung voi `semantic_class=road_boundary`.

### 2.1 Lane Divider

Dung cho vach phan lan that tren mat duong.

Vach dut:

```text
Primitive: LineString
type: line_thin
subtype: dashed
semantic_class: lane_divider
```

Vach lien:

```text
Primitive: LineString
type: line_thin
subtype: solid
semantic_class: lane_divider
```

Hai vach lien, neu editor/parser ho tro:

```text
Primitive: LineString
type: line_thin
subtype: solid_solid
semantic_class: lane_divider
```

Khong ve `lane_divider` neu ngoai doi khong co vach son that. Khi can chia lan de topology, dung `virtual_lane_divider`.

### 2.2 Road Edge Marking

Dung cho vach mep trang that tren mat duong, khong gop voi bo via/curb.

```text
Primitive: LineString
type: line_thin
subtype: solid
semantic_class: road_edge_marking
```

Neu vach mep trang bi mo nhung van xac dinh duoc bang mat nguoi, van ve `road_edge_marking`.

Co the them tag phu neu muon:

```text
visibility: low
```

### 2.3 Road Boundary

Dung cho mep ket thuc mat duong, bo via, curb, road border vat ly.

```text
Primitive: LineString
type: road_border
subtype: solid
semantic_class: road_boundary
```

Trong model class, doi tuong nay se thanh:

```text
boundary
```

Neu khong co curb/via he/mep duong that, khong ve `road_boundary`; khi can bien cho planner, ve `virtual_boundary`.

### 2.4 Stop Line

Dung cho vach dung that.

```text
Primitive: LineString
type: stop_line
subtype: solid
semantic_class: stop_line
```

Khong gop stop line vao `lane_divider`.

### 2.5 Pedestrian Crossing

Khuyen nghi ve bang lanelet crosswalk relation de loader nhan dung `ped_crossing`.

```text
Primitive: Lanelet relation
type: lanelet
subtype: crosswalk
semantic_class: ped_crossing
```

Member way:
- role `left`
- role `right`

Neu editor can them way marking rieng, co the dung:

```text
Primitive: LineString
type: pedestrian_marking
subtype: crosswalk
semantic_class: ped_crossing
```

Nhung loader hien tai uu tien nhan crosswalk tu relation `type=lanelet, subtype=crosswalk`.

### 2.6 Speed Bump

Neu ve bang LineString:

```text
Primitive: LineString
type: speed_bump
subtype: solid
semantic_class: speed_bump
```

Neu editor ve duoc polygon va can bao vung:

```text
Primitive: Polygon
type: speed_bump
subtype: solid
semantic_class: speed_bump
```

Hien tai loader MapTR dang doc way/linestring la chinh. Neu dung polygon, can dam bao pipeline co parser polygon hoac convert polygon thanh line/vector phu hop.

## 3. Cac doi tuong topology/ao khong train MapTR luc dau

Cac doi tuong nay van nen co trong GT `.osm` de HD map day du, nhung perception loader bo qua khi train MapTR.

### 3.1 Virtual Lane Divider

Dung khi can chia lan logic nhung ngoai doi khong co vach son.

```text
Primitive: LineString
type: virtual
subtype: dashed
semantic_class: virtual_lane_divider
```

### 3.2 Virtual Boundary

Dung cho bien lan/hanh lang xe chay ao khi khong co curb, road border, hoac vach mep that.

```text
Primitive: LineString
type: virtual
subtype: solid
semantic_class: virtual_boundary
```

### 3.3 Virtual Transition Boundary

Dung trong vung nhap/tach lan, chuyen tiep so lan, hoac cua/giao lo can bien ao rieng.

```text
Primitive: LineString
type: virtual
subtype: solid
semantic_class: virtual_transition_boundary
```

### 3.4 Virtual Road Boundary

Dung khi can mo ta road boundary ao thay cho road boundary vat ly khong ton tai/khong quan sat duoc.

```text
Primitive: LineString
type: virtual
subtype: solid
semantic_class: virtual_road_boundary
```

### 3.5 Drivable Lane

Lanelet xe chay thong thuong.

```text
Primitive: Lanelet relation
type: lanelet
subtype: road
semantic_class: drivable_lane
```

### 3.6 Centerline

Duong dieu huong o giua lanelet/lane ao. Dung cho planner va topology, khong train MapTR perception luc dau.

```text
Primitive: LineString
type: virtual
subtype: centerline
semantic_class: centerline
```

Centerline co the duoc them vao lanelet relation voi role `centerline` neu editor/Lanelet2 workflow can.

### 3.7 Virtual Lanelet

Quy chung moi lanelet ao vao mot semantic class nay: di thang qua giao lo, cua trai, cua phai, quay dau, nhap/tach lan, chuyen tiep so lan, lane tao tu bien ao.

```text
Primitive: Lanelet relation
type: lanelet
subtype: road
semantic_class: virtual_lanelet
```

Neu can phan biet thao tac cho planner, them tag phu tuy chon nhu:

```text
maneuver: straight / left / right / u_turn / merge / split
```

### 3.8 Sidewalk

Neu can ve via he cho HD map.

```text
Primitive: Lanelet relation
type: lanelet
subtype: walkway
semantic_class: sidewalk
```

## 4. Xu ly duong Viet Nam thieu vach/mo vach

### Truong hop A: vach son mo nhung con xac dinh duoc

Van ve class that:

```text
lane_divider
road_edge_marking
stop_line
```

Co the them:

```text
visibility: low
```

### Truong hop B: khong co vach phan lan that

Khong ve `lane_divider`.

Neu can chia lan cho planner:

```text
semantic_class: virtual_lane_divider
type: virtual
subtype: dashed
```

### Truong hop C: khong co vach mep trang nhung co curb/via he

Ve:

```text
semantic_class: road_boundary
type: road_border
subtype: solid
```

Khong ve `road_edge_marking`.

### Truong hop D: khong co curb/via he/road border ro rang

Khong ve `road_boundary`.

Neu can bien de tao lanelet:

```text
semantic_class: virtual_boundary
type: virtual
subtype: solid
```

### Truong hop E: giao lo/cua/quay dau/nhap tach lan

Khong co gang ve lane divider that neu ngoai doi khong co vach son.

Dung cac object topology:

```text
virtual_boundary
virtual_transition_boundary
centerline
virtual_lanelet
```

## 5. Quy tac uu tien khi ve

Thu tu uu tien:

1. Neu co object that nhin thay duoc, ve semantic that.
2. Neu object that bi mo nhung con xac dinh duoc, van ve semantic that va them `visibility=low` neu can.
3. Neu khong co object that, nhung HD map can duong de xe chay, ve `virtual_*`.
4. Lanelet relation dung de mo ta topology, khong thay the marking/road border that.
5. Khong dung cung mot cap `type/subtype` de phan biet class neu khong co `semantic_class`.

## 6. Bang tom tat nhanh

| Doi tuong | Primitive | type | subtype | semantic_class | Train MapTR |
|---|---|---|---|---|---|
| Vach phan lan dut | LineString | line_thin | dashed | lane_divider | yes |
| Vach phan lan lien | LineString | line_thin | solid | lane_divider | yes |
| Vach mep trang | LineString | line_thin | solid | road_edge_marking | yes |
| Bo via/curb/mep duong | LineString | road_border | solid | road_boundary | yes, as boundary |
| Vach dung | LineString | stop_line | solid | stop_line | yes |
| Vach qua duong | Lanelet | lanelet | crosswalk | ped_crossing | yes |
| Go giam toc | LineString/Polygon | speed_bump | solid | speed_bump | yes |
| Chia lan ao | LineString | virtual | dashed | virtual_lane_divider | no |
| Bien ao | LineString | virtual | solid | virtual_boundary | no |
| Bien chuyen tiep ao | LineString | virtual | solid | virtual_transition_boundary | no |
| Lane xe chay that/topology thuong | Lanelet | lanelet | road | drivable_lane | no |
| Centerline dieu huong | LineString | virtual | centerline | centerline | no |
| Lanelet ao giao lo/cua/quay dau/nhap tach | Lanelet | lanelet | road | virtual_lanelet | no |
| Via he | Lanelet | lanelet | walkway | sidewalk | no |
