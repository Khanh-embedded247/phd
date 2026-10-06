# Vector HD Map Rules for Phenikaa MapTR

File nay tom tat quy tac ve GT va quy tac export prediction sang `.osm` de dung thong nhat trong benchmark MapTR Phenikaa.

Tai lieu chi tiet ve GT nam o:

```text
/home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/GT_OSM_DRAWING_GUIDE.md
```

## 1. Muc tieu

Pipeline hien tai tach 2 loai doi tuong:

- Perception vector: cac doi tuong camera/LiDAR co the quan sat truc tiep va MapTR can hoc.
- Topology/logic vector: cac doi tuong do nguoi ve de xe chay duoc trong HD map, nhung sensor khong nhin thay truc tiep.

MapTR giai doan nay chi train perception vector. Topology/logic vector co the giu trong GT OSM, nhung loader se bo qua khi train.

## 2. Bang class perception de model hoc

| Model class | Primitive khuyen nghi | type | subtype | semantic_class trong OSM | Ghi chu |
|---|---|---|---|---|---|
| `lane_divider` | LineString | `line_thin` | `dashed` hoac `solid` | `lane_divider` | Vach phan lan that tren mat duong |
| `road_edge_marking` | LineString | `line_thin` | `solid` | `road_edge_marking` | Vach mep trang that, khong gop voi curb |
| `stop_line` | LineString | `stop_line` | `solid` | `stop_line` | Vach dung that |
| `ped_crossing` | Lanelet relation hoac LineString phu | `lanelet`/`pedestrian_marking` | `crosswalk` | `ped_crossing` | Vach/zone nguoi di bo qua duong |
| `boundary` | LineString | `road_border` | `solid` | `road_boundary` | Bo via, curb, mep ket thuc mat duong that |
| `speed_bump` | Polygon la chinh, LineString neu can vector | `speed_bump` | `solid` hoac rong | `speed_bump` | Go giam toc |

Quy tac quan trong: `semantic_class` la nhan hoc chinh. Khong chi dua vao `type/subtype`, vi `line_thin + solid` co the la `lane_divider` hoac `road_edge_marking`.


## 2.1 Bang tag OSM chuan sau khi export prediction

File prediction sau postprocess mac dinh phai gon nhu GT perception, khong ghi tag debug. Moi way chi nen co cac tag semantic chinh sau:

| Model class | type | subtype | semantic_class | colour | Ghi chu |
|---|---|---|---|---|---|
| `lane_divider` | `line_thin` | `dashed` | `lane_divider` | `white` | Prediction divider xuat thanh vach phan lan |
| `road_edge_marking` | `line_thin` | `solid` | `road_edge_marking` | `white` | Vach mep trang |
| `stop_line` | `stop_line` | `solid` | `stop_line` | `white` | Duoc fit rieng thanh doan thang ngan |
| `ped_crossing` | `pedestrian_marking` | `crosswalk` | `ped_crossing` | `white` | Neu model co prediction class nay |
| `boundary` | `road_border` | `solid` | `road_boundary` | rong | Bo via/curb/road border |
| `speed_bump` | `speed_bump` | `solid` | `speed_bump` | rong | Neu model co prediction class nay |

Cac tag debug sau chi duoc ghi khi bat `debug_tags: true` trong `pipeline.yaml`:

```text
source=...
model_class=...
score=...
sample_token=...
pred_index=...
```

Mac dinh `debug_tags: false` de file `.osm` sau export giong quy chuan ve HD map, de mo bang Vector Map Builder khong bi roi tag.

## 3. Bang topology/logic khong train perception luc dau

| Doi tuong | Primitive | type | subtype | semantic_class | Cach dung |
|---|---|---|---|---|---|
| Duong chia lan ao | LineString | `virtual` | `dashed` | `virtual_lane_divider` | Chia lane logic khi ngoai doi khong co vach son |
| Bien ao | LineString | `virtual` | `solid` | `virtual_boundary` | Tao hanh lang chay khi khong co curb/vach mep that |
| Bien ao nhap/tach/cua | LineString | `virtual` | `solid` | `virtual_transition_boundary` | Vung chuyen tiep, giao lo, cua, nhap/tach lan |
| Centerline dieu huong | LineString | `virtual` | `centerline` | `centerline` | Huong di ben trong lanelet |
| Lanelet ao | Lanelet relation | `lanelet` | `road` | `virtual_lanelet` | Gom chung giao lo, thang, cua trai/phai, quay dau, nhap/tach |
| Lane xe that/topology | Lanelet relation | `lanelet` | `road` | `drivable_lane` | Topology HD map, khong train MapTR luc dau |
| Via he | Lanelet relation | `lanelet` | `walkway` | `sidewalk` | Topology nguoi di bo |

## 4. Quy tac ve GT khi duong Viet Nam thieu vach

- Neu co vach son/curb/road edge that, ve class perception that.
- Neu vach bi mo nhung nguoi gan nhan van xac dinh duoc, van ve class perception that va co the them `visibility=low`.
- Neu ngoai doi khong co vach, khong gan nhan thanh `lane_divider`, `road_edge_marking`, hay `boundary` chi vi planner can duong. Hay ve `virtual_*` hoac `virtual_lanelet`.
- Giao lo, cua, quay dau, nhap/tach lan nen dung `virtual_lanelet` va `centerline`, khong ep MapTR phai hoc cac duong sensor khong quan sat duoc.

## 5. Quy tac export prediction sang OSM

File prediction co 2 muc dich khac nhau:

- `predicted_vector_map_graph.osm`: giu nhieu vector goc hon, dung de xem model dang du doan gi.
- `predicted_vector_map_thin.osm`: loc bot vector trung lap, dung de xem ban do gon hon.

`thin` khong duoc tao topology moi. No chi duoc:

- Loc cac vector diem qua it hoac score thap.
- Gom diem/vector cung class vao spatial graph.
- Chi tao edge neu edge do den tu segment ma model da du doan.
- Rut moi connected component thanh path dai nhat de bien bo net day thanh mot net mong.
- Voi `stop_line`, `ped_crossing`, `speed_bump`, chi dedup overlap/trung lap; khong noi dai.

`thin` khong duoc:

- Noi cac endpoint xa nhau trong giao lo.
- Noi cac nhanh re/cua thanh mot duong dai.
- Trung binh tat ca diem theo truc xe chay de tao polyline moi.
- Doi `semantic_class` thanh `sample_token` hay metadata khac.

Metadata phu neu can debug phai ghi thanh tag rieng:

```text
source=maptr_prediction_merged
model_class=...
score=...
sample_token=...
pred_index=...
```

## 6. Quy tac merge trong file 21

Mac dinh script `21_postprocess_predictions_to_thin_osm.py` dung:

```text
merge_mode=spatial_graph
```

Nghia la:

- `lane_divider`, `road_edge_marking`, `boundary`: gom bo prediction day dac thanh cell graph theo tung class.
- Graph chi noi cac cell neu segment do tung xuat hien trong prediction goc.
- Moi connected component lay path dai nhat lam net dai dien, giup tranh ca hai loi: qua day nhu raw prediction va noi cheo theo truc xe.
- `stop_line`: khong dua vao graph dai; gom cum gan nhau roi fit thanh mot doan thang ngan dai dien.
- `ped_crossing`, `speed_bump`: chi dedup overlap/trung lap; khong noi dai.
- Moi output way van giu `type/subtype/semantic_class` theo bang GT/export table.

Cac che do phu:

```text
merge_mode=geometry
```

chi dedup representative goc. An toan nhung thuong van day vi prediction moi frame lech nhau mot chut.

```text
merge_mode=station_lateral
```

la cach cu de so sanh thuc nghiem. Che do nay co nguy co sinh duong cheo o giao lo/cua vi no gom diem theo quang duong xe va lateral bin.

## 7. Cau lenh nen chay sau train

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/11_infer_visualize_maptr_phenikaa.py
/home/khanh247/miniconda3/envs/maptr/bin/python /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/22_export_prediction_osm_graph.py
/home/khanh247/miniconda3/envs/maptr/bin/python /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/21_postprocess_predictions_to_thin_osm.py
/home/khanh247/miniconda3/envs/maptr/bin/python /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/16_visualize_maptr_paper_style.py
```

Hoac dung orchestrator:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python /home/khanh247/Documents/Survey/phd/phenikaa/benchmark_maptr/30_run_pipeline_from_config.py --steps infer,export,visualize
```
