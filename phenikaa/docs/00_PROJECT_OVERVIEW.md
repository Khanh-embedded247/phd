# 00. Project Overview

Muc tieu: xay dung pipeline tao **vector HD map** cho du lieu Phenikaa. Giai doan hien tai dung MapTR camera-only de hoc cac vector co the quan sat truc tiep, sau do export/postprocess sang OSM.

## Dau vao

- 12 camera quanh xe.
- `dump/traj_lidar.txt` lam pose/trajectory.
- `dump/frames/laz/` dung cho QA/cache/fusion sau nay.
- GT `.osm` ve theo quy tac semantic HD map.

## Dau ra

- `predictions.json/pkl`: prediction raw cua model.
- `predicted_vector_map_graph.osm`: vector prediction dang graph, gan raw.
- `predicted_vector_map_thin.osm`: vector HD map da lam mong.
- `visualizations/professional/`: hinh bao cao/demo.

## Class model dang hoc

```python
["lane_divider", "road_edge_marking", "stop_line", "ped_crossing", "boundary", "speed_bump"]
```

`boundary` trong model tuong ung `semantic_class=road_boundary` trong OSM.
