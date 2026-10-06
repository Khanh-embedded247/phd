# 04. Outputs And Cleanup

Scenario hien tai:

```text
outputs/only_camera/road_work_traffic/
```

## Nen giu

| Path | Ly do |
|---|---|
| `work_dirs/maptr_12cam_osm/epoch_4.pth` | Bo nao model da hoc |
| `phenikaa_maptr_infos_full.pkl` | Manifest de infer/visualize lai |
| `final_run/no_metric_full_data/predictions.json` | Export/visualize lai khong can infer |
| `predicted_vector_map_graph.osm` | Prediction graph de debug |
| `predicted_vector_map_thin.osm` | Output HD map gon |
| `visualizations/professional/` | Anh bao cao/demo |
| `qa/` | Bang chung QA |

## Co the xoa neu can tiet kiem

| Path | Ghi chu |
|---|---|
| `images_undistorted/` | Cache, tao lai tu raw camera |
| `lidar_synced_to_camera/` | Cache, tao lai tu raw LiDAR + trajectory |
| `_intermediate/` | Xoa thi muon export/visualize raw phai infer lai, tru khi giu `predictions.json` |
| `epoch_2.pth`, `epoch_3.pth` | Checkpoint cu |
| `ckpts/maptr_init_12cam_partial.pth` | Chi can train lai tu init |
| `*.log`, `*.log.json` | Giu log cuoi neu can bao cao |
