# MapTR Pipeline Scripts

Day la cac script Python chinh cua pipeline Phenikaa MapTR. Moi file co the chay truc tiep bang Python.

Thu tu chay tu raw dump den vector map:

1. `01_check_raw_scenario.py`: kiem tra raw scenario, anh, lidar, pose, timestamp.
2. `02_check_pose_quality.py`: kiem tra chat luong pose/ego motion.
3. `04_check_gt_osm_semantic.py`: kiem tra GT OSM theo rule semantic.
4. `03_build_phenikaa_infos.py`: tao `phenikaa_maptr_infos_full.pkl` tu raw data.
5. `06_filter_infos_with_osm_gt.py`: loc sample co GT vector hop le.
6. `07_make_phenikaa_train_config.py`: sinh/cap nhat config train MapTR.
7. `08_make_12cam_init_checkpoint.py`: tao checkpoint init 12 camera, bo head cu.
8. `09_check_train_dataset_osm.py`: smoke test dataset/GT truoc train.
9. `10_train_maptr_phenikaa_osm.py`: train model.
10. `11_infer_visualize_maptr_phenikaa.py`: infer va render nhanh prediction.
11. `22_export_prediction_osm_graph.py`: export prediction thanh OSM graph.
12. `21_postprocess_predictions_to_thin_osm.py`: gom/lam mong vector chong lan.
13. `17_visualize_maptr_professional.py`: tao visualization chuyen nghiep cho bao cao.

Nen chay qua runner tong:

```bash
/home/khanh247/miniconda3/envs/maptr/bin/python scripts/maptr/30_run_pipeline_from_config.py --steps infer,export,visualize
```
