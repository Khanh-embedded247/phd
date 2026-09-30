# =============================================================================
# B2 — MapTR tiny fusion (camera+LiDAR), eval GPU 8GB + nuScenes mini
# Upstream: maptr_tiny_fusion_24e.py (LO_TRINH B2)
# =============================================================================
_base_ = './maptr_tiny_fusion_24e.py'

# Dùng cùng pkl mini như B1 (có field annotation → Offline dataset)
dataset_type = 'CustomNuScenesOfflineLocalMapDataset'
data_root = 'data/nuscenes/'
ann_train = data_root + 'nuscenes_map_infos_temporal_train.pkl'
ann_val = data_root + 'nuscenes_map_infos_temporal_val.pkl'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=ann_train,
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_val,
        samples_per_gpu=1,
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_val,
    ),
)
