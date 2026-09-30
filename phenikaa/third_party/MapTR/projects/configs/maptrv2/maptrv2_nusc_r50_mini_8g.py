# =============================================================================
# [TIẾNG VIỆT] Config "nhẹ" cho GPU 8GB — file BẠN đang chạy.
# Không chứa kiến trúc đầy đủ: kế thừa gần như hết từ maptrv2_nusc_r50_24ep.py
# Chỉ ghi đè: batch=1, tắt one2many, tắt aux_seg, epochs=2.
# Muốn xem dataset/pipeline/model → mở file 24ep.py
# =============================================================================
_base_ = './maptrv2_nusc_r50_24ep.py'

# Lightweight settings for single 8GB GPU (e.g. RTX 3070 Ti) + nuScenes mini smoke test.
# Disables expensive one-to-many matching and aux segmentation; keeps bs=1.

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)

model = dict(
    pts_bbox_head=dict(
        num_vec_one2many=0,
        k_one2many=0,
        aux_seg=dict(
            use_aux_seg=False,
            bev_seg=False,
            pv_seg=False,
            seg_classes=1,
            feat_down_sample=32,
            pv_thickness=1,
        ),
    ),
)

# Faster smoke loop on mini
total_epochs = 2
evaluation = dict(interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(max_keep_ckpts=1, interval=1)
log_config = dict(interval=10)
