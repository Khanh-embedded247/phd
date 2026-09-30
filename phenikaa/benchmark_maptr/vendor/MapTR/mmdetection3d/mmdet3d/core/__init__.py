# Copyright (c) OpenMMLab. All rights reserved.
from .anchor import *  # noqa: F401, F403
from .bbox import *  # noqa: F401, F403
try:
    from .evaluation import *  # noqa: F401, F403
except Exception as exc:
    # Phenikaa/MapTR smoke tests do not need KITTI evaluation.
    # Old numba builds can fail with newer NumPy during this import.
    print(f"Warning: skip mmdet3d.core.evaluation import: {exc}")
from .points import *  # noqa: F401, F403
try:
    from .post_processing import *  # noqa: F401, F403
except Exception as exc:
    print(f"Warning: skip mmdet3d.core.post_processing import: {exc}")

    def box3d_multiclass_nms(*args, **kwargs):
        """Fallback de cac module 3D detection cu import duoc khi NMS op loi.

        MapTR vector-map inference khong dung ham nay. Neu pipeline nao that su
        can 3D NMS thi can sua/build lai extension post_processing.
        """
        raise ImportError(
            "box3d_multiclass_nms khong kha dung vi mmdet3d.core.post_processing "
            "khong import duoc. Hay build/sua NMS extension neu can 3D detection."
        )

    def circle_nms(*args, **kwargs):
        """Fallback cho cac detector 3D cu; MapTR khong dung ham nay."""
        raise ImportError(
            "circle_nms khong kha dung vi mmdet3d.core.post_processing "
            "khong import duoc. Hay build/sua NMS extension neu can 3D detection."
        )

    def merge_aug_bboxes_3d(*args, **kwargs):
        """Fallback cho test-time augmentation 3D; MapTR vector khong dung."""
        raise ImportError(
            "merge_aug_bboxes_3d khong kha dung vi mmdet3d.core.post_processing "
            "khong import duoc. Hay build/sua post_processing neu can 3D detection."
        )
from .utils import *  # noqa: F401, F403
from .visualizer import *  # noqa: F401, F403
try:
    from .voxel import *  # noqa: F401, F403
except Exception as exc:
    print(f"Warning: skip mmdet3d.core.voxel import: {exc}")
