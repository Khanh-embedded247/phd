# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.builder import build_dataloader
from .builder import DATASETS, build_dataset
from .custom_3d import Custom3DDataset
from .nuscenes_dataset import NuScenesDataset
try:
    from .custom_3d_seg import Custom3DSegDataset
    from .kitti_dataset import KittiDataset
    from .kitti_mono_dataset import KittiMonoDataset
    from .lyft_dataset import LyftDataset
    from .nuscenes_mono_dataset import NuScenesMonoDataset
except Exception as exc:
    print(f"Warning: skip optional mmdet3d datasets import: {exc}")
    Custom3DSegDataset = KittiDataset = KittiMonoDataset = None
    LyftDataset = NuScenesMonoDataset = None
# yapf: disable
from .pipelines import (BackgroundPointsFilter, GlobalAlignment,
                        GlobalRotScaleTrans, IndoorPatchPointSample,
                        IndoorPointSample, LoadAnnotations3D,
                        LoadPointsFromFile, LoadPointsFromMultiSweeps,
                        NormalizePointsColor, ObjectNameFilter, ObjectNoise,
                        ObjectRangeFilter, ObjectSample, PointSample,
                        PointShuffle, PointsRangeFilter, RandomDropPointsColor,
                        RandomFlip3D, RandomJitterPoints,
                        VoxelBasedPointSampler)
# yapf: enable
try:
    from .s3dis_dataset import S3DISDataset, S3DISSegDataset
    from .scannet_dataset import ScanNetDataset, ScanNetSegDataset
    from .semantickitti_dataset import SemanticKITTIDataset
    from .sunrgbd_dataset import SUNRGBDDataset
except Exception as exc:
    print(f"Warning: skip optional indoor/seg datasets import: {exc}")
    S3DISDataset = S3DISSegDataset = ScanNetDataset = None
    ScanNetSegDataset = SemanticKITTIDataset = SUNRGBDDataset = None
from .utils import get_loading_pipeline
try:
    from .waymo_dataset import WaymoDataset
except Exception as exc:
    print(f"Warning: skip WaymoDataset import: {exc}")
    WaymoDataset = None

__all__ = [
    'KittiDataset', 'KittiMonoDataset', 'build_dataloader', 'DATASETS',
    'build_dataset', 'NuScenesDataset', 'NuScenesMonoDataset', 'LyftDataset',
    'ObjectSample', 'RandomFlip3D', 'ObjectNoise', 'GlobalRotScaleTrans',
    'PointShuffle', 'ObjectRangeFilter', 'PointsRangeFilter',
    'LoadPointsFromFile', 'S3DISSegDataset', 'S3DISDataset',
    'NormalizePointsColor', 'IndoorPatchPointSample', 'IndoorPointSample',
    'PointSample', 'LoadAnnotations3D', 'GlobalAlignment', 'SUNRGBDDataset',
    'ScanNetDataset', 'ScanNetSegDataset', 'SemanticKITTIDataset',
    'Custom3DDataset', 'Custom3DSegDataset', 'LoadPointsFromMultiSweeps',
    'WaymoDataset', 'BackgroundPointsFilter', 'VoxelBasedPointSampler',
    'get_loading_pipeline', 'RandomDropPointsColor', 'RandomJitterPoints',
    'ObjectNameFilter'
]
