# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.pipelines import Compose
try:
    from .dbsampler import DataBaseSampler
except Exception as exc:
    DataBaseSampler = None
    print(f"Warning: skip mmdet3d.datasets.pipelines.dbsampler import: {exc}")
from .formating import Collect3D, DefaultFormatBundle, DefaultFormatBundle3D
from .loading import (LoadAnnotations3D, LoadImageFromFileMono3D,
                      LoadMultiViewImageFromFiles, LoadPointsFromFile,
                      LoadPointsFromMultiSweeps, NormalizePointsColor,
                      PointSegClassMapping)
from .test_time_aug import MultiScaleFlipAug3D
try:
    from .transforms_3d import (BackgroundPointsFilter, GlobalAlignment,
                                GlobalRotScaleTrans, IndoorPatchPointSample,
                                IndoorPointSample, ObjectNameFilter, ObjectNoise,
                                ObjectRangeFilter, ObjectSample, PointSample,
                                PointShuffle, PointsRangeFilter,
                                RandomDropPointsColor, RandomFlip3D,
                                RandomJitterPoints, VoxelBasedPointSampler)
except Exception as exc:
    print(f"Warning: skip mmdet3d.datasets.pipelines.transforms_3d import: {exc}")
    BackgroundPointsFilter = GlobalAlignment = GlobalRotScaleTrans = None
    IndoorPatchPointSample = IndoorPointSample = ObjectNameFilter = None
    ObjectNoise = ObjectRangeFilter = ObjectSample = PointSample = None
    PointShuffle = PointsRangeFilter = RandomDropPointsColor = None
    RandomFlip3D = RandomJitterPoints = VoxelBasedPointSampler = None

__all__ = [
    'ObjectSample', 'RandomFlip3D', 'ObjectNoise', 'GlobalRotScaleTrans',
    'PointShuffle', 'ObjectRangeFilter', 'PointsRangeFilter', 'Collect3D',
    'Compose', 'LoadMultiViewImageFromFiles', 'LoadPointsFromFile',
    'DefaultFormatBundle', 'DefaultFormatBundle3D', 'DataBaseSampler',
    'NormalizePointsColor', 'LoadAnnotations3D', 'IndoorPointSample',
    'PointSample', 'PointSegClassMapping', 'MultiScaleFlipAug3D',
    'LoadPointsFromMultiSweeps', 'BackgroundPointsFilter',
    'VoxelBasedPointSampler', 'GlobalAlignment', 'IndoorPatchPointSample',
    'LoadImageFromFileMono3D', 'ObjectNameFilter', 'RandomDropPointsColor',
    'RandomJitterPoints'
]
