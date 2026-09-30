# Copyright (c) OpenMMLab. All rights reserved.
from .base import Base3DDetector
from .mvx_two_stage import MVXTwoStageDetector

# Phenikaa/MapTR chi can MVXTwoStageDetector lam lop nen.
# Cac detector 3D khac co the keo NMS/numba extension cu, nen de optional.
try:
    from .centerpoint import CenterPoint
except Exception as exc:
    print(f"Warning: skip CenterPoint import: {exc}")
    CenterPoint = None
try:
    from .dynamic_voxelnet import DynamicVoxelNet
except Exception as exc:
    print(f"Warning: skip DynamicVoxelNet import: {exc}")
    DynamicVoxelNet = None
try:
    from .fcos_mono3d import FCOSMono3D
except Exception as exc:
    print(f"Warning: skip FCOSMono3D import: {exc}")
    FCOSMono3D = None
try:
    from .groupfree3dnet import GroupFree3DNet
except Exception as exc:
    print(f"Warning: skip GroupFree3DNet import: {exc}")
    GroupFree3DNet = None
try:
    from .h3dnet import H3DNet
except Exception as exc:
    print(f"Warning: skip H3DNet import: {exc}")
    H3DNet = None
try:
    from .imvotenet import ImVoteNet
except Exception as exc:
    print(f"Warning: skip ImVoteNet import: {exc}")
    ImVoteNet = None
try:
    from .imvoxelnet import ImVoxelNet
except Exception as exc:
    print(f"Warning: skip ImVoxelNet import: {exc}")
    ImVoxelNet = None
try:
    from .mvx_faster_rcnn import DynamicMVXFasterRCNN, MVXFasterRCNN
except Exception as exc:
    print(f"Warning: skip MVXFasterRCNN import: {exc}")
    DynamicMVXFasterRCNN = None
    MVXFasterRCNN = None
try:
    from .parta2 import PartA2
except Exception as exc:
    print(f"Warning: skip PartA2 import: {exc}")
    PartA2 = None
try:
    from .single_stage_mono3d import SingleStageMono3DDetector
except Exception as exc:
    print(f"Warning: skip SingleStageMono3DDetector import: {exc}")
    SingleStageMono3DDetector = None
try:
    from .ssd3dnet import SSD3DNet
except Exception as exc:
    print(f"Warning: skip SSD3DNet import: {exc}")
    SSD3DNet = None
try:
    from .votenet import VoteNet
except Exception as exc:
    print(f"Warning: skip VoteNet import: {exc}")
    VoteNet = None
try:
    from .voxelnet import VoxelNet
except Exception as exc:
    print(f"Warning: skip VoxelNet import: {exc}")
    VoxelNet = None

__all__ = [
    'Base3DDetector', 'VoxelNet', 'DynamicVoxelNet', 'MVXTwoStageDetector',
    'DynamicMVXFasterRCNN', 'MVXFasterRCNN', 'PartA2', 'VoteNet', 'H3DNet',
    'CenterPoint', 'SSD3DNet', 'ImVoteNet', 'SingleStageMono3DDetector',
    'FCOSMono3D', 'ImVoxelNet', 'GroupFree3DNet'
]
