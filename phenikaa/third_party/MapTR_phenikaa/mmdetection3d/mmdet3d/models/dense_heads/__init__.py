# Copyright (c) OpenMMLab. All rights reserved.
#
# Phenikaa/MapTR camera-vector experiments chi can registry/model builder.
# Mot so dense head 3D cu keo theo numba/NMS extension va co the loi tren
# moi truong hien tai. Boc optional de MapTR import/build duoc; neu can train
# cac detector 3D nay thi can sua moi truong extension rieng.
try:
    from .anchor3d_head import Anchor3DHead
except Exception as exc:
    print(f"Warning: skip Anchor3DHead import: {exc}")
    Anchor3DHead = None
try:
    from .anchor_free_mono3d_head import AnchorFreeMono3DHead
except Exception as exc:
    print(f"Warning: skip AnchorFreeMono3DHead import: {exc}")
    AnchorFreeMono3DHead = None
try:
    from .base_conv_bbox_head import BaseConvBboxHead
except Exception as exc:
    print(f"Warning: skip BaseConvBboxHead import: {exc}")
    BaseConvBboxHead = None
try:
    from .base_mono3d_dense_head import BaseMono3DDenseHead
except Exception as exc:
    print(f"Warning: skip BaseMono3DDenseHead import: {exc}")
    BaseMono3DDenseHead = None
try:
    from .centerpoint_head import CenterHead
except Exception as exc:
    print(f"Warning: skip CenterHead import: {exc}")
    CenterHead = None
try:
    from .fcos_mono3d_head import FCOSMono3DHead
except Exception as exc:
    print(f"Warning: skip FCOSMono3DHead import: {exc}")
    FCOSMono3DHead = None
try:
    from .free_anchor3d_head import FreeAnchor3DHead
except Exception as exc:
    print(f"Warning: skip FreeAnchor3DHead import: {exc}")
    FreeAnchor3DHead = None
try:
    from .groupfree3d_head import GroupFree3DHead
except Exception as exc:
    print(f"Warning: skip GroupFree3DHead import: {exc}")
    GroupFree3DHead = None
try:
    from .parta2_rpn_head import PartA2RPNHead
except Exception as exc:
    print(f"Warning: skip PartA2RPNHead import: {exc}")
    PartA2RPNHead = None
try:
    from .shape_aware_head import ShapeAwareHead
except Exception as exc:
    print(f"Warning: skip ShapeAwareHead import: {exc}")
    ShapeAwareHead = None
try:
    from .ssd_3d_head import SSD3DHead
except Exception as exc:
    print(f"Warning: skip SSD3DHead import: {exc}")
    SSD3DHead = None
try:
    from .vote_head import VoteHead
except Exception as exc:
    print(f"Warning: skip VoteHead import: {exc}")
    VoteHead = None

__all__ = [
    'Anchor3DHead', 'FreeAnchor3DHead', 'PartA2RPNHead', 'VoteHead',
    'SSD3DHead', 'BaseConvBboxHead', 'CenterHead', 'ShapeAwareHead',
    'BaseMono3DDenseHead', 'AnchorFreeMono3DHead', 'FCOSMono3DHead',
    'GroupFree3DHead'
]
