# Copyright (c) OpenMMLab. All rights reserved.
from .builder import (FUSION_LAYERS, MIDDLE_ENCODERS, VOXEL_ENCODERS,
                      build_backbone, build_detector, build_fusion_layer,
                      build_head, build_loss, build_middle_encoder,
                      build_model, build_neck, build_roi_extractor,
                      build_shared_head, build_voxel_encoder)


def _optional_import(name, import_fn):
    """Import nhom module neu moi truong ho tro."""
    try:
        import_fn()
    except Exception as exc:
        print(f"Warning: skip mmdet3d.models.{name} import: {exc}")


_optional_import("backbones", lambda: exec("from .backbones import *", globals()))
_optional_import("decode_heads", lambda: exec("from .decode_heads import *", globals()))
_optional_import("dense_heads", lambda: exec("from .dense_heads import *", globals()))
_optional_import("detectors", lambda: exec("from .detectors import *", globals()))
_optional_import("fusion_layers", lambda: exec("from .fusion_layers import *", globals()))
_optional_import("losses", lambda: exec("from .losses import *", globals()))
_optional_import("middle_encoders", lambda: exec("from .middle_encoders import *", globals()))
_optional_import("model_utils", lambda: exec("from .model_utils import *", globals()))
_optional_import("necks", lambda: exec("from .necks import *", globals()))
_optional_import("roi_heads", lambda: exec("from .roi_heads import *", globals()))
_optional_import("segmentors", lambda: exec("from .segmentors import *", globals()))
_optional_import("voxel_encoders", lambda: exec("from .voxel_encoders import *", globals()))

__all__ = [
    'VOXEL_ENCODERS', 'MIDDLE_ENCODERS', 'FUSION_LAYERS', 'build_backbone',
    'build_neck', 'build_roi_extractor', 'build_shared_head', 'build_head',
    'build_loss', 'build_detector', 'build_fusion_layer', 'build_model',
    'build_middle_encoder', 'build_voxel_encoder'
]
