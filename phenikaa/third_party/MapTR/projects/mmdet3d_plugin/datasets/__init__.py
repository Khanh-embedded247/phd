from .nuscenes_dataset import CustomNuScenesDataset
from .builder import custom_build_dataset

from .nuscenes_map_dataset import CustomNuScenesLocalMapDataset
from .nuscenes_offlinemap_dataset import CustomNuScenesOfflineLocalMapDataset

# AV2 datasets need full av2 deps; import lazily so nuScenes-only runs work.
try:
    from .av2_map_dataset import CustomAV2LocalMapDataset
    from .av2_offlinemap_dataset import CustomAV2OfflineLocalMapDataset
except ImportError:
    CustomAV2LocalMapDataset = None
    CustomAV2OfflineLocalMapDataset = None

__all__ = [
    'CustomNuScenesDataset', 'CustomNuScenesLocalMapDataset'
]
