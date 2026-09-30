# Eval override for single 8GB GPU (RTX 4060) on nuScenes mini val.
_base_ = './maptrv2_nusc_r50_24ep.py'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)
