"""
Tutorial 04: Load and print 3D annotations from a label file.
This script show you which information is available in the annotation files.

"""

from pathlib import Path

from phenikaa_paths import (
    SEQUENCE_FOLDER,
    TIMESTAMP,
    TIMESTAMP_FIRST,
    TIMESTAMP_LAST,
    TIMESTAMPS,
)

CHECK_RANGE = 30

print(f"Sequence: {SEQUENCE_FOLDER.name}")
print(f"Frames: {len(TIMESTAMPS)} ({TIMESTAMP_FIRST} → {TIMESTAMP_LAST})")
print(f"Using TIMESTAMP: {TIMESTAMP}")

# Load annotations
ann_path = SEQUENCE_FOLDER / "Label" / f"{TIMESTAMP}.txt"
annotations = []

if not ann_path.exists():
    raise FileNotFoundError(f"Label file not found: {ann_path}")

with open(ann_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        x, y, z, l, w, h, yaw = map(float, parts[:7])
        class_name = ' '.join(parts[7:]).title()
        annotations.append({'class': class_name, 'x': x, 'y': y, 'z': z, 'l': l, 'w': w, 'h': h, 'yaw': yaw})

print(f"Loaded {len(annotations)} valid 3D annotations")
for ann in annotations[:CHECK_RANGE]:
    print(ann)