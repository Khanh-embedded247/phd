"""
Tutorial 05: Load and print information from a LiDAR point cloud (.laz file).
This script demonstrates how to read point cloud data and extract key attributes.
Note: The LiDAR timestamp is the save as last time of scan range

"""

import laspy
from pathlib import Path

from phenikaa_paths import SEQUENCE_FOLDER, TIMESTAMP
CHECK_RANGE = 20

# Construct path
laz_path = SEQUENCE_FOLDER / "Lidar" / f"{TIMESTAMP}.laz"

if not laz_path.exists():
    print(f"Error: File not found: {laz_path}")
else:
    print(f"\nProcessing file: {laz_path.name}")
    las = laspy.read(str(laz_path))
    
    x = las.x
    y = las.y
    z = las.z
    intensity = las.intensity
    
    # Ring or return_number
    if "ring" in las.point_format.dimension_names:
        ring = las.ring
    else:
        ring = las.return_number
    
    # Timestamp if available
    if "point_timestamp" in las.point_format.dimension_names:
        timestamp = las.point_timestamp
    else:
        timestamp = [0.0] * len(x)
    
    num_points = min(CHECK_RANGE, len(x))
    print(f"Total points: {len(x)}")
    print(f"Showing first {num_points} points:")
    print("Index | X          | Y          | Z        | Intensity | Ring | Timestamp")
    print("-" * 75)
    
    for i in range(num_points):
        print(f"{i:5d} | {x[i]:10.4f} | {y[i]:10.4f} | {z[i]:8.4f} | {intensity[i]:9d} | {ring[i]:4d} | {timestamp[i]:.9f}")