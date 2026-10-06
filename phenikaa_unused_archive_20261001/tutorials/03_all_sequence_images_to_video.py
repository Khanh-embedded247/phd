"""
Tutorial 03: Create a video from all sequence images in a 5x2 multi-camera grid.
This script processes all timestamps to generate a video.

"""

import cv2
import numpy as np
from pathlib import Path
from phenikaa_paths import CAMERAS, RESULT_ROOT, SEQUENCE_FOLDER, TIMESTAMPS

IMAGE_ROOT = SEQUENCE_FOLDER / "Image"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

sequence_name = SEQUENCE_FOLDER.name.replace(" ", "_")
OUTPUT_VIDEO = RESULT_ROOT / f"{sequence_name}_multi_camera_5x2.mp4"
FPS = 10

# Grid settings
GRID_COLS = 5
GRID_ROWS = 2
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 720
CELL_WIDTH = OUTPUT_WIDTH // GRID_COLS  # 384
CELL_HEIGHT = OUTPUT_HEIGHT // GRID_ROWS  # 360

timestamps = TIMESTAMPS
print(f"Found {len(timestamps)} frames ({timestamps[0]} → {timestamps[-1]}).")
print(f"Output: {OUTPUT_VIDEO}")

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, FPS, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

# Text settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_thickness = 1
text_color = (255, 255, 255)
padding_left = 12
padding_top = 40

frame_count = 0
for ts in timestamps:
    frame_count += 1
    print(f"Processing {frame_count}/{len(timestamps)} → {ts}", end="\r")

    canvas = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)  # Dark background

    for idx, camera in enumerate(CAMERAS):
        img_path = IMAGE_ROOT / camera / f"{ts}.jpg"
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        x = col * CELL_WIDTH
        y = row * CELL_HEIGHT

        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                resized = cv2.resize(img, (CELL_WIDTH, CELL_HEIGHT), interpolation=cv2.INTER_AREA)
                canvas[y:y + CELL_HEIGHT, x:x + CELL_WIDTH] = resized
                cv2.putText(canvas, camera, (x + padding_left, y + padding_top),
                            font, font_scale, text_color, font_thickness)
        else:
            cv2.putText(canvas, camera, (x + padding_left, y + padding_top),
                        font, font_scale, (0, 0, 255), font_thickness)
            cv2.putText(canvas, "MISSING", (x + padding_left, y + padding_top + 40),
                        font, font_scale, (0, 0, 255), font_thickness)

    # Timestamp
    cv2.putText(canvas, ts, (OUTPUT_WIDTH - 500, OUTPUT_HEIGHT - 30),
                font, 0.8, (255, 255, 255), 1)

    video_writer.write(canvas)

video_writer.release()
print("\nVideo created successfully!")