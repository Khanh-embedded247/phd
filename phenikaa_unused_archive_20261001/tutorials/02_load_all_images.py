"""
Tutorial 02: Load images from all cameras for a single timestamp and display in a 5x2 grid.
This script shows how to compose a multi-camera view.

"""

import cv2
import numpy as np
from pathlib import Path

from phenikaa_paths import CAMERAS, SEQUENCE_FOLDER, TIMESTAMP

# Output dimensions
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 720
GRID_COLS = 5
GRID_ROWS = 2
CELL_WIDTH = OUTPUT_WIDTH // GRID_COLS  # 384
CELL_HEIGHT = OUTPUT_HEIGHT // GRID_ROWS  # 360

print(f"Grid: {GRID_ROWS}x{GRID_COLS} → each cell: {CELL_WIDTH}x{CELL_HEIGHT} px")
print("Loading images for timestamp:", TIMESTAMP)

# Prepare canvas
canvas = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)

loaded_count = 0
images = []

for camera in CAMERAS:
    image_path = SEQUENCE_FOLDER / "Image" / camera / f"{TIMESTAMP}.jpg"
    if not image_path.exists():
        print(f"[{camera}] MISSING")
        images.append(None)
        continue
    
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[{camera}] FAILED to load")
        images.append(None)
        continue
    
    resized = cv2.resize(img, (CELL_WIDTH, CELL_HEIGHT))
    images.append(resized)
    loaded_count += 1
    print(f"[{camera}] Loaded → resized to {CELL_WIDTH}x{CELL_HEIGHT}")

# Text settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_thickness = 1
text_color = (255, 255, 255)
padding_left = 10
padding_top = 35

# Place images and labels
for idx, img in enumerate(images):
    row = idx // GRID_COLS
    col = idx % GRID_COLS
    y_start = row * CELL_HEIGHT
    x_start = col * CELL_WIDTH
    
    if img is not None:
        canvas[y_start:y_start + CELL_HEIGHT, x_start:x_start + CELL_WIDTH] = img
        cv2.putText(canvas, CAMERAS[idx], (x_start + padding_left, y_start + padding_top),
                    font, font_scale, text_color, font_thickness)
    else:
        text = f"{CAMERAS[idx]} MISSING"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x_start + (CELL_WIDTH - text_size[0]) // 2
        text_y = y_start + (CELL_HEIGHT + text_size[1]) // 2
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

# Display
window_name = f"All Cameras - {TIMESTAMP} (5x2 @ 1920x720)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow(window_name, canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"\nDisplayed {loaded_count}/{len(CAMERAS)} images.")