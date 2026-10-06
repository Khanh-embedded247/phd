"""
Tutorial 01: Load and display a single image from a specific camera and timestamp.

"""

import cv2
from pathlib import Path

from phenikaa_paths import SEQUENCE_FOLDER, TIMESTAMP

CAMERA = "CAM_P_F"  # Can be changed to any valid camera name

# Construct image path
image_path = SEQUENCE_FOLDER / "Image" / CAMERA / f"{TIMESTAMP}.jpg"

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

# Load and display the image
img = cv2.imread(str(image_path))
print(f"Loaded image: {img.shape} (H, W, C)")

height, width = img.shape[:2]
display_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

window_name = f"{CAMERA} - {TIMESTAMP}"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name, display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()