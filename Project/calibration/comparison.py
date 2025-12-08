import cv2
import numpy as np
import os

# If your images are stereo pairs (left|right concatenated)
root = os.path.dirname(os.path.abspath(__file__))
rectified_path = 'rectified/image_02/data/0000000000.png'
raw_path = os.path.join(root, '../34759_final_project_raw/calib/image_02/data/0000000000.png')

# Check if files exist
if not os.path.exists(rectified_path):
    print(f"Error: Rectified image not found at {rectified_path}")
    print("Run calibration.py first to generate rectified images")
    exit(1)

if not os.path.exists(raw_path):
    print(f"Error: Raw image not found at {raw_path}")
    exit(1)

rectified = cv2.imread(rectified_path)
raw = cv2.imread(raw_path)

line_spacing = 80  # pixels between lines
color = (0, 255, 0)  # Green
thickness = 2

# Draw lines across BOTH left and right images
for y in range(0, raw.shape[0], line_spacing):
    cv2.line(raw, (0, y), (raw.shape[1], y), color, thickness)
    cv2.line(rectified, (0, y), (rectified.shape[1], y), color, thickness)

# Optional: Add a vertical divider between left/right
mid = raw.shape[1] // 2
cv2.line(raw, (mid, 0), (mid, raw.shape[0]), (255, 0, 0), 2)
cv2.line(rectified, (mid, 0), (mid, rectified.shape[0]), (255, 0, 0), 2)

cv2.imwrite('0000000000-raw-annotated.png', raw)
cv2.imwrite('0000000000-rectified-annotated.png', rectified)
