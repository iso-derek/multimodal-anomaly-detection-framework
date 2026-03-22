from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# Fix import path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.detection_image_compare import compare_image_set
from src.detection_video_compare import compare_video_set


print("\n### IMAGE + VIDEO COMPARISON TEST ###\n")

# -------------------
# IMAGE COMPARISON
# -------------------
print("Image Comparison Test")

img1 = np.zeros((32, 32))
img2 = np.zeros((32, 32))
img3 = np.ones((32, 32))  # anomaly

img_result = compare_image_set(
    images=[img1, img2, img3],
    names=["normal_1", "normal_2", "abnormal"],
    threshold=0.5
)

print("Most abnormal image:", img_result["most_abnormal"]["name"])
print("Score:", img_result["most_abnormal"]["score_norm"])
print()

# -------------------
# VIDEO COMPARISON
# -------------------
print("Video Comparison Test")

# Replace with your real UCSD paths if needed
clips = [
    "data/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001",
    "data/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test002",
]

vid_result = compare_video_set(
    clip_paths=clips,
    names=["clip1", "clip2"],
    threshold=0.65
)

print("Most abnormal video:", vid_result["most_abnormal"]["name"])
print("Score:", vid_result["most_abnormal"]["score_norm"])

print("\n### COMPARISON TEST COMPLETE ###")