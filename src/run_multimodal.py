from __future__ import annotations

from pathlib import Path
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray

from src.anomaly_router import detect_anomaly
from src.fusion import (
    summarize_tabular,
    summarize_timeseries,
    summarize_image,
    summarize_video,
    fuse_weighted_average
)


def main():
    # -----------------------
    # Example inputs
    # -----------------------

    # Tabular example
    tabular_data = [1, 1, 1, 100, 1, 1]

    # Timeseries example
    timeseries_data = [0, 0, 0.1, 0.2, 3.5, 0.1, 0.0]

    # -----------------------
    # Image example
    # -----------------------
    image_path = Path("images/normal_arm.jpg")  # change if needed

    img = imread(str(image_path))
    if img.ndim == 3:  # RGB -> grayscale
        img = rgb2gray(img)

    img = img.astype(np.float32)
    image_data = img

    # -----------------------
    # Video example
    # -----------------------
    video_data = {
        "root": "data/ucsd/UCSD_Anomaly_Dataset.v1p2",
        "dataset": "UCSDped1",
        "split": "Test"
    }

    # -----------------------
    # Run modalities via router
    # -----------------------
    tab_result = detect_anomaly(tabular_data, "tabular")
    ts_result = detect_anomaly(timeseries_data, "timeseries")
    img_result = detect_anomaly(image_data, "image")
    vid_result = detect_anomaly(video_data, "video")

    # -----------------------
    # Summarize into scalar scores
    # -----------------------
    summaries = [
        summarize_tabular(tab_result),
        summarize_timeseries(ts_result),
        summarize_image(img_result),
        summarize_video(vid_result, score_mode="p95"),
    ]

    # -----------------------
    # Fuse (UPDATED WEIGHTS)
    # -----------------------
    weights = {
        "video": 2.0,
        "tabular": 1.5,
        "timeseries": 1.0,
        "image": 0.8,
    }

    fused = fuse_weighted_average(summaries, weights=weights)

    print("\n=== Multimodal Fusion Result ===")
    print("Final score:", fused["final_score"])
    print("Final label:", "ANOMALY" if fused["final_label"] == -1 else "NORMAL")
    print("Reason:", fused.get("reason", ""))
    print("\nBy modality:")
    for m in fused["by_modality"]:
        print(
            "-",
            m["modality"],
            "score=",
            round(m["score"], 4),
            "label=",
            m["label"],
            "meta=",
            m["meta"],
        )


if __name__ == "__main__":
    main()
