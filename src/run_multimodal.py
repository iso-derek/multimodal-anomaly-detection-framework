from __future__ import annotations

from pathlib import Path
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray

from src.anomaly_router import detect_anomaly
from src.fusion import fuse_weighted_average


def main():
    # ----------------------------
    # Choose tabular mode
    # ----------------------------
    # Options: "numeric" or "mixed"
    tabular_mode = "numeric"

    # ----------------------------
    # Example inputs
    # ----------------------------

    if tabular_mode == "mixed":
        tabular_data = [
            {"age": 21, "department": "CS", "score": 70},
            {"age": 22, "department": "Math", "score": 72},
            {"age": 20, "department": "CS", "score": 68},
            {"age": 50, "department": "Unknown", "score": 5},
        ]
    else:
        # Numeric tabular example
        tabular_data = [1, 1, 1, 100, 1, 1]

    # Time-series example
    timeseries_data = [0, 0, 0.1, 0.2, 3.5, 0.1, 0.0]

    # Image example
    image_path = Path("images/normal_arm.jpg")  # change if needed
    img = imread(str(image_path))
    if img.ndim == 3:
        img = rgb2gray(img)
    img = img.astype(np.float32)
    image_data = img

    # Video example: use ONE clip folder for fusion
    video_clip_path = Path(
        "data/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001"
    )

    # ----------------------------
    # Run modalities via router
    # ----------------------------

    tab_result = detect_anomaly(
        data=tabular_data,
        data_type="tabular",
        tabular_mode=tabular_mode,
        threshold=0.65,
    )

    ts_result = detect_anomaly(
        data=timeseries_data,
        data_type="timeseries",
        threshold=0.65,
        timeseries_window=3,
    )

    img_result = detect_anomaly(
        data=image_data,
        data_type="image",
        threshold=0.65,
    )

    vid_result = detect_anomaly(
        data=video_clip_path,
        data_type="video",
        threshold=0.65,
        video_dataset_mode=False,
    )

    # ----------------------------
    # Collect fusion-ready outputs
    # ----------------------------

    results = {
        "tabular": tab_result,
        "timeseries": ts_result,
        "image": img_result,
        "video": vid_result,
    }

    # ----------------------------
    # Fusion weights
    # ----------------------------

    weights = {
        "video": 2.0,
        "tabular": 1.5,
        "timeseries": 1.0,
        "image": 0.8,
    }

    # ----------------------------
    # Fuse
    # ----------------------------

    fused = fuse_weighted_average(results, weights=weights)

    # ----------------------------
    # Print results
    # ----------------------------

    print("\n=== Multimodal Fusion Result ===")
    print("Tabular mode:", tabular_mode)
    print("Final score:", fused["final_score"])
    print("Final label:", "ANOMALY" if fused["final_label"] == 1 else "NORMAL")
    print("Reason:", fused.get("reason", ""))

    print("\nBy modality:")
    for modality, info in fused["by_modality"].items():
        print(
            "-",
            modality,
            "score=",
            round(info["score"], 4),
            "label=",
            info["label"],
            "method=",
            info.get("method", ""),
            "meta=",
            info.get("meta", {}),
        )


if __name__ == "__main__":
    main()