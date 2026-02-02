from __future__ import annotations

import json
from datetime import datetime
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
    fuse_weighted_average,
)


def load_gray_image(path: str) -> np.ndarray:
    img = imread(path)
    if img.ndim == 3:
        img = rgb2gray(img)
    return img.astype(np.float32)


def run_case(
    name: str,
    tabular_data,
    ts_data,
    image_path: str,
    video_cfg: dict,
    weights: dict,
) -> dict:
    # Run modalities
    tab_result = detect_anomaly(tabular_data, "tabular")
    ts_result = detect_anomaly(ts_data, "timeseries")

    img = load_gray_image(image_path)
    img_result = detect_anomaly(img, "image")

    vid_result = detect_anomaly(video_cfg, "video")

    summaries = [
        summarize_tabular(tab_result),
        summarize_timeseries(ts_result),
        summarize_image(img_result),
        summarize_video(vid_result, score_mode="p95"),
    ]

    fused = fuse_weighted_average(summaries, weights=weights)

    return {
        "case": name,
        "final_score": fused["final_score"],
        "final_label": fused["final_label"],
        "reason": fused.get("reason", ""),
        "by_modality": fused["by_modality"],
    }


def main():
    out_path = Path("outputs/fusion_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fusion weights (same as run_multimodal.py)
    weights = {
        "video": 2.0,
        "tabular": 1.5,
        "timeseries": 1.0,
        "image": 0.8,
    }

    # Image (must exist)
    image_path = "images/normal_arm.jpg"  # change if needed

    # UCSD configs
    video_ped1_test = {
        "root": "data/ucsd/UCSD_Anomaly_Dataset.v1p2",
        "dataset": "UCSDped1",
        "split": "Test",
    }

    video_ped1_train = {
        "root": "data/ucsd/UCSD_Anomaly_Dataset.v1p2",
        "dataset": "UCSDped1",
        "split": "Train",
    }

    cases = []

    # Case 1: clear tabular anomaly + anomalous video
    cases.append(
        run_case(
            name="tabular_anomaly_with_anomalous_video",
            tabular_data=[1, 1, 1, 100, 1, 1],
            ts_data=[0, 0, 0.1, 0.2, 0.1, 0.0, 0.1],
            image_path=image_path,
            video_cfg=video_ped1_test,
            weights=weights,
        )
    )

    # Case 2: normal tabular + anomalous video
    cases.append(
        run_case(
            name="all_normal_but_anomalous_video",
            tabular_data=[1, 1, 1, 1, 1, 1],
            ts_data=[0, 0, 0.1, 0.2, 0.1, 0.0, 0.1],
            image_path=image_path,
            video_cfg=video_ped1_test,
            weights=weights,
        )
    )

    # Case 3: all normal inputs + NORMAL video (TRAIN split)
    cases.append(
        run_case(
            name="all_normal_with_normal_video",
            tabular_data=[1, 1, 1, 1, 1, 1],
            ts_data=[0, 0, 0.1, 0.2, 0.1, 0.0, 0.1],
            image_path=image_path,
            video_cfg=video_ped1_train,
            weights=weights,
        )
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "weights": weights,
        "cases": cases,
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote fusion evaluation to {out_path}")


if __name__ == "__main__":
    main()
