from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from skimage.io import imread

from src.detection_tabular import detect_tabular_for_fusion
from src.detection_timeseries import detect_timeseries_for_fusion
from src.detection_image import detect_image_for_fusion
from src.detection_video import detect_video_clip_for_fusion


def load_image(path: str) -> np.ndarray:
    """
    Loads image as numpy array.
    - If your image AE is RGB, do not convert to grayscale here.
    - X-ray images are typically already grayscale anyway.
    """
    img = imread(path)
    return img.astype(np.float32)


def pick_first_ucsd_clip(video_cfg: dict) -> Path:
    """
    video_cfg = {"root": "...", "dataset": "UCSDped1", "split": "Test"}
    Picks the first clip folder in that split (e.g., Test001).
    """
    base = Path(video_cfg["root"]) / video_cfg["dataset"] / video_cfg["split"]
    if not base.exists():
        raise FileNotFoundError(f"UCSD split folder not found: {base}")

    clip_dirs = sorted(
        [p for p in base.iterdir() if p.is_dir() and not p.name.lower().endswith("_gt")]
    )
    if not clip_dirs:
        raise RuntimeError(f"No UCSD clip folders found in: {base}")

    return clip_dirs[0]


from src.fusion import fuse_weighted_average


def run_case(name: str, tabular_data, ts_data, image_path: str, video_cfg: dict, weights: dict) -> dict:
    # Tabular
    tab_result = detect_tabular_for_fusion(np.asarray(tabular_data))

    # Time-series
    ts_result = detect_timeseries_for_fusion(np.asarray(ts_data), window=3)

    # Image
    img = load_image(image_path)
    img_result = detect_image_for_fusion(model=None, image=img)

    # Video
    clip_dir = pick_first_ucsd_clip(video_cfg)
    vid_result = detect_video_clip_for_fusion(clip_dir, threshold=0.90)

    results = {
        "tabular": tab_result,
        "timeseries": ts_result,
        "image": img_result,
        "video": vid_result,
    }

    fused = fuse_weighted_average(results, weights=weights)

    return {
        "case": name,
        "final_score": fused["final_score"],
        "final_label": fused["final_label"],
        "reason": fused.get("reason", ""),
        "by_modality": fused["by_modality"],
        "inputs": {
            "image_path": image_path,
            "video_clip": str(clip_dir),
        },
    }


def run_suite(image_path: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    weights = {
        "video": 2.0,
        "tabular": 1.5,
        "timeseries": 1.0,
        "image": 0.8,
    }

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

    ts_data_anom = [0.0, 0.0, 0.1, 2.0, 0.1, 0.0, 0.1]
    ts_data_normal = [0.1, 0.1, 0.11, 0.1, 0.09, 0.1, 0.1]

    cases = [
        run_case(
            name="tabular_anomaly_with_test_video_clip",
            tabular_data=[1, 1, 1, 100, 1, 1],
            ts_data=ts_data_anom,
            image_path=image_path,
            video_cfg=video_ped1_test,
            weights=weights,
        ),
        run_case(
            name="all_normal_with_train_video_clip",
            tabular_data=[1, 1, 1, 1, 1, 1],
            ts_data=ts_data_normal,
            image_path=image_path,
            video_cfg=video_ped1_train,
            weights=weights,
        ),
        run_case(
            name="normal_signals_with_test_video_clip",
            tabular_data=[1, 1, 1, 1, 1, 1],
            ts_data=ts_data_normal,
            image_path=image_path,
            video_cfg=video_ped1_test,
            weights=weights,
        ),
    ]

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "weights": weights,
        "image_path": image_path,
        "cases": cases,
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote fusion evaluation to {out_path}")


def main():
    run_suite(
        image_path="images/normal_arm.jpg",
        out_path=Path("outputs/fusion_summary_normal_arm.json"),
    )
    run_suite(
        image_path="images/abnormal_arm.jpg",
        out_path=Path("outputs/fusion_summary_abnormal_arm.json"),
    )


if __name__ == "__main__":
    main()