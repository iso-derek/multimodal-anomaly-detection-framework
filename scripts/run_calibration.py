from pathlib import Path
import json
import numpy as np

from src.detection_tabular import detect_tabular_for_fusion
from src.detection_timeseries import detect_timeseries_for_fusion
from src.detection_image import detect_image_for_fusion
from src.detection_video import detect_video_clip_for_fusion

# Output folder
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)


def save_calibration(name, scores):
    scores = np.array(scores, dtype=float)

    if scores.size == 0:
        print(f"No scores collected for {name}, skipping save")
        return

    calib = {
        "p10": float(np.percentile(scores, 10)),
        "p90": float(np.percentile(scores, 90)),
        "median": float(np.median(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "n_samples": int(len(scores)),
    }

    path = OUT_DIR / f"{name}_calibration.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)

    print(f"Saved {name} calibration -> {path}")


# TABULAR DATA
def calibrate_tabular():
    scores = []

    for _ in range(30):
        data = np.random.normal(0, 1, (20, 3))
        data[0, 0] = np.random.uniform(5, 10)

        result = detect_tabular_for_fusion(data)
        scores.append(result["score_raw"])

    save_calibration("tabular", scores)


# TIMESERIES
def calibrate_timeseries():
    scores = []

    for _ in range(30):
        ts = np.random.normal(0, 1, 50)
        ts[25] += np.random.uniform(5, 10)

        result = detect_timeseries_for_fusion(ts, window=3)
        scores.append(result["score_raw"])

    save_calibration("timeseries", scores)


# IMAGE
def calibrate_image():
    scores = []

    for _ in range(30):
        img = np.random.rand(128, 128)

        result = detect_image_for_fusion(
            model=None,
            image=img,
            threshold=0.65,
            calib=None,
        )
        scores.append(result["score_raw"])

    save_calibration("image", scores)


# VIDEO
def calibrate_video():
    scores = []

    base = Path("data/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test")

    if not base.exists():
        print("UCSD dataset not found, skipping video calibration")
        return

    clip_dirs = [
        p for p in base.iterdir()
        if p.is_dir() and p.name.lower().startswith("test")
    ]

    if not clip_dirs:
        print("No valid UCSD clip folders found, skipping video calibration")
        return

    for clip in clip_dirs[:10]:
        try:
            result = detect_video_clip_for_fusion(clip)
            scores.append(result["score_raw"])
            print(f"Processed video clip: {clip.name}")
        except Exception as e:
            print(f"Skipping {clip.name}: {e}")

    save_calibration("video", scores)


if __name__ == "__main__":
    print("Running calibration...")

    calibrate_tabular()
    calibrate_timeseries()
    calibrate_image()
    calibrate_video()

    print("Calibration complete.")