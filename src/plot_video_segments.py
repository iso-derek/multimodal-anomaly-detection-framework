import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.video_postprocess import smooth, percentile_threshold, segments_from_mask


def plot_segments(
    csv_path: str,
    top_percent: float = 5.0,
    smooth_window: int = 5,
    min_len: int = 3,
    gap_merge: int = 2,
):
    csv_path = str(csv_path)
    df = pd.read_csv(csv_path)

    frame_index = df["frame_index"].to_numpy(dtype=int)
    scores = df["score"].to_numpy(dtype=np.float32)

    s = smooth(scores, window=smooth_window)
    tau = percentile_threshold(s, top_percent=top_percent)
    mask = s > tau

    segments = segments_from_mask(
        frame_index=frame_index,
        mask=mask,
        min_len=min_len,
        gap_merge=gap_merge,
    )

    plt.figure(figsize=(12, 4))
    plt.plot(frame_index, s, label="score (smoothed)")
    plt.axhline(tau, linestyle="--", label=f"threshold (top {top_percent}%)")

    # Shade anomaly segments
    for seg in segments:
        plt.axvspan(seg.start_frame, seg.end_frame, alpha=0.2)

    clip_name = Path(csv_path).stem
    plt.title(
        f"{clip_name} | top_percent={top_percent} | w={smooth_window} | "
        f"segments={len(segments)}"
    )
    plt.xlabel("Frame index")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("CSV:", csv_path)
    print("Threshold:", float(tau))
    print("Segments:")
    for seg in segments:
        print(f"  frames {seg.start_frame} → {seg.end_frame} (len={seg.length})")


if __name__ == "__main__":
    # Default file if none provided
    csv = r"outputs\video_ucsd\UCSDped1_Test_Test001_scores.csv"
    if len(sys.argv) >= 2:
        csv = sys.argv[1]

    plot_segments(
        csv_path=csv,
        top_percent=5.0,
        smooth_window=5,
        min_len=3,
        gap_merge=2,
    )
