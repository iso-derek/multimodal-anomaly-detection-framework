from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Segment:
    start_frame: int
    end_frame: int
    length: int


# ----------------------------
# Load scores
# ----------------------------

def load_scores(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "frame_index" not in df.columns or "score" not in df.columns:
        raise ValueError("CSV must contain columns: frame_index, score")
    return df


# ----------------------------
# Smoothing
# ----------------------------

def smooth(scores: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Simple moving average smoothing.
    window=1 disables smoothing.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if window <= 1:
        return scores

    kernel = np.ones(window, dtype=np.float32) / window
    pad = window // 2
    padded = np.pad(scores, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ----------------------------
# Percentile threshold
# ----------------------------

def percentile_threshold(scores: np.ndarray, top_percent: float = 5.0) -> float:
    """
    Mark top 'top_percent' highest scores as anomalies.
    Example: top_percent=5 => threshold at 95th percentile.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if not (0.0 < top_percent < 100.0):
        raise ValueError("top_percent must be between 0 and 100.")
    q = 100.0 - top_percent
    return float(np.percentile(scores, q))


# ----------------------------
# Segment extraction
# ----------------------------

def segments_from_mask(
    frame_index: np.ndarray,
    mask: np.ndarray,
    min_len: int = 3,
    gap_merge: int = 2,
) -> List[Segment]:
    """
    Convert boolean anomaly mask to contiguous anomaly segments.
    """
    idx = np.asarray(frame_index, dtype=int)
    m = np.asarray(mask, dtype=bool)

    segments: List[Segment] = []
    start: Optional[int] = None

    for i in range(len(m)):
        if m[i] and start is None:
            start = idx[i]
        elif not m[i] and start is not None:
            end = idx[i - 1]
            length = end - start + 1
            if length >= min_len:
                segments.append(Segment(start, end, length))
            start = None

    if start is not None:
        end = idx[-1]
        length = end - start + 1
        if length >= min_len:
            segments.append(Segment(start, end, length))

    # Merge close segments
    if not segments:
        return []

    merged: List[Segment] = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg.start_frame <= last.end_frame + gap_merge:
            merged[-1] = Segment(
                last.start_frame,
                max(last.end_frame, seg.end_frame),
                max(last.end_frame, seg.end_frame) - last.start_frame + 1,
            )
        else:
            merged.append(seg)

    return merged


# ----------------------------
# Main analysis function
# ----------------------------

def analyze_clip_percentile(
    csv_path: str | Path,
    top_percent: float = 5.0,
    smooth_window: int = 5,
    min_len: int = 3,
    gap_merge: int = 2,
) -> dict:
    """
    Full percentile-based analysis for one video clip.
    """
    df = load_scores(csv_path)

    raw_scores = df["score"].to_numpy(dtype=np.float32)
    smoothed = smooth(raw_scores, window=smooth_window)

    tau = percentile_threshold(smoothed, top_percent=top_percent)
    mask = smoothed > tau

    segments = segments_from_mask(
        frame_index=df["frame_index"].to_numpy(),
        mask=mask,
        min_len=min_len,
        gap_merge=gap_merge,
    )

    return {
        "csv": str(csv_path),
        "top_percent": top_percent,
        "threshold": tau,
        "n_frames": len(smoothed),
        "n_anomaly_frames": int(mask.sum()),
        "segments": segments,
    }


# ----------------------------
# Test run
# ----------------------------

if __name__ == "__main__":
    import sys

    # Default CSV if none provided
    csv = r"outputs\video_ucsd\UCSDped1_Test_Test001_scores.csv"
    if len(sys.argv) >= 2:
        csv = sys.argv[1]

    summary = analyze_clip_percentile(
        csv_path=csv,
        top_percent=5.0,
        smooth_window=5,
        min_len=3,
        gap_merge=2,
    )

    print("CSV:", summary["csv"])
    print("Top percent:", summary["top_percent"])
    print("Threshold:", summary["threshold"])
    print("Anomaly frames:", summary["n_anomaly_frames"])
    print("Segments:")
    for seg in summary["segments"]:
        print(f"  frames {seg.start_frame} → {seg.end_frame} (len={seg.length})")
