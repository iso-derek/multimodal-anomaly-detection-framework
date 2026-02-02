"""
Video (UCSD) anomaly detection — baseline.
Works on UCSDped1 / UCSDped2 frame folders (tif/png).
Produces an anomaly score per frame using simple frame-difference motion energy.

This is intentionally lightweight and defensible for an FYP:
- fast to run
- no GPU required
- easy to explain + extend later (optical flow / autoencoders)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional deps (only needed for reading images)
import cv2
import tifffile as tiff


# ----------------------------
# Utilities
# ----------------------------

def _read_gray(img_path: Path) -> np.ndarray:
    """Read an image into a float32 grayscale array. Robust to broken TIFFs."""
    suf = img_path.suffix.lower()

    # 1) Try tifffile first for tif/tiff
    if suf in {".tif", ".tiff"}:
        try:
            img = tiff.imread(str(img_path))
            if img is not None:
                if img.ndim == 3:
                    img = img[..., 0]
                return img.astype(np.float32)
        except Exception:
            pass

        # 2) Pillow fallback (often succeeds where tifffile/opencv fail)
        try:
            with Image.open(img_path) as im:
                im = im.convert("L")  # grayscale
                return np.array(im, dtype=np.float32)
        except Exception:
            pass

    # 3) Final fallback: OpenCV (for png/jpg etc)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    return img.astype(np.float32)




def _list_frames(frames_dir: Path) -> List[Path]:
    """List frame files inside a sequence folder."""
    exts = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp")
    frames: List[Path] = []
    for e in exts:
        frames.extend(frames_dir.glob(e))
    frames = sorted(frames)
    return frames


def _normalize_01(x: np.ndarray) -> np.ndarray:
    """Normalize an array to [0,1]."""
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    return (x - mn) / (mx - mn + 1e-8)


# ----------------------------
# Core baseline detector
# ----------------------------
def frame_difference_scores(sequence_dir: Path) -> np.ndarray:
    """
    Compute anomaly scores for one UCSD clip folder (e.g., Train001/Test001).
    Score at time t is mean absolute difference between frame(t) and frame(t-1).
    Returns length (num_frames - 1) normalized to [0,1].
    Skips corrupted / unreadable frames safely.
    """
    frames = _list_frames(sequence_dir)
    if len(frames) < 2:
        raise ValueError(f"Need at least 2 frames in {sequence_dir}, found {len(frames)}")

    # Read first valid frame
    idx = 0
    prev = None
    while idx < len(frames) and prev is None:
        try:
            prev = _read_gray(frames[idx])
        except Exception as e:
            print(f"[WARN] Skipping unreadable frame: {frames[idx].name} ({e})")
            idx += 1

    if prev is None:
        raise ValueError(f"No readable frames in {sequence_dir}")

    scores: List[float] = []
    bad = 0

    for f in frames[idx + 1:]:
        try:
            cur = _read_gray(f)
        except Exception as e:
            bad += 1
            print(f"[WARN] Skipping unreadable frame: {f.name} ({e})")
            continue

        # Safety: enforce same size
        if cur.shape != prev.shape:
            cur = cv2.resize(cur, (prev.shape[1], prev.shape[0]))

        diff = np.abs(cur - prev)
        scores.append(float(diff.mean()))
        prev = cur

    if len(scores) < 1:
        raise ValueError(
            f"Not enough readable frames in {sequence_dir} (skipped {bad})"
        )

    scores_arr = np.asarray(scores, dtype=np.float32)
    return _normalize_01(scores_arr)




# ----------------------------
# Runner over dataset
# ----------------------------

@dataclass
class UcsdRunResult:
    dataset: str
    split: str
    clip: str
    n_frames: int
    scores_path: str


def run_ucsd_split(
    ucsd_root: str | Path,
    dataset: str = "UCSDped1",
    split: str = "Test",
    out_dir: str | Path = "outputs/video_ucsd",
) -> List[UcsdRunResult]:
    """
    Run the baseline detector over all clips in UCSD dataset split.
    Saves one CSV per clip with per-frame anomaly scores.

    ucsd_root should be the folder that contains UCSDped1 and UCSDped2.
    Example: Path("data/ucsd") if you have data/ucsd/UCSDped1/...
    """
    ucsd_root = Path(ucsd_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = ucsd_root / dataset / split
    if not base.exists():
        raise FileNotFoundError(f"Could not find split folder: {base}")

    # Each clip is usually a folder like Train001, Test001, ...
    clip_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not clip_dirs:
        raise RuntimeError(f"No clip folders found inside: {base}")

    results: List[UcsdRunResult] = []
    for clip_dir in clip_dirs:
        scores = frame_difference_scores(clip_dir)

        # frames count = scores+1 because difference uses pairs
        n_frames = len(scores) + 1

        # Save CSV
        csv_path = out_dir / f"{dataset}_{split}_{clip_dir.name}_scores.csv"
        # Columns: frame_index (starting at 1), score
        # frame_index 1 corresponds to diff between frame0 and frame1
        import pandas as pd
        df = pd.DataFrame({
            "frame_index": np.arange(1, len(scores) + 1, dtype=int),
            "score": scores
        })
        df.to_csv(csv_path, index=False)

        results.append(
            UcsdRunResult(
                dataset=dataset,
                split=split,
                clip=clip_dir.name,
                n_frames=n_frames,
                scores_path=str(csv_path),
            )
        )

    return results



# Quick CLI usage
# ----------------------------

if __name__ == "__main__":
    """
    Example (run from project root):
    python -m src.detection_video

    Make sure your dataset lives at:
    data/ucsd/UCSDped1/Train, data/ucsd/UCSDped1/Test, etc.
    """
    # Adjust if your folder differs:
    ucsd_root = Path("data/ucsd/UCSD_Anomaly_Dataset.v1p2")


    print("Running UCSDped1 Test split baseline...")
    results = run_ucsd_split(ucsd_root=ucsd_root, dataset="UCSDped1", split="Test")

    print(f"Done. Wrote {len(results)} clip CSVs to outputs/video_ucsd/")
    print("First 3 results:")
    for r in results[:3]:
        print(r)
