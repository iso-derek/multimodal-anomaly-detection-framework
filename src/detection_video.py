"""
Video (UCSD) anomaly detection — baseline.
Works on UCSDped1 / UCSDped2 frame folders (tif/png).
Produces an anomaly score per frame using simple frame-difference motion energy.

Fusion summary (Option A):
- Use p95 of per-frame scores as the clip-level score (robust spike capture).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import cv2
import tifffile as tiff
from PIL import Image  # FIX: required for TIFF fallback

from src.common.normalise import normalise_percentile


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

        # 2) Pillow fallback
        try:
            with Image.open(img_path) as im:
                im = im.convert("L")
                return np.array(im, dtype=np.float32)
        except Exception:
            pass

    # 3) Final fallback: OpenCV
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    return img.astype(np.float32)


def _list_frames(frames_dir: Path) -> List[Path]:
    exts = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp")
    frames: List[Path] = []
    for e in exts:
        frames.extend(frames_dir.glob(e))
    return sorted(frames)


def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    return (x - mn) / (mx - mn + 1e-8)


# ----------------------------
# Core baseline detector
# ----------------------------
def frame_difference_scores(sequence_dir: Path) -> np.ndarray:
    """
    Score at time t = mean abs diff between frame(t) and frame(t-1).
    Returns length (num_frames - 1) normalized to [0,1].
    Skips unreadable frames safely.
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

        if cur.shape != prev.shape:
            cur = cv2.resize(cur, (prev.shape[1], prev.shape[0]))

        diff = np.abs(cur - prev)
        scores.append(float(diff.mean()))
        prev = cur

    if len(scores) < 1:
        raise ValueError(f"Not enough readable frames in {sequence_dir} (skipped {bad})")

    return _normalize_01(np.asarray(scores, dtype=np.float32))


# ----------------------------
# Fusion-ready single clip summary (Option A = p95)
# ----------------------------
def detect_video_clip_for_fusion(
    clip_dir: str | Path,
    threshold: float = 0.65,
    calib: dict | None = None,
) -> dict:
    """
    Clip -> fused-ready dict using p95 of per-frame motion scores.
    """
    clip_dir = Path(clip_dir)
    scores = frame_difference_scores(clip_dir)  # already [0,1]
    score_raw = float(np.percentile(scores, 95))  # ✅ Option A

    if calib and "p10" in calib and "p90" in calib:
        score_norm = normalise_percentile(score_raw, float(calib["p10"]), float(calib["p90"]))
    else:
        score_norm = float(np.clip(score_raw, 0.0, 1.0))

    label = 1 if score_norm >= threshold else 0

    return {
        "modality": "video",
        "score_raw": score_raw,
        "score_norm": score_norm,
        "label": label,
        "method": "frame_diff_p95",
        "meta": {
            "clip_dir": str(clip_dir),
            "n_scores": int(len(scores)),
            "threshold": threshold,
            "calib_used": bool(calib is not None),
        },
    }


# ----------------------------
# Dataset runner (kept for experiments)
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
    Runs baseline detector over all clips in UCSD split.
    Saves CSV per clip with per-frame anomaly scores.
    """
    import pandas as pd

    ucsd_root = Path(ucsd_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = ucsd_root / dataset / split
    if not base.exists():
        raise FileNotFoundError(f"Could not find split folder: {base}")

    clip_dirs = sorted([p for p in base.iterdir() if p.is_dir() and not p.name.lower().endswith("_gt")])
    if not clip_dirs:
        raise RuntimeError(f"No clip folders found inside: {base}")

    results: List[UcsdRunResult] = []
    for clip_dir in clip_dirs:
        scores = frame_difference_scores(clip_dir)
        n_frames = len(scores) + 1

        csv_path = out_dir / f"{dataset}_{split}_{clip_dir.name}_scores.csv"
        df = pd.DataFrame({
            "frame_index": np.arange(1, len(scores) + 1, dtype=int),
            "score": scores
        })
        df.to_csv(csv_path, index=False)

        results.append(UcsdRunResult(dataset=dataset, split=split, clip=clip_dir.name, n_frames=n_frames, scores_path=str(csv_path)))

    return results