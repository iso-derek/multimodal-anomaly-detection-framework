from __future__ import annotations

from pathlib import Path

import numpy as np

from src.detection_video import detect_video_clip_for_fusion


def extract_video_features(clip_path: str | Path) -> np.ndarray:
    """
    Build a lightweight feature vector for one video clip.

    Uses the existing video detector result and creates a compact summary.
    """
    result = detect_video_clip_for_fusion(clip_path, threshold=0.90)

    raw = float(result.get("score_raw", 0.0))
    norm = float(result.get("score_norm", 0.0))
    label = float(result.get("label", 0))

    # Compact summary vector
    feats = np.array([
        raw,
        norm,
        label,
    ], dtype=np.float32)

    return feats


def _normalise_distances(distances: np.ndarray) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float32)

    if distances.size == 0:
        return distances

    dmin = float(np.min(distances))
    dmax = float(np.max(distances))

    if dmax - dmin < 1e-12:
        return np.zeros_like(distances)

    return (distances - dmin) / (dmax - dmin)


def compare_video_set(
    clip_paths: list[str | Path],
    names: list[str] | None = None,
    threshold: float = 0.65,
) -> dict:
    """
    Compare multiple video clips against each other using distance from group centroid.
    """
    if not clip_paths:
        raise ValueError("No video clips were provided for comparison.")

    if names is None:
        names = [Path(p).name for p in clip_paths]

    if len(names) != len(clip_paths):
        raise ValueError("names and clip_paths must have the same length.")

    feature_matrix = np.vstack([extract_video_features(p) for p in clip_paths])
    centroid = np.mean(feature_matrix, axis=0)

    distances = np.linalg.norm(feature_matrix - centroid, axis=1)
    score_norm = _normalise_distances(distances)
    labels = (score_norm >= threshold).astype(int)

    items = []
    for i, name in enumerate(names):
        items.append({
            "name": name,
            "clip_path": str(clip_paths[i]),
            "score_raw": float(distances[i]),
            "score_norm": float(score_norm[i]),
            "label": int(labels[i]),
            "features": feature_matrix[i].tolist(),
        })

    items_sorted = sorted(items, key=lambda x: x["score_raw"], reverse=True)

    return {
        "modality": "video_compare",
        "method": "centroid_distance_video",
        "threshold": threshold,
        "n_items": int(len(clip_paths)),
        "centroid": centroid.tolist(),
        "items": items,
        "ranked_items": items_sorted,
        "most_abnormal": items_sorted[0] if items_sorted else None,
    }