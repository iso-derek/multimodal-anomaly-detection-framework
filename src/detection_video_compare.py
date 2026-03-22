from __future__ import annotations

from pathlib import Path

import numpy as np

from src.detection_video import detect_video_clip_for_fusion


def extract_video_features(clip_path: str | Path) -> tuple[np.ndarray, dict]:
    """
    Build a lightweight feature vector for one video clip.

    Uses the existing video detector result and creates a compact summary.
    Returns both the feature vector and the original detector result.
    """
    result = detect_video_clip_for_fusion(clip_path, threshold=0.90)

    raw = float(result.get("score_raw", 0.0))
    norm = float(result.get("score_norm", 0.0))
    label = float(result.get("label", 0))

    meta = result.get("meta", {}) or {}

    # Optional metadata fields if available from the detector
    n_frames = float(meta.get("n_frames", 0.0))
    p95_motion = float(meta.get("p95_motion", raw))
    mean_motion = float(meta.get("mean_motion", 0.0))

    # Compact but slightly richer summary vector
    feats = np.array([
        raw,
        norm,
        label,
        n_frames,
        p95_motion,
        mean_motion,
    ], dtype=np.float32)

    return feats, result


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

    Returns:
    - per-video centroid distances
    - normalized comparison scores
    - labels
    - ranked results
    - original detector summaries
    """
    if not clip_paths:
        raise ValueError("No video clips were provided for comparison.")

    if names is None:
        names = [Path(p).name for p in clip_paths]

    if len(names) != len(clip_paths):
        raise ValueError("names and clip_paths must have the same length.")

    features_and_results = [extract_video_features(p) for p in clip_paths]
    feature_matrix = np.vstack([fr[0] for fr in features_and_results])
    detector_results = [fr[1] for fr in features_and_results]

    centroid = np.mean(feature_matrix, axis=0)

    distances = np.linalg.norm(feature_matrix - centroid, axis=1)
    score_norm = _normalise_distances(distances)
    labels = (score_norm >= threshold).astype(int)

    items = []
    for i, name in enumerate(names):
        detector_result = detector_results[i]
        detector_meta = detector_result.get("meta", {}) or {}

        items.append({
            "name": name,
            "clip_path": str(clip_paths[i]),
            "score_raw": float(distances[i]),
            "score_norm": float(score_norm[i]),
            "label": int(labels[i]),
            "features": feature_matrix[i].tolist(),
            "detector_score_raw": float(detector_result.get("score_raw", 0.0)),
            "detector_score_norm": float(detector_result.get("score_norm", 0.0)),
            "detector_label": int(detector_result.get("label", 0)),
            "n_frames": detector_meta.get("n_frames"),
            "p95_motion": detector_meta.get("p95_motion"),
            "mean_motion": detector_meta.get("mean_motion"),
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