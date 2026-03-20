from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from skimage.io import imread


def _to_float_image(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image).astype(np.float32)

    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.shape[-1] == 3:
            img = img.mean(axis=-1)

    if img.max() > 1.5:
        img = img / 255.0

    return img


def load_image_input(item: str | Path | Any) -> np.ndarray:
    """
    Supports:
    - file path
    - Path object
    - uploaded file object from Streamlit
    """
    img = imread(item)
    return _to_float_image(img)


def extract_image_features(image: np.ndarray) -> np.ndarray:
    """
    Lightweight feature vector for group-based comparison.
    """
    img = _to_float_image(image).flatten()

    if img.size == 0:
        return np.zeros(5, dtype=np.float32)

    feats = np.array([
        float(np.mean(img)),
        float(np.std(img)),
        float(np.min(img)),
        float(np.max(img)),
        float(np.percentile(img, 95)),
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


def compare_image_set(
    images: list[np.ndarray],
    names: list[str] | None = None,
    threshold: float = 0.65,
) -> dict:
    """
    Compare multiple images against each other using distance from group centroid.

    Returns:
    - per-image distances
    - normalized scores
    - labels
    - ranked results
    """
    if not images:
        raise ValueError("No images were provided for comparison.")

    if names is None:
        names = [f"image_{i+1}" for i in range(len(images))]

    if len(names) != len(images):
        raise ValueError("names and images must have the same length.")

    feature_matrix = np.vstack([extract_image_features(img) for img in images])
    centroid = np.mean(feature_matrix, axis=0)

    distances = np.linalg.norm(feature_matrix - centroid, axis=1)
    score_norm = _normalise_distances(distances)
    labels = (score_norm >= threshold).astype(int)

    items = []
    for i, name in enumerate(names):
        items.append({
            "name": name,
            "score_raw": float(distances[i]),
            "score_norm": float(score_norm[i]),
            "label": int(labels[i]),
            "features": feature_matrix[i].tolist(),
        })

    items_sorted = sorted(items, key=lambda x: x["score_raw"], reverse=True)

    return {
        "modality": "image_compare",
        "method": "centroid_distance_image",
        "threshold": threshold,
        "n_items": int(len(images)),
        "centroid": centroid.tolist(),
        "items": items,
        "ranked_items": items_sorted,
        "most_abnormal": items_sorted[0] if items_sorted else None,
    }