from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize


def _to_float_image(image: np.ndarray, target_size: tuple[int, int] | None = None) -> np.ndarray:
    img = np.asarray(image).astype(np.float32)

    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.shape[-1] == 3:
            img = img.mean(axis=-1)

    if img.max() > 1.5:
        img = img / 255.0

    if target_size is not None:
        img = resize(
            img,
            target_size,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)

    img = np.clip(img, 0.0, 1.0)
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


def _compute_ssim_matrix(
    images: list[np.ndarray],
    target_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """
    Pairwise SSIM between uploaded images.
    Higher SSIM means more visually similar.
    """
    n = len(images)
    sims = np.eye(n, dtype=np.float32)

    prepared = [_to_float_image(img, target_size=target_size) for img in images]

    for i in range(n):
        for j in range(i + 1, n):
            score = float(ssim(prepared[i], prepared[j], data_range=1.0))
            sims[i, j] = score
            sims[j, i] = score

    return sims


def compare_image_set(
    images: list[np.ndarray],
    names: list[str] | None = None,
    threshold: float = 0.65,
) -> dict:
    """
    Compare multiple images using:
    1. centroid distance (main anomaly ranking)
    2. pairwise SSIM (supporting similarity evidence)

    Returns:
    - per-image centroid distances
    - normalized anomaly scores
    - labels
    - ranked results
    - pairwise SSIM matrix
    - per-image average SSIM and closest/farthest partners
    """
    if not images:
        raise ValueError("No images were provided for comparison.")

    if names is None:
        names = [f"image_{i+1}" for i in range(len(images))]

    if len(names) != len(images):
        raise ValueError("names and images must have the same length.")

    n = len(images)

    # Main anomaly logic: centroid distance
    feature_matrix = np.vstack([extract_image_features(img) for img in images])
    centroid = np.mean(feature_matrix, axis=0)

    distances = np.linalg.norm(feature_matrix - centroid, axis=1)
    score_norm = _normalise_distances(distances)
    labels = (score_norm >= threshold).astype(int)

    # Supporting similarity logic: SSIM
    ssim_matrix = _compute_ssim_matrix(images)

    items = []
    for i, name in enumerate(names):
        # Ignore self-similarity on the diagonal
        other_indices = [j for j in range(n) if j != i]

        if other_indices:
            sims_to_others = [float(ssim_matrix[i, j]) for j in other_indices]
            avg_ssim = float(np.mean(sims_to_others))

            most_similar_idx = max(other_indices, key=lambda j: float(ssim_matrix[i, j]))
            least_similar_idx = min(other_indices, key=lambda j: float(ssim_matrix[i, j]))

            most_similar_name = names[most_similar_idx]
            least_similar_name = names[least_similar_idx]
            most_similar_ssim = float(ssim_matrix[i, most_similar_idx])
            least_similar_ssim = float(ssim_matrix[i, least_similar_idx])
        else:
            avg_ssim = 1.0
            most_similar_name = name
            least_similar_name = name
            most_similar_ssim = 1.0
            least_similar_ssim = 1.0

        items.append({
            "name": name,
            "score_raw": float(distances[i]),
            "score_norm": float(score_norm[i]),
            "label": int(labels[i]),
            "features": feature_matrix[i].tolist(),
            "avg_ssim_to_others": avg_ssim,
            "most_similar_image": most_similar_name,
            "most_similar_ssim": most_similar_ssim,
            "least_similar_image": least_similar_name,
            "least_similar_ssim": least_similar_ssim,
        })

    items_sorted = sorted(items, key=lambda x: x["score_raw"], reverse=True)

    return {
        "modality": "image_compare",
        "method": "centroid_distance_image",
        "supporting_method": "pairwise_ssim",
        "threshold": threshold,
        "n_items": int(n),
        "centroid": centroid.tolist(),
        "items": items,
        "ranked_items": items_sorted,
        "most_abnormal": items_sorted[0] if items_sorted else None,
        "ssim_matrix": ssim_matrix.tolist(),
        "ssim_names": names,
    }