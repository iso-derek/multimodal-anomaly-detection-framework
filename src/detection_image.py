from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.io import imread
from skimage.metrics import structural_similarity
from skimage.transform import resize

from src.common.normalise import normalise_percentile


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _to_float_image(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image).astype(np.float32)

    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.shape[-1] == 3:
            img = img.mean(axis=-1)

    if img.max() > 1.5:
        img = img / 255.0

    return np.clip(img, 0.0, 1.0)


def _resize_to_match(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if image.shape == target_shape:
        return image.astype(np.float32)

    return resize(
        image,
        target_shape,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def _score_with_std_heuristic(
    image: np.ndarray,
    threshold: float,
    calib: dict | None = None,
) -> dict:
    score_raw = float(np.std(image))

    if calib and "p10" in calib and "p90" in calib:
        score_norm = float(
            normalise_percentile(
                score_raw,
                float(calib["p10"]),
                float(calib["p90"]),
            )
        )
    else:
        score_norm = float(np.clip(score_raw, 0.0, 1.0))

    label = int(score_norm >= threshold)

    return {
        "score_raw": score_raw,
        "score_norm": score_norm,
        "label": label,
        "method": "std_heuristic",
        "threshold": threshold,
    }


def _list_reference_images(reference_dir: str | Path) -> list[Path]:
    ref_dir = Path(reference_dir)
    if not ref_dir.exists() or not ref_dir.is_dir():
        return []

    return sorted(
        p for p in ref_dir.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    )


def _score_against_reference_bank(
    image: np.ndarray,
    reference_dir: str | Path,
    threshold: float,
) -> dict:
    reference_paths = _list_reference_images(reference_dir)

    if not reference_paths:
        raise FileNotFoundError(f"No reference images found in: {reference_dir}")

    img = _to_float_image(image)
    target_shape = img.shape[:2]

    best_ssim = -1.0
    best_reference_name = None

    for ref_path in reference_paths:
        ref_img = imread(str(ref_path))
        ref_img = _to_float_image(ref_img)
        ref_img = _resize_to_match(ref_img, target_shape)

        ssim_value = float(structural_similarity(img, ref_img, data_range=1.0))

        if ssim_value > best_ssim:
            best_ssim = ssim_value
            best_reference_name = ref_path.name

    score_raw = float(1.0 - best_ssim)
    score_norm = float(np.clip(score_raw, 0.0, 1.0))
    label = int(score_norm >= threshold)

    return {
        "score_raw": score_raw,
        "score_norm": score_norm,
        "label": label,
        "method": "reference_ssim",
        "best_reference": best_reference_name,
        "best_ssim": best_ssim,
        "n_references": len(reference_paths),
        "threshold": threshold,
    }


def detect_image_for_fusion(
    model,
    image,
    threshold: float = 0.65,
    calib: dict | None = None,
    reference_dir: str | Path | None = None,
    reference_threshold: float = 0.35,
) -> dict:
    """
    Image anomaly detection for fusion.

    Priority:
    1. Try autoencoder if a model is available
    2. Fall back to reference-bank SSIM if reference_dir is available
    3. Fall back to std heuristic otherwise

    Parameters
    ----------
    threshold:
        Threshold used for autoencoder mode and std heuristic mode.
    reference_threshold:
        Threshold used specifically for the reference-bank SSIM fallback.
    """
    img = _to_float_image(image)
    fallback_reason = None

    
    # Mode 1: Autoencoder
    
    if model is not None:
        try:
            x = img
            if x.ndim == 2:
                x = x[..., None]
            x = x[None, ...]  # (1, H, W, C)

            recon = model.predict(x, verbose=0)
            score_raw = float(np.mean((x - recon) ** 2))

            if calib and "p10" in calib and "p90" in calib:
                score_norm = float(
                    normalise_percentile(
                        score_raw,
                        float(calib["p10"]),
                        float(calib["p90"]),
                    )
                )
            else:
                score_norm = float(np.clip(score_raw, 0.0, 1.0))

            label = int(score_norm >= threshold)

            return {
                "score_raw": score_raw,
                "score_norm": score_norm,
                "label": label,
                "method": "reconstruction_error",
                "threshold": threshold,
            }
        except Exception as e:
            fallback_reason = f"Autoencoder unavailable during inference: {e}"

    
    # Mode 2: Reference bank
   
    if reference_dir is not None:
        try:
            reference_result = _score_against_reference_bank(
                image=img,
                reference_dir=reference_dir,
                threshold=reference_threshold,
            )
            if fallback_reason:
                reference_result["fallback_reason"] = fallback_reason
            return reference_result
        except Exception as e:
            if fallback_reason:
                fallback_reason = f"{fallback_reason} | Reference-bank fallback failed: {e}"
            else:
                fallback_reason = f"Reference-bank fallback failed: {e}"

    
    # Mode 3: Lightweight heuristic
    
    heuristic_result = _score_with_std_heuristic(
        image=img,
        threshold=threshold,
        calib=calib,
    )

    if fallback_reason:
        heuristic_result["fallback_reason"] = fallback_reason

    return heuristic_result