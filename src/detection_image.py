from __future__ import annotations

import numpy as np
from skimage.transform import resize

from src.common.normalise import normalise_percentile


def _to_model_ready_image(image: np.ndarray, target_size=(128, 128)) -> np.ndarray:
    img = np.asarray(image).astype(np.float32)

    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.shape[-1] == 3:
            img = img.mean(axis=-1)

    if img.max() > 1.5:
        img = img / 255.0

    img = resize(
        img,
        target_size,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    img = np.clip(img, 0.0, 1.0)
    img = img[..., None]   # (H, W, 1)
    return img


def _to_float_image(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image).astype(np.float32)

    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.shape[-1] == 3:
            img = img.mean(axis=-1)

    if img.max() > 1.5:
        img = img / 255.0

    img = np.clip(img, 0.0, 1.0)
    return img


def detect_image_for_fusion(
    model,
    image,
    threshold: float = 0.65,
    calib: dict | None = None,
) -> dict:
    """
    If model is provided (autoencoder), uses reconstruction MSE.
    Else uses std heuristic.

    Returns fused-ready dict.
    """

    if model is not None:
        x = _to_model_ready_image(image, target_size=(128, 128))
        x = x[None, ...]  # (1, 128, 128, 1)

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
            "modality": "image",
            "score_raw": score_raw,
            "score_norm": score_norm,
            "label": label,
            "method": "reconstruction_error",
            "meta": {
                "original_shape": tuple(np.asarray(image).shape),
                "model_input_shape": tuple(x.shape),
                "threshold": threshold,
                "calib_used": bool(calib is not None),
            },
        }

    # fallback heuristic
    img = _to_float_image(image)
    score_raw = float(img.std())

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
        "modality": "image",
        "score_raw": score_raw,
        "score_norm": score_norm,
        "label": label,
        "method": "std_heuristic",
        "meta": {
            "original_shape": tuple(np.asarray(image).shape),
            "threshold": threshold,
            "calib_used": bool(calib is not None),
        },
    }