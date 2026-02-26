import numpy as np
from src.common.normalise import normalise_percentile

def detect_image_for_fusion(model, image, threshold: float = 0.65, calib: dict | None = None) -> dict:
    """
    If model is provided (autoencoder), uses reconstruction MSE.
    Else uses std heuristic.

    Returns fused-ready dict.
    """
    img = np.asarray(image).astype(np.float32)

    if img.max() > 1.5:
        img = img / 255.0

    if model is not None:
        x = img
        if x.ndim == 2:
            x = x[..., None]
        x = x[None, ...]  # (1,H,W,C)

        recon = model.predict(x, verbose=0)
        score_raw = float(np.mean((x - recon) ** 2))

        if calib and "p10" in calib and "p90" in calib:
            score_norm = normalise_percentile(score_raw, float(calib["p10"]), float(calib["p90"]))
        else:
            score_norm = float(np.clip(score_raw, 0.0, 1.0))

        label = 1 if score_norm >= threshold else 0
        return {
            "modality": "image",
            "score_raw": score_raw,
            "score_norm": score_norm,
            "label": label,
            "method": "reconstruction_error",
            "meta": {"shape": tuple(img.shape), "threshold": threshold, "calib_used": bool(calib is not None)},
        }

    # fallback heuristic
    score_raw = float(img.std())
    if calib and "p10" in calib and "p90" in calib:
        score_norm = normalise_percentile(score_raw, float(calib["p10"]), float(calib["p90"]))
    else:
        score_norm = float(np.clip(score_raw, 0.0, 1.0))

    label = 1 if score_norm >= threshold else 0
    return {
        "modality": "image",
        "score_raw": score_raw,
        "score_norm": score_norm,
        "label": label,
        "method": "std_heuristic",
        "meta": {"shape": tuple(img.shape), "threshold": threshold, "calib_used": bool(calib is not None)},
    }