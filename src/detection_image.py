import numpy as np


def detect_image_anomaly(model, image, threshold: float = None):
    """
    If model is provided (e.g., autoencoder), compute reconstruction error.
    Otherwise uses a simple heuristic score (variance / mean intensity).

    image can be:
    - numpy array (H,W) grayscale
    - numpy array (H,W,3) colour
    """
    img = np.asarray(image).astype(np.float32)

    # Normalize to [0,1] if looks like 0..255
    if img.max() > 1.5:
        img = img / 255.0

    # If colour, keep as colour; if grayscale, ensure (H,W,1) if needed by model
    if model is not None:
        x = img
        if x.ndim == 2:
            x = x[..., None]  # (H,W,1)
        x = x[None, ...]     # batch dimension (1,H,W,C)

        recon = model.predict(x, verbose=0)
        err = float(np.mean((x - recon) ** 2))
        is_anomaly = bool(err > (threshold if threshold is not None else err))
        return {"score": err, "is_anomaly": is_anomaly, "method": "reconstruction_error"}

    # No model fallback heuristic
    score = float(img.std())
    if threshold is None:
        threshold = 0.25  # simple default; tune later

    return {"score": score, "is_anomaly": bool(score > threshold), "method": "std_heuristic"}
