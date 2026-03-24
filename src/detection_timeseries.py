import numpy as np

from src.common.normalise import normalise_percentile


def detect_timeseries_anomalies(
    series: np.ndarray,
    window: int = 25,
    z_thresh: float = 3.0,
):
    x = np.asarray(series, dtype=float).flatten()

    if len(x) < window + 2:
        return {
            "anomaly_indices": [],
            "scores": [],
            "method": "rolling_zscore",
            "z_thresh": z_thresh,
            "window": window,
        }

    scores = np.zeros_like(x, dtype=float)
    anomaly_indices = []

    for i in range(window, len(x)):
        w = x[i - window:i]
        mu = w.mean()
        sigma = w.std() + 1e-8
        z = abs((x[i] - mu) / sigma)
        scores[i] = z

        if z >= z_thresh:
            anomaly_indices.append(i)

    return {
        "anomaly_indices": anomaly_indices,
        "scores": scores.tolist(),
        "method": "rolling_zscore",
        "z_thresh": z_thresh,
        "window": window,
    }


def detect_timeseries_for_fusion(
    series: np.ndarray,
    window: int = 25,
    z_thresh: float = 3.0,
    threshold: float = 0.65,
    calib: dict | None = None,
) -> dict:
    """
    Fusion-ready time-series detector.

    Uses rolling z-score anomaly scoring and summarizes anomaly strength
    using the maximum z-score observed in the sequence.
    """
    series_arr = np.asarray(series, dtype=float).flatten()

    out = detect_timeseries_anomalies(series_arr, window=window, z_thresh=z_thresh)
    scores = np.asarray(out.get("scores", []), dtype=float)

    score_raw = float(np.max(scores)) if scores.size else 0.0

    if calib and "p10" in calib and "p90" in calib:
        score_norm = float(
            normalise_percentile(
                score_raw,
                float(calib["p10"]),
                float(calib["p90"]),
            )
        )
    else:
        # Smooth mapping of z-score -> [0,1]
        score_norm = float(1.0 - np.exp(-score_raw / max(z_thresh, 1e-6)))
        score_norm = float(np.clip(score_norm, 0.0, 1.0))

    label = int(score_norm >= threshold)

    return {
        "modality": "timeseries",
        "score_raw": score_raw,
        "score_norm": score_norm,
        "label": label,
        "method": "rolling_zscore_max",
        "meta": {
            "window": window,
            "z_thresh": z_thresh,
            "threshold": threshold,
            "n": int(series_arr.size),
            "anomaly_indices": out.get("anomaly_indices", []),
            "scores_preview": out.get("scores", [])[:50],
            "calib_used": bool(calib is not None),
        },
    }