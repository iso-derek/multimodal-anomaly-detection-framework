import numpy as np


def detect_timeseries_anomalies(series: np.ndarray, window: int = 25, z_thresh: float = 3.0):
    """
    Simple rolling z-score anomaly detection.
    series: array-like shape (n,)
    Returns indices of anomalies and scores.
    """
    x = np.asarray(series).astype(float).flatten()

    if len(x) < window + 2:
        return {"anomaly_indices": [], "scores": [], "method": "rolling_zscore"}

    scores = np.zeros_like(x)
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
        "window": window
    }
