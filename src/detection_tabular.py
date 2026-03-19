import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.common.normalise import normalise_percentile


def detect_tabular_anomalies(
    data: np.ndarray,
    contamination: float = 0.05,
    random_state: int = 42,
):
    data = np.asarray(data, dtype=float)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
    )

    labels = model.fit_predict(X)      # -1 anomaly, 1 normal
    scores = -model.score_samples(X)   # higher = more anomalous

    return {
        "labels": labels.tolist(),
        "scores": scores.tolist(),
        "n_anomalies": int((labels == -1).sum()),
        "method": "IsolationForest",
    }


def detect_tabular_for_fusion(
    data: np.ndarray,
    contamination: float = 0.05,
    random_state: int = 42,
    threshold: float = 0.45,
    calib: dict | None = None,
) -> dict:
    """
    Fusion-ready tabular detector.

    Changes made:
    - contamination increased from 0.01 to 0.05
      so obvious demo outliers are easier to detect
    - threshold reduced from 0.65 to 0.45
      so modality label is easier to trigger
    - score summary changed from p95 to max
      so the strongest anomaly directly drives the tabular score

    This works better for both small and large raw values because:
    - StandardScaler removes raw scale dominance
    - Isolation Forest works on standardized values
    - max captures the strongest anomaly clearly
    """
    data_arr = np.asarray(data, dtype=float)

    if data_arr.ndim == 1:
        data_arr = data_arr.reshape(-1, 1)

    out = detect_tabular_anomalies(
        data_arr,
        contamination=contamination,
        random_state=random_state,
    )

    scores = np.asarray(out.get("scores", []), dtype=float)

    # Use strongest anomaly directly instead of p95
    score_raw = float(np.max(scores)) if scores.size else 0.0

    if calib and "p10" in calib and "p90" in calib:
        score_norm = normalise_percentile(
            score_raw,
            float(calib["p10"]),
            float(calib["p90"]),
        )
    else:
        # Smooth squash into [0,1]
        score_norm = float(1.0 - np.exp(-score_raw))
        score_norm = float(np.clip(score_norm, 0.0, 1.0))

    label = 1 if score_norm >= threshold else 0

    labels = np.asarray(out.get("labels", []), dtype=int)
    n_anom = int((labels == -1).sum()) if labels.size else 0

    return {
        "modality": "tabular",
        "score_raw": score_raw,
        "score_norm": score_norm,
        "label": label,
        "method": "IsolationForest_max",
        "meta": {
            "n_samples": int(data_arr.shape[0]),
            "n_anomalies": n_anom,
            "contamination": contamination,
            "threshold": threshold,
            "scores_preview": out.get("scores", [])[:50],
            "calib_used": bool(calib is not None),
        },
    }