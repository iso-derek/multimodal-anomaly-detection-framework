import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_tabular_anomalies(data: np.ndarray, contamination: float = 0.05, random_state: int = 42):
    """
    Detect anomalies in tabular data.
    Expects shape (n_samples, n_features) or (n_samples,).
    Returns a dict with labels and scores.
    """
    data = np.asarray(data)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(X)  # -1 anomaly, 1 normal
    scores = -model.score_samples(X)  # higher = more anomalous

    return {
        "labels": labels.tolist(),
        "scores": scores.tolist(),
        "n_anomalies": int((labels == -1).sum()),
        "method": "IsolationForest"
    }
