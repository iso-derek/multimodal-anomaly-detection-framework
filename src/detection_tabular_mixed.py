from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.normalise import normalise_percentile


def _ensure_dataframe(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise ValueError("Mixed tabular input must be a pandas DataFrame, dict, or list of dicts.")

    if df.empty:
        raise ValueError("Input data is empty.")

    return df


def _split_columns(df: pd.DataFrame):
    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def _build_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))

    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    if not transformers:
        raise ValueError("No usable columns found for preprocessing.")

    return ColumnTransformer(transformers=transformers)


def fit_mixed_tabular_model(
    train_data,
    contamination: float = 0.05,
    random_state: int = 42
) -> Pipeline:
    df = _ensure_dataframe(train_data)

    numeric_cols, categorical_cols = _split_columns(df)
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("detector", IsolationForest(contamination=contamination, random_state=random_state)),
    ])

    pipeline.fit(df)
    return pipeline


def detect_mixed_tabular_for_fusion(
    model_pipeline: Pipeline,
    data,
    threshold: float = 0.65,
    calib: dict | None = None,
) -> dict:
    df = _ensure_dataframe(data)

    labels = model_pipeline.predict(df)           # -1 anomaly, 1 normal
    scores = -model_pipeline.score_samples(df)    # higher = more anomalous

    score_raw = float(np.percentile(scores, 95)) if scores.size else 0.0

    if calib and "p10" in calib and "p90" in calib:
        score_norm = float(
            normalise_percentile(
                score_raw,
                float(calib["p10"]),
                float(calib["p90"])
            )
        )
    else:
        score_norm = float(1.0 - np.exp(-score_raw))
        score_norm = float(np.clip(score_norm, 0.0, 1.0))

    label = 1 if score_norm >= threshold else 0
    n_anom = int((labels == -1).sum()) if labels.size else 0

    return {
        "modality": "tabular",
        "score_raw": score_raw,
        "score_norm": score_norm,
        "label": label,
        "method": "IsolationForestMixed_p95",
        "meta": {
            "n_samples": int(len(df)),
            "n_features": int(df.shape[1]),
            "n_anomalies": n_anom,
            "threshold": threshold,
            "scores_preview": scores.tolist()[:50],
            "columns": df.columns.tolist(),
            "calib_used": bool(calib is not None),
        },
    }