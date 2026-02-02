from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np


# ----------------------------
# Utilities
# ----------------------------

def _to_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def normalize_01(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0,1]. Safe for empty/constant arrays."""
    scores = _to_float_array(scores)
    if scores.size == 0:
        return scores
    mn = float(scores.min())
    mx = float(scores.max())
    if mx - mn < 1e-8:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def robust_scalar_from_scores(scores: np.ndarray, mode: str = "max") -> float:
    """
    Convert a score series into a single scalar anomaly score.
    mode:
      - "max"  : strongest anomaly
      - "mean" : average anomaly level
      - "p95"  : robust high percentile
    """
    scores = _to_float_array(scores)
    if scores.size == 0:
        return 0.0

    mode = mode.lower().strip()
    if mode == "max":
        return float(scores.max())
    if mode == "mean":
        return float(scores.mean())
    if mode == "p95":
        return float(np.percentile(scores, 95))

    raise ValueError("mode must be one of: 'max', 'mean', 'p95'")


# ----------------------------
# Modality summary
# ----------------------------

@dataclass
class ModalitySummary:
    modality: str
    score: float                 # normalized [0,1]
    raw_score: float
    label: Optional[int] = None  # 1 = normal, -1 = anomaly
    meta: Optional[Dict[str, Any]] = None


def summarize_tabular(result: Dict[str, Any]) -> ModalitySummary:
    scores = normalize_01(_to_float_array(result.get("scores", [])))
    scalar = robust_scalar_from_scores(scores, mode="max")
    label = -1 if int(result.get("n_anomalies", 0)) > 0 else 1

    return ModalitySummary(
        modality="tabular",
        score=float(scalar),
        raw_score=float(scalar),
        label=label,
        meta={"method": result.get("method"), "n_anomalies": result.get("n_anomalies")}
    )


def summarize_timeseries(result: Dict[str, Any]) -> ModalitySummary:
    """
    Handles:
      {'anomaly_indices': [...], 'scores': [...], 'method': 'rolling_zscore'}
    """
    scores = _to_float_array(result.get("scores", []))

    if scores.size == 0:
        scalar = 0.0
    else:
        scalar = robust_scalar_from_scores(normalize_01(scores), mode="max")

    n_anoms = len(result.get("anomaly_indices", []))
    label = -1 if n_anoms > 0 else 1

    return ModalitySummary(
        modality="timeseries",
        score=float(scalar),
        raw_score=float(scalar),
        label=label,
        meta={"method": result.get("method"), "n_anomalies": n_anoms}
    )


def summarize_image(result: Dict[str, Any] | float | int) -> ModalitySummary:
    if isinstance(result, dict):
        raw = float(result.get("score", 0.0))
        label = result.get("label", None)
        meta = {k: v for k, v in result.items() if k not in {"score", "label"}}
    else:
        raw = float(result)
        label = None
        meta = None

    score = raw if 0.0 <= raw <= 1.0 else 0.0

    return ModalitySummary(
        modality="image",
        score=float(score),
        raw_score=float(raw),
        label=label if isinstance(label, int) else None,
        meta=meta
    )


def summarize_video(results: List[Any], score_mode: str = "p95") -> ModalitySummary:
    import pandas as pd

    clip_scores: List[float] = []
    n_clips = 0

    for r in results:
        path = getattr(r, "scores_path", None)
        if not path:
            continue
        n_clips += 1
        df = pd.read_csv(path)
        if "score" not in df.columns:
            continue
        scores = _to_float_array(df["score"].to_numpy())
        clip_scores.append(
            robust_scalar_from_scores(scores, mode=score_mode)
        )

    if not clip_scores:
        return ModalitySummary(
            modality="video",
            score=0.0,
            raw_score=0.0,
            label=1,
            meta={"n_clips": 0}
        )

    clip_scores_n = normalize_01(_to_float_array(clip_scores))
    fused_score = robust_scalar_from_scores(clip_scores_n, mode="p95")
    label = -1 if fused_score >= 0.7 else 1

    return ModalitySummary(
        modality="video",
        score=float(fused_score),
        raw_score=float(fused_score),
        label=label,
        meta={"n_clips": n_clips, "score_mode": score_mode}
    )


# ----------------------------
# Fusion strategy
# ----------------------------

def fuse_weighted_average(
    summaries: List[ModalitySummary],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:

    if not summaries:
        return {"final_score": 0.0, "final_label": 1, "by_modality": []}

    if weights is None:
        weights = {}

    ws = []
    xs = []

    for s in summaries:
        ws.append(float(weights.get(s.modality, 1.0)))
        xs.append(float(s.score))

    ws_arr = np.asarray(ws, dtype=np.float32)
    xs_arr = np.asarray(xs, dtype=np.float32)

    denom = ws_arr.sum() if ws_arr.sum() > 1e-8 else 1.0
    final_score = float((ws_arr * xs_arr).sum() / denom)

    # Strong-modality rule
    strong_anomaly = bool((xs_arr >= 0.8).any())
    final_label = -1 if (strong_anomaly or final_score >= 0.6) else 1

    return {
        "final_score": final_score,
        "final_label": final_label,
        "by_modality": [
            {
                "modality": s.modality,
                "score": s.score,
                "label": s.label,
                "meta": s.meta
            }
            for s in summaries
        ],
    }
