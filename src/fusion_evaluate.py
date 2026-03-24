from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from skimage.io import imread

from src.detection_tabular import detect_tabular_for_fusion
from src.detection_timeseries import detect_timeseries_for_fusion
from src.detection_image import detect_image_for_fusion
from src.detection_video import detect_video_clip_for_fusion


def load_image(path: str) -> np.ndarray:
    """
    Loads image as numpy array.
    - If your image AE is RGB, do not convert to grayscale here.
    - X-ray images are typically already grayscale anyway.
    """
    img = imread(path)
    return img.astype(np.float32)


def pick_first_ucsd_clip(video_cfg: dict) -> Path:
    """
    video_cfg = {"root": "...", "dataset": "UCSDped1", "split": "Test"}
    Picks the first clip folder in that split (e.g., Test001).
    """
    base = Path(video_cfg["root"]) / video_cfg["dataset"] / video_cfg["split"]
    if not base.exists():
        raise FileNotFoundError(f"UCSD split folder not found: {base}")

    clip_dirs = sorted(
        [p for p in base.iterdir() if p.is_dir() and not p.name.lower().endswith("_gt")]
    )
    if not clip_dirs:
        raise RuntimeError(f"No UCSD clip folders found in: {base}")

    return clip_dirs[0]


def fuse_weighted_average(
    results: dict,
    weights: dict,
    strong_threshold: float = 0.80,
    strong_threshold_video: float = 0.95,
    fused_threshold: float = 0.65,
    vote_k: int = 2,
    strong_thresholds: dict | None = None,
) -> dict:
    """
    Minimal fusion that accepts any subset of modalities.

    results:
        {
            "tabular": {...},
            "timeseries": {...},
            "image": {...},
            "video": {...}
        }

    Each modality dict must contain:
    - score_norm
    - label

    Supports:
    - weighted fusion
    - vote-aware decision
    - strong-modality override
    - optional per-modality strong thresholds
    """
    if strong_thresholds is None:
        strong_thresholds = {}

    modality_thresholds = {
        "tabular": float(strong_thresholds.get("tabular", strong_threshold)),
        "timeseries": float(strong_thresholds.get("timeseries", strong_threshold)),
        "image": float(strong_thresholds.get("image", strong_threshold)),
        "video": float(strong_thresholds.get("video", strong_threshold_video)),
    }

    # Strong-modality override
    strong_hits = []
    for m, r in results.items():
        s = float(r["score_norm"])
        thr = float(modality_thresholds.get(m, strong_threshold))
        if s >= thr:
            strong_hits.append((m, s, thr))

    if strong_hits:
        strong_hits.sort(key=lambda x: x[1], reverse=True)
        top_m, top_s, top_thr = strong_hits[0]
        return {
            "final_score": float(top_s),
            "final_label": 1,
            "reason": f"strong modality '{top_m}' score={top_s:.4f} >= {top_thr:.2f}",
            "by_modality": {
                m: {
                    "score": float(r["score_norm"]),
                    "label": int(r["label"]),
                    "method": r.get("method", ""),
                    "meta": r.get("meta", {}),
                    "strong_threshold": float(modality_thresholds.get(m, strong_threshold)),
                }
                for m, r in results.items()
            },
        }

    # Weighted average score
    num, den = 0.0, 0.0
    for m, r in results.items():
        w = float(weights.get(m, 1.0))
        s = float(r["score_norm"])
        num += w * s
        den += w

    fused_score = num / den if den > 0 else 0.0
    fused_score = float(np.clip(fused_score, 0.0, 1.0))

    # Vote + score threshold
    votes = sum(1 for r in results.values() if int(r["label"]) == 1)
    vote_label = 1 if votes >= vote_k else 0
    score_label = 1 if fused_score >= fused_threshold else 0
    final_label = 1 if (vote_label == 1 or score_label == 1) else 0

    return {
        "final_score": fused_score,
        "final_label": final_label,
        "reason": (
            f"fused_score={fused_score:.4f} (thr={fused_threshold:.2f}) -> {score_label} | "
            f"votes={votes} (k={vote_k}) -> {vote_label}"
        ),
        "by_modality": {
            m: {
                "score": float(r["score_norm"]),
                "label": int(r["label"]),
                "method": r.get("method", ""),
                "meta": r.get("meta", {}),
                "strong_threshold": float(modality_thresholds.get(m, strong_threshold)),
            }
            for m, r in results.items()
        },
    }


def run_case(name: str, tabular_data, ts_data, image_path: str, video_cfg: dict, weights: dict) -> dict:
    # Tabular
    tab_result = detect_tabular_for_fusion(np.asarray(tabular_data))

    # Time-series
    ts_result = detect_timeseries_for_fusion(np.asarray(ts_data), window=3)

    # Image
    img = load_image(image_path)
    img_result = detect_image_for_fusion(model=None, image=img)

    # Video
    clip_dir = pick_first_ucsd_clip(video_cfg)
    vid_result = detect_video_clip_for_fusion(clip_dir, threshold=0.90)

    results = {
        "tabular": tab_result,
        "timeseries": ts_result,
        "image": img_result,
        "video": vid_result,
    }

    fused = fuse_weighted_average(results, weights=weights)

    return {
        "case": name,
        "final_score": fused["final_score"],
        "final_label": fused["final_label"],
        "reason": fused.get("reason", ""),
        "by_modality": fused["by_modality"],
        "inputs": {
            "image_path": image_path,
            "video_clip": str(clip_dir),
        },
    }


def run_suite(image_path: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    weights = {
        "video": 2.0,
        "tabular": 1.5,
        "timeseries": 1.0,
        "image": 0.8,
    }

    video_ped1_test = {
        "root": "data/ucsd/UCSD_Anomaly_Dataset.v1p2",
        "dataset": "UCSDped1",
        "split": "Test",
    }
    video_ped1_train = {
        "root": "data/ucsd/UCSD_Anomaly_Dataset.v1p2",
        "dataset": "UCSDped1",
        "split": "Train",
    }

    ts_data_anom = [0.0, 0.0, 0.1, 2.0, 0.1, 0.0, 0.1]
    ts_data_normal = [0.1, 0.1, 0.11, 0.1, 0.09, 0.1, 0.1]

    cases = [
        run_case(
            name="tabular_anomaly_with_test_video_clip",
            tabular_data=[1, 1, 1, 100, 1, 1],
            ts_data=ts_data_anom,
            image_path=image_path,
            video_cfg=video_ped1_test,
            weights=weights,
        ),
        run_case(
            name="all_normal_with_train_video_clip",
            tabular_data=[1, 1, 1, 1, 1, 1],
            ts_data=ts_data_normal,
            image_path=image_path,
            video_cfg=video_ped1_train,
            weights=weights,
        ),
        run_case(
            name="normal_signals_with_test_video_clip",
            tabular_data=[1, 1, 1, 1, 1, 1],
            ts_data=ts_data_normal,
            image_path=image_path,
            video_cfg=video_ped1_test,
            weights=weights,
        ),
    ]

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "weights": weights,
        "image_path": image_path,
        "cases": cases,
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote fusion evaluation to {out_path}")


def main():
    run_suite(
        image_path="images/normal_arm.jpg",
        out_path=Path("outputs/fusion_summary_normal_arm.json"),
    )
    run_suite(
        image_path="images/abnormal_arm.jpg",
        out_path=Path("outputs/fusion_summary_abnormal_arm.json"),
    )


if __name__ == "__main__":
    main()