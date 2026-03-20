from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# Fix import path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.fusion_evaluate import fuse_weighted_average


print("\n### FUSION TEST ###\n")

# Simulated modality outputs
results = {
    "tabular": {"score_norm": 0.7, "label": 1},
    "timeseries": {"score_norm": 0.2, "label": 0},
    "image": {"score_norm": 0.3, "label": 0},
    "video": {"score_norm": 0.95, "label": 1},  # strong anomaly
}

weights = {
    "tabular": 1.5,
    "timeseries": 1.0,
    "image": 0.8,
    "video": 2.0,
}

fused = fuse_weighted_average(
    results=results,
    weights=weights,
    strong_threshold=0.8,
    strong_threshold_video=0.95,
    fused_threshold=0.65,
    vote_k=2,
)

print("Final Score:", fused["final_score"])
print("Final Label:", fused["final_label"])
print("Reason:", fused["reason"])

print("\n### FUSION TEST COMPLETE ###")