from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.detection_tabular import detect_tabular_for_fusion
from src.detection_tabular_mixed import (
    fit_mixed_tabular_model,
    detect_mixed_tabular_for_fusion,
)
from src.detection_tabular_single_table import detect_single_table_tabular_for_fusion
from src.detection_timeseries import detect_timeseries_for_fusion


print("\n### DEMO TEST RUN ###\n")

# Numeric tabular
print("Numeric Tabular Test")
tab_data = np.array([[1, 1], [1, 1], [1, 100], [1, 1]], dtype=float)
tab_result = detect_tabular_for_fusion(tab_data)
print("Score:", tab_result["score_norm"], "Label:", tab_result["label"])
print()

# Mixed tabular
print("Mixed Tabular Test")
train_df = pd.DataFrame({
    "age": [20, 21, 22, 23],
    "city": ["A", "A", "A", "A"]
})

test_df = pd.DataFrame({
    "age": [21, 22, 100],
    "city": ["A", "A", "A"]
})

model = fit_mixed_tabular_model(train_df)
mixed_result = detect_mixed_tabular_for_fusion(model, test_df)
print("Score:", mixed_result["score_norm"], "Label:", mixed_result["label"])
print()

# Single table
print("Single Table Test")
df = pd.DataFrame({
    "x": [1, 1, 1, 100],
    "y": [1, 1, 1, 100]
})

single_result = detect_single_table_tabular_for_fusion(df)
print("Score:", single_result["score_norm"], "Label:", single_result["label"])
print("Anomalous rows:", single_result.get("anomalous_rows", []))
print()

# Time-series
print("Time-Series Test")
ts = np.array([0, 0, 0.1, 3.5, 0.1, 0], dtype=float)
ts_result = detect_timeseries_for_fusion(ts, window=3)
print("Score:", ts_result["score_norm"], "Label:", ts_result["label"])
print()

print("### ALL TABULAR AND TIME-SERIES TESTS COMPLETED ###")