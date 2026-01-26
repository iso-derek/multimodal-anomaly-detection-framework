from src.anomaly_router import detect_anomaly
import numpy as np

print("=== TABULAR TEST ===")
tab = np.array([[1], [1], [1], [100], [1], [1]])
print(detect_anomaly(tab, "tabular"))

print("\n=== TIMESERIES TEST ===")
ts = np.array([1]*50 + [10] + [1]*50, dtype=float)
print(detect_anomaly(ts, "timeseries"))

print("\n=== IMAGE TEST (no model) ===")
img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)  # colour image
print(detect_anomaly(img, "image", model=None))
