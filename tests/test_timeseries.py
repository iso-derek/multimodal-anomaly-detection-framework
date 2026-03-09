import unittest
import numpy as np

from src.detection_timeseries import (
    detect_timeseries_anomalies,
    detect_timeseries_for_fusion,
)


class TestTimeseriesDetection(unittest.TestCase):
    def test_rolling_zscore_detects_spike(self):
        series = np.array([1] * 50 + [10] + [1] * 50, dtype=float)
        result = detect_timeseries_anomalies(series, window=25, z_thresh=3.0)

        self.assertIn("anomaly_indices", result)
        self.assertIn("scores", result)
        self.assertIn("method", result)
        self.assertIn(50, result["anomaly_indices"])

    def test_timeseries_fusion_output(self):
        series = np.array([1] * 50 + [10] + [1] * 50, dtype=float)
        result = detect_timeseries_for_fusion(
            series, window=25, z_thresh=3.0, threshold=0.65
        )

        self.assertEqual(result["modality"], "timeseries")
        self.assertIn("score_raw", result)
        self.assertIn("score_norm", result)
        self.assertIn("label", result)
        self.assertIn("meta", result)

        self.assertGreaterEqual(result["score_norm"], 0.0)
        self.assertLessEqual(result["score_norm"], 1.0)
        self.assertIn(result["label"], [0, 1])

        # Strong spike should usually produce anomaly in fusion-ready form
        self.assertEqual(result["label"], 1)


if __name__ == "__main__":
    unittest.main()