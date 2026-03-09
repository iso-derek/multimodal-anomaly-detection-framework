import unittest
import numpy as np

from src.detection_tabular import detect_tabular_anomalies, detect_tabular_for_fusion


class TestTabularDetection(unittest.TestCase):
    def test_isolation_forest_detects_obvious_outlier(self):
        data = np.array([[1], [1], [1], [100], [1], [1]], dtype=float)
        result = detect_tabular_anomalies(data, contamination=0.05, random_state=42)

        self.assertIn("labels", result)
        self.assertIn("scores", result)
        self.assertIn("n_anomalies", result)
        self.assertEqual(len(result["labels"]), len(data))
        self.assertEqual(len(result["scores"]), len(data))
        self.assertGreaterEqual(result["n_anomalies"], 1)

        # The outlier at index 3 should be flagged
        self.assertEqual(result["labels"][3], -1)

    def test_tabular_fusion_output_shape(self):
        data = np.array([[1], [1], [1], [100], [1], [1]], dtype=float)
        result = detect_tabular_for_fusion(data, contamination=0.05, random_state=42)

        self.assertEqual(result["modality"], "tabular")
        self.assertIn("score_raw", result)
        self.assertIn("score_norm", result)
        self.assertIn("label", result)
        self.assertIn("method", result)
        self.assertIn("meta", result)

        self.assertGreaterEqual(result["score_norm"], 0.0)
        self.assertLessEqual(result["score_norm"], 1.0)
        self.assertIn(result["label"], [0, 1])


if __name__ == "__main__":
    unittest.main()