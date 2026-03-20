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

    def test_multicolumn_tabular_data(self):
        data = np.array([
            [1.0, 10.0],
            [1.1, 10.2],
            [0.9, 9.8],
            [1.0, 10.1],
            [20.0, 100.0],  # obvious outlier row
            [1.2, 10.0],
        ], dtype=float)

        result = detect_tabular_anomalies(data, contamination=0.1, random_state=42)

        self.assertEqual(len(result["labels"]), len(data))
        self.assertEqual(len(result["scores"]), len(data))
        self.assertIn(-1, result["labels"])

    def test_float_values_are_supported(self):
        data = np.array([
            [0.12],
            [0.15],
            [0.11],
            [0.14],
            [2.75],   # outlier
            [0.13],
        ], dtype=float)

        result = detect_tabular_anomalies(data, contamination=0.1, random_state=42)

        self.assertEqual(len(result["labels"]), len(data))
        self.assertEqual(len(result["scores"]), len(data))
        self.assertEqual(result["labels"][4], -1)

    def test_negative_and_positive_values(self):
        data = np.array([
            [-2.0],
            [-2.1],
            [-1.9],
            [-2.2],
            [15.0],   # outlier
            [-2.0],
        ], dtype=float)

        result = detect_tabular_anomalies(data, contamination=0.1, random_state=42)

        self.assertEqual(len(result["labels"]), len(data))
        self.assertEqual(result["labels"][4], -1)

    def test_1d_input_is_reshaped_correctly(self):
        data = np.array([1, 1, 1, 100, 1, 1], dtype=float)

        result = detect_tabular_anomalies(data, contamination=0.05, random_state=42)

        self.assertEqual(len(result["labels"]), len(data))
        self.assertEqual(len(result["scores"]), len(data))
        self.assertEqual(result["labels"][3], -1)

    def test_fusion_output_for_multicolumn_data(self):
        data = np.array([
            [5.0, 2.0, 8.0],
            [5.1, 2.1, 8.2],
            [4.9, 1.9, 7.8],
            [5.0, 2.0, 8.1],
            [30.0, 20.0, 50.0],  # outlier
            [5.2, 2.1, 8.0],
        ], dtype=float)

        result = detect_tabular_for_fusion(data, contamination=0.1, random_state=42)

        self.assertEqual(result["modality"], "tabular")
        self.assertIn("score_raw", result)
        self.assertIn("score_norm", result)
        self.assertIn("label", result)
        self.assertIn("method", result)
        self.assertIn("meta", result)

        self.assertGreaterEqual(result["score_norm"], 0.0)
        self.assertLessEqual(result["score_norm"], 1.0)
        self.assertIn(result["label"], [0, 1])

    def test_all_normal_like_values_do_not_crash(self):
        data = np.array([
            [10.0],
            [10.1],
            [9.9],
            [10.05],
            [10.0],
            [9.95],
        ], dtype=float)

        result = detect_tabular_anomalies(data, contamination=0.1, random_state=42)

        self.assertIn("labels", result)
        self.assertIn("scores", result)
        self.assertEqual(len(result["labels"]), len(data))
        self.assertEqual(len(result["scores"]), len(data))


if __name__ == "__main__":
    unittest.main()