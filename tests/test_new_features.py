import unittest
import numpy as np
import pandas as pd

from src.anomaly_router import detect_anomaly
from src.detection_tabular_mixed import detect_mixed_tabular_for_fusion
from src.detection_tabular_single_table import detect_single_table_tabular_for_fusion
from src.detection_image_compare import compare_image_set


class TestNewFeatures(unittest.TestCase):

    # Mixed Tabular
    def test_mixed_tabular(self):
        test = pd.DataFrame({
            "age": [21, 22, 100],
            "city": ["A", "A", "A"]
        })

        result = detect_mixed_tabular_for_fusion(test)

        self.assertIn("score_norm", result)
        self.assertIn("label", result)
        self.assertTrue(0.0 <= result["score_norm"] <= 1.0)

    # Single Table Detection
    def test_single_table_detection(self):
        df = pd.DataFrame({
            "x": [1, 1, 1, 100],
            "y": [1, 1, 1, 100]
        })

        result = detect_single_table_tabular_for_fusion(df)

        self.assertIn("score_norm", result)
        self.assertIn("label", result)
        self.assertTrue(isinstance(result.get("anomalous_rows", []), list))

    # Image Comparison
    def test_image_compare(self):
        img1 = np.zeros((32, 32))
        img2 = np.zeros((32, 32))
        img3 = np.ones((32, 32))

        result = compare_image_set(
            images=[img1, img2, img3],
            names=["img1", "img2", "img3"],
            threshold=0.5
        )

        self.assertIn("ranked_items", result)
        self.assertIn("most_abnormal", result)
        self.assertEqual(len(result["items"]), 3)

    # Router mixed tabular test
    def test_router_mixed_tabular(self):
        mixed_data = [
            {"age": 21, "department": "CS", "score": 70},
            {"age": 22, "department": "Math", "score": 72},
            {"age": 20, "department": "CS", "score": 68},
            {"age": 50, "department": "Unknown", "score": 5},
        ]

        result = detect_anomaly(
            data=mixed_data,
            data_type="tabular",
            tabular_mode="mixed",
        )

        self.assertIn("score_norm", result)
        self.assertIn("label", result)
        self.assertEqual(result["modality"], "tabular")

    # Router single-table tabular test
    def test_router_single_table_tabular(self):
        single_table_data = [
            {"age": 21, "department": "CS", "score": 70},
            {"age": 22, "department": "Math", "score": 72},
            {"age": 20, "department": "CS", "score": 68},
            {"age": 50, "department": "Unknown", "score": 5},
        ]

        result = detect_anomaly(
            data=single_table_data,
            data_type="tabular",
            tabular_mode="single_table",
        )

        self.assertIn("score_norm", result)
        self.assertIn("label", result)
        self.assertEqual(result["modality"], "tabular")
        self.assertTrue(isinstance(result.get("anomalous_rows", []), list))


if __name__ == "__main__":
    unittest.main()