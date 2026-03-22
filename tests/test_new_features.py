import unittest
import numpy as np
import pandas as pd

from src.detection_tabular_mixed import (
    fit_mixed_tabular_model,
    detect_mixed_tabular_for_fusion,
)
from src.detection_tabular_single_table import (
    detect_single_table_tabular_for_fusion,
)
from src.detection_image_compare import compare_image_set


class TestNewFeatures(unittest.TestCase):

    # Mixed Tabular
    def test_mixed_tabular(self):
        train = pd.DataFrame({
            "age": [20, 21, 22, 23],
            "city": ["A", "A", "A", "A"]
        })

        test = pd.DataFrame({
            "age": [21, 22, 100],
            "city": ["A", "A", "A"]
        })

        model = fit_mixed_tabular_model(train)
        result = detect_mixed_tabular_for_fusion(model, test)

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


if __name__ == "__main__":
    unittest.main()