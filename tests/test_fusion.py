import unittest

from src.fusion import ModalitySummary, fuse_weighted_average


class TestFusion(unittest.TestCase):
    def test_strong_modality_override(self):
        summaries = [
            ModalitySummary(modality="tabular", score=1.0, raw_score=1.0, label=-1, meta={}),
            ModalitySummary(modality="timeseries", score=0.1, raw_score=0.1, label=1, meta={}),
            ModalitySummary(modality="image", score=0.2, raw_score=0.2, label=1, meta={}),
        ]

        result = fuse_weighted_average(
            summaries,
            weights={"tabular": 1.5, "timeseries": 1.0, "image": 0.8},
            strong_threshold=0.8,
            avg_threshold=0.6,
        )

        self.assertEqual(result["final_label"], -1)
        self.assertIn("strong modality", result["reason"])

    def test_weighted_average_anomaly_without_strong_override(self):
        summaries = [
            ModalitySummary(modality="tabular", score=0.7, raw_score=0.7, label=-1, meta={}),
            ModalitySummary(modality="timeseries", score=0.65, raw_score=0.65, label=-1, meta={}),
            ModalitySummary(modality="image", score=0.4, raw_score=0.4, label=1, meta={}),
        ]

        result = fuse_weighted_average(
            summaries,
            weights={"tabular": 1.5, "timeseries": 1.0, "image": 0.8},
            strong_threshold=0.8,
            avg_threshold=0.6,
        )

        self.assertEqual(result["final_label"], -1)
        self.assertIn("weighted average", result["reason"])

    def test_weighted_average_normal_case(self):
        summaries = [
            ModalitySummary(modality="tabular", score=0.1, raw_score=0.1, label=1, meta={}),
            ModalitySummary(modality="timeseries", score=0.2, raw_score=0.2, label=1, meta={}),
            ModalitySummary(modality="image", score=0.3, raw_score=0.3, label=1, meta={}),
        ]

        result = fuse_weighted_average(
            summaries,
            weights={"tabular": 1.5, "timeseries": 1.0, "image": 0.8},
            strong_threshold=0.8,
            avg_threshold=0.6,
        )

        self.assertEqual(result["final_label"], 1)
        self.assertIn("no strong modality", result["reason"])


if __name__ == "__main__":
    unittest.main()