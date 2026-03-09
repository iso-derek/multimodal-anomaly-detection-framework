import unittest
from pathlib import Path

from src.detection_video import detect_video_clip_for_fusion


class TestVideoDetection(unittest.TestCase):
    def test_video_clip_for_fusion_runs_if_dataset_present(self):
        base = Path("data/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train")
        if not base.exists():
            self.skipTest("UCSD dataset not found.")

        clip_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
        if not clip_dirs:
            self.skipTest("No UCSD clip folders found.")

        result = detect_video_clip_for_fusion(clip_dirs[0], threshold=0.90)

        self.assertEqual(result["modality"], "video")
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