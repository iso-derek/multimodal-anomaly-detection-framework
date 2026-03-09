import unittest
import numpy as np
from pathlib import Path
from skimage.io import imread

from src.detection_image import detect_image_for_fusion


class TestImageDetection(unittest.TestCase):
    def test_image_fusion_output_random_image(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = detect_image_for_fusion(model=None, image=img)

        self.assertEqual(result["modality"], "image")
        self.assertIn("score_raw", result)
        self.assertIn("score_norm", result)
        self.assertIn("label", result)
        self.assertIn("method", result)
        self.assertIn("meta", result)

        self.assertGreaterEqual(result["score_norm"], 0.0)
        self.assertLessEqual(result["score_norm"], 1.0)
        self.assertIn(result["label"], [0, 1])

    def test_real_arm_images_run_if_present(self):
        normal_path = Path("images/normal_arm.jpg")
        abnormal_path = Path("images/abnormal_arm.jpg")

        if not (normal_path.exists() and abnormal_path.exists()):
            self.skipTest("Arm images not found.")

        normal_img = imread(str(normal_path))
        abnormal_img = imread(str(abnormal_path))

        normal_result = detect_image_for_fusion(model=None, image=normal_img)
        abnormal_result = detect_image_for_fusion(model=None, image=abnormal_img)

        self.assertEqual(normal_result["modality"], "image")
        self.assertEqual(abnormal_result["modality"], "image")
        self.assertGreaterEqual(normal_result["score_norm"], 0.0)
        self.assertLessEqual(normal_result["score_norm"], 1.0)
        self.assertGreaterEqual(abnormal_result["score_norm"], 0.0)
        self.assertLessEqual(abnormal_result["score_norm"], 1.0)


if __name__ == "__main__":
    unittest.main()