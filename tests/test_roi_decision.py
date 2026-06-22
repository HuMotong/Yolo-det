from __future__ import annotations

import unittest

import cv2
import numpy as np

from mildew_seg.decision import DecisionThresholds, classify_image
from mildew_seg.roi import SeedLocalizationError, locate_seed


class RoiAndDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "roi": {
                "border_fraction": 0.05,
                "color_distance_threshold": 30,
                "morphology_kernel": 3,
                "min_seed_area": 1000,
            },
            "tiling": {"roi_margin": 8},
        }

    def test_locate_seed_on_blue_background(self) -> None:
        image = np.full((300, 400, 3), (240, 40, 40), dtype=np.uint8)
        cv2.ellipse(image, (200, 150), (45, 90), 0, 0, 360, (40, 180, 150), -1)
        region = locate_seed(image, self.config)
        self.assertGreater(region.area, 10000)
        left, top, right, bottom = region.roi
        self.assertLess(left, 155)
        self.assertGreater(right, 245)
        self.assertLess(top, 60)
        self.assertGreater(bottom, 240)

    def test_locate_seed_fails_without_foreground(self) -> None:
        image = np.full((100, 100, 3), (240, 40, 40), dtype=np.uint8)
        with self.assertRaises(SeedLocalizationError):
            locate_seed(image, self.config)

    def test_decision_rule(self) -> None:
        thresholds = DecisionThresholds(0.25, 0.30, 3, 0.10)
        positive = classify_image([0.4, 0.5, 0.6], 5, 100, thresholds)
        self.assertEqual(positive.label, "positive")
        area_positive = classify_image([0.8], 20, 100, thresholds)
        self.assertEqual(area_positive.label, "positive")
        negative = classify_image([0.1, 0.2], 20, 100, thresholds)
        self.assertEqual(negative.label, "negative")


if __name__ == "__main__":
    unittest.main()
