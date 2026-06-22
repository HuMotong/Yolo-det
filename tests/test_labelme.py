from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mildew_seg.labelme import load_labelme_record, normalize_label


def config() -> dict:
    return {
        "data": {
            "labels": {
                "seed": "seed",
                "mildew_spot": ["median", "meidian", "mildew_spot"],
            },
            "repair": {
                "infer_seed_min_image_ratio": 0.01,
                "infer_seed_min_area_ratio": 20.0,
            },
        }
    }


class LabelMeTests(unittest.TestCase):
    def test_label_aliases(self) -> None:
        self.assertEqual(normalize_label(" median ", config()), "mildew_spot")
        self.assertEqual(normalize_label("MEIDIAN", config()), "mildew_spot")
        self.assertEqual(normalize_label("seed", config()), "seed")
        self.assertIsNone(normalize_label("other", config()))

    def test_degenerate_polygon_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "imageWidth": 100,
                "imageHeight": 100,
                "shapes": [
                    {
                        "label": "seed",
                        "shape_type": "polygon",
                        "points": [[10, 10], [90, 10], [90, 90], [10, 90]],
                    },
                    {
                        "label": "median",
                        "shape_type": "polygon",
                        "points": [[20, 20], [21, 21]],
                    },
                ],
            }
            json_path = root / "sample.json"
            json_path.write_text(json.dumps(payload), encoding="utf-8")
            record = load_labelme_record(
                json_path, root / "sample.png", "positive", config()
            )
            self.assertEqual(len(record.mildew_spots), 0)
            self.assertEqual(
                sum(issue["code"] == "degenerate_polygon" for issue in record.issues),
                1,
            )

    def test_unique_oversized_spot_is_inferred_as_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "imageWidth": 200,
                "imageHeight": 200,
                "shapes": [
                    {
                        "label": "meidian",
                        "shape_type": "polygon",
                        "points": [[20, 20], [180, 20], [180, 180], [20, 180]],
                    },
                    {
                        "label": "meidian",
                        "shape_type": "polygon",
                        "points": [[50, 50], [52, 50], [52, 52], [50, 52]],
                    },
                ],
            }
            json_path = root / "sample.json"
            json_path.write_text(json.dumps(payload), encoding="utf-8")
            record = load_labelme_record(
                json_path, root / "sample.png", "positive", config()
            )
            self.assertTrue(record.seed_inferred)
            self.assertEqual(len(record.seed_polygons), 1)
            self.assertEqual(len(record.mildew_spots), 1)


if __name__ == "__main__":
    unittest.main()
