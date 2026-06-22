from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from mildew_seg.validate import run_category_slice_metrics


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def val(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            results_dict={
                "metrics/precision(B)": 0.5,
                "metrics/recall(B)": 0.6,
                "metrics/mAP50(B)": 0.7,
                "metrics/mAP50-95(B)": 0.8,
                "metrics/precision(M)": 0.55,
                "metrics/recall(M)": 0.65,
                "metrics/mAP50(M)": 0.75,
                "metrics/mAP50-95(M)": 0.85,
            }
        )


class ValidateTests(unittest.TestCase):
    def test_category_slice_metrics_are_split_by_positive_negative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiles = root / "tiles.csv"
            tiles.write_text(
                "\n".join(
                    [
                        "source_image,category,split,tile_image",
                        f"{root / 'p1.png'},positive,test,{root / 'tile_p1.png'}",
                        f"{root / 'p1.png'},positive,test,{root / 'tile_p2.png'}",
                        f"{root / 'n1.png'},negative,test,{root / 'tile_n1.png'}",
                        f"{root / 'ignore.png'},positive,val,{root / 'tile_v1.png'}",
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )
            predictor = SimpleNamespace(model=_FakeModel())
            config = {
                "predict": {
                    "imgsz": 128,
                    "batch": 2,
                    "device": "cpu",
                    "max_det": 100,
                }
            }
            results = run_category_slice_metrics(
                predictor, tiles, root / "out", config, "test"
            )
            self.assertEqual(set(results), {"negative", "positive"})
            self.assertEqual(results["positive"]["tile_count"], 2)
            self.assertEqual(results["negative"]["tile_count"], 1)
            self.assertEqual(len(predictor.model.calls), 2)
            for category in ("positive", "negative"):
                data_yaml = (
                    root / "out" / "slice_metrics_by_category" / f"test_{category}.yaml"
                )
                self.assertTrue(data_yaml.is_file())
                payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
                self.assertEqual(payload["names"], {0: "mildew_spot"})
            self.assertTrue(
                (
                    root
                    / "out"
                    / "slice_metrics_by_category"
                    / "test_metrics_by_category.csv"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
