from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from mildew_seg.images import write_image
from mildew_seg.prepare import prepare_dataset
from mildew_seg.utils import read_csv


class PrepareIntegrationTests(unittest.TestCase):
    def test_prepare_synthetic_labelme_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "mildew"
            for category in ("positive", "negative"):
                (raw / category).mkdir(parents=True)
                for index in range(4):
                    self._write_sample(raw / category, category, index)
            config = self._config(root)
            output = root / "runs" / "data"
            summary = prepare_dataset(config, output)
            self.assertEqual(summary["original_count"], 8)
            manifest = read_csv(output / "manifest.csv")
            self.assertEqual(len({row["source_id"] for row in manifest}), 8)
            tiles = read_csv(output / "tiles.csv")
            self.assertGreater(len(tiles), 8)
            label_paths = [Path(row["tile_label"]) for row in tiles]
            for label_path in label_paths:
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    self.assertTrue(line.startswith("0 "))
            with (output / "data.yaml").open("r", encoding="utf-8") as handle:
                data_yaml = yaml.safe_load(handle)
            self.assertEqual(data_yaml["names"], {0: "mildew_spot"})

    def _write_sample(self, directory: Path, category: str, index: int) -> None:
        image = np.full((400, 400, 3), (240, 40, 40), dtype=np.uint8)
        image[80:330, 150:250] = (50, 170, 140)
        stem = f"{category}_{index}"
        write_image(directory / f"{stem}.png", image)
        payload = {
            "version": "5.0",
            "imagePath": f"{stem}.png",
            "imageWidth": 400,
            "imageHeight": 400,
            "shapes": [
                {
                    "label": "seed",
                    "shape_type": "polygon",
                    "points": [[150, 80], [250, 80], [250, 330], [150, 330]],
                },
                {
                    "label": "meidian" if category == "positive" else "median",
                    "shape_type": "polygon",
                    "points": [
                        [180 + index, 120],
                        [188 + index, 120],
                        [188 + index, 128],
                        [180 + index, 128],
                    ],
                },
            ],
        }
        (directory / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")

    def _config(self, root: Path) -> dict:
        return {
            "_project_root": str(root),
            "paths": {
                "raw_data": "mildew",
                "runs": "runs",
                "checkpoint": "checkpoints/yolo11n-seg.pt",
            },
            "data": {
                "image_extensions": [".png"],
                "positive_dir": "positive",
                "negative_dir": "negative",
                "labels": {
                    "seed": "seed",
                    "mildew_spot": ["median", "meidian", "mildew_spot"],
                },
                "split": {
                    "train": 0.70,
                    "val": 0.15,
                    "test": 0.15,
                    "seed": 42,
                },
                "repair": {
                    "infer_seed_min_image_ratio": 0.01,
                    "infer_seed_min_area_ratio": 20,
                },
            },
            "tiling": {"size": 128, "overlap": 32, "roi_margin": 8},
            "roi": {},
            "train": {},
            "predict": {},
            "decision": {},
        }


if __name__ == "__main__":
    unittest.main()
