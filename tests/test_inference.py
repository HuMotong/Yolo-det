from __future__ import annotations

import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from mildew_seg.decision import DecisionThresholds
from mildew_seg.geometry import build_tiles
from mildew_seg.inference import MildewPredictor
from mildew_seg.roi import locate_seed


class _FakeTensor:
    def __init__(self, values: list[float]) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class _FakeModel:
    def __init__(self, polygons_by_tile: list[list[np.ndarray]]) -> None:
        self.polygons_by_tile = polygons_by_tile

    def predict(self, **_: object) -> list[SimpleNamespace]:
        results = []
        for polygons in self.polygons_by_tile:
            results.append(
                SimpleNamespace(
                    masks=SimpleNamespace(xy=polygons) if polygons else None,
                    boxes=(
                        SimpleNamespace(conf=_FakeTensor([0.8] * len(polygons)))
                        if polygons
                        else None
                    ),
                )
            )
        return results


class InferenceTests(unittest.TestCase):
    def test_single_image_coordinate_restore_and_decision(self) -> None:
        config = {
            "tiling": {"size": 128, "overlap": 32, "roi_margin": 8},
            "roi": {
                "border_fraction": 0.05,
                "color_distance_threshold": 30,
                "morphology_kernel": 3,
                "min_seed_area": 1000,
            },
            "predict": {
                "imgsz": 256,
                "batch": 4,
                "device": "cpu",
                "conf": 0.05,
                "iou": 0.7,
                "max_det": 100,
                "retina_masks": True,
                "mask_alpha": 0.4,
            },
            "decision": {
                "instance_confidence": 0.25,
                "mean_confidence": 0.25,
                "count_threshold": 1,
                "area_ratio_threshold": 1.0,
            },
        }
        image = np.full((300, 400, 3), (240, 40, 40), dtype=np.uint8)
        cv2.ellipse(image, (200, 150), (45, 90), 0, 0, 360, (40, 180, 150), -1)
        seed = locate_seed(image, config)
        tiles = build_tiles(seed.roi, 400, 300, 128, 32)
        target = (200.0, 150.0)
        owner = next(tile for tile in tiles if tile.owns(target))
        polygons_by_tile: list[list[np.ndarray]] = [[] for _ in tiles]
        polygons_by_tile[owner.index] = [
            np.asarray(
                [
                    [target[0] - owner.x - 4, target[1] - owner.y - 4],
                    [target[0] - owner.x + 4, target[1] - owner.y - 4],
                    [target[0] - owner.x + 4, target[1] - owner.y + 4],
                    [target[0] - owner.x - 4, target[1] - owner.y + 4],
                ],
                dtype=np.float32,
            )
        ]
        predictor = MildewPredictor("unused.pt", config)
        predictor._model = _FakeModel(polygons_by_tile)
        result, visualization = predictor.predict_array(
            image,
            thresholds=DecisionThresholds(0.25, 0.25, 1, 1.0),
        )
        self.assertEqual(result.decision.label, "positive")
        self.assertEqual(result.decision.spot_count, 1)
        self.assertEqual(len(result.instances), 1)
        center = result.instances[0]["center"]
        self.assertAlmostEqual(center[0], target[0], places=3)
        self.assertAlmostEqual(center[1], target[1], places=3)
        self.assertEqual(visualization.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
