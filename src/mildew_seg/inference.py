from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .decision import Decision, DecisionThresholds, classify_image
from .geometry import build_tiles, polygon_centroid
from .images import read_image, write_image
from .roi import SeedRegion, locate_seed
from .utils import LOGGER, write_json
from .visualize import render_predictions


@dataclass
class PredictionResult:
    source: str
    instances: list[dict[str, Any]]
    decision: Decision
    roi: tuple[int, int, int, int]
    elapsed_seconds: float
    error: str = ""

    def to_dict(self, include_instances: bool = True) -> dict[str, Any]:
        payload = {
            "source": self.source,
            **asdict(self.decision),
            "roi": list(self.roi),
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }
        if include_instances:
            payload["instances"] = self.instances
        return payload


class MildewPredictor:
    def __init__(
        self,
        model_path: str | Path,
        config: dict[str, Any],
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.config = config
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics is not installed. Run `pip install -e .` first."
                ) from exc
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"Model checkpoint not found: {self.model_path}"
                )
            self._model = YOLO(str(self.model_path))
        return self._model

    def predict_path(
        self,
        source: str | Path,
        thresholds: DecisionThresholds | None = None,
    ) -> tuple[PredictionResult, np.ndarray]:
        path = Path(source).expanduser().resolve()
        return self.predict_array(
            read_image(path), source=str(path), thresholds=thresholds
        )

    def predict_array(
        self,
        image: np.ndarray,
        source: str = "<array>",
        thresholds: DecisionThresholds | None = None,
    ) -> tuple[PredictionResult, np.ndarray]:
        started = time.perf_counter()
        thresholds = thresholds or DecisionThresholds.from_config(self.config)
        seed = locate_seed(image, self.config)
        instances = self._predict_instances(image, seed)
        decision = summarize_instances(
            instances,
            seed.mask,
            seed.area,
            thresholds,
        )
        selected = [
            instance
            for instance in instances
            if instance["confidence"] >= thresholds.instance_confidence
        ]
        visualization = render_predictions(
            image,
            selected,
            decision.label,
            float(self.config["predict"]["mask_alpha"]),
        )
        result = PredictionResult(
            source=source,
            instances=instances,
            decision=decision,
            roi=seed.roi,
            elapsed_seconds=time.perf_counter() - started,
        )
        return result, visualization

    def _predict_instances(
        self,
        image: np.ndarray,
        seed: SeedRegion,
    ) -> list[dict[str, Any]]:
        height, width = image.shape[:2]
        tiling = self.config["tiling"]
        tiles = build_tiles(
            seed.roi,
            width,
            height,
            int(tiling["size"]),
            int(tiling["overlap"]),
        )
        crops = [
            image[tile.y : tile.y + tile.size, tile.x : tile.x + tile.size]
            for tile in tiles
        ]
        settings = self.config["predict"]
        results = self.model.predict(
            source=crops,
            imgsz=int(settings["imgsz"]),
            batch=int(settings["batch"]),
            device=settings["device"],
            conf=float(settings["conf"]),
            iou=float(settings["iou"]),
            max_det=int(settings["max_det"]),
            retina_masks=bool(settings["retina_masks"]),
            verbose=False,
        )
        instances: list[dict[str, Any]] = []
        for tile, result in zip(tiles, results, strict=True):
            if result.masks is None or result.boxes is None:
                continue
            confidences = result.boxes.conf.detach().cpu().numpy().tolist()
            polygons = result.masks.xy
            for confidence, local_polygon in zip(confidences, polygons, strict=True):
                if len(local_polygon) < 3:
                    continue
                absolute = np.asarray(local_polygon, dtype=np.float32)
                absolute[:, 0] += tile.x
                absolute[:, 1] += tile.y
                center = polygon_centroid([(float(x), float(y)) for x, y in absolute])
                if not tile.owns(center):
                    continue
                center_x = min(width - 1, max(0, int(round(center[0]))))
                center_y = min(height - 1, max(0, int(round(center[1]))))
                if seed.mask[center_y, center_x] == 0:
                    continue
                instances.append(
                    {
                        "confidence": float(confidence),
                        "polygon": absolute.tolist(),
                        "center": [float(center[0]), float(center[1])],
                        "tile_index": tile.index,
                    }
                )
        LOGGER.debug(
            "Predicted %d raw instances across %d tiles", len(instances), len(tiles)
        )
        return instances


def summarize_instances(
    instances: list[dict[str, Any]],
    seed_mask: np.ndarray,
    seed_area: int,
    thresholds: DecisionThresholds,
) -> Decision:
    selected = [
        instance
        for instance in instances
        if float(instance["confidence"]) >= thresholds.instance_confidence
    ]
    mildew_mask = np.zeros(seed_mask.shape, dtype=np.uint8)
    for instance in selected:
        polygon = np.rint(np.asarray(instance["polygon"])).astype(np.int32)
        if len(polygon) >= 3:
            cv2.fillPoly(mildew_mask, [polygon], 1)
    mildew_mask &= seed_mask.astype(np.uint8)
    mildew_area = int(np.count_nonzero(mildew_mask))
    return classify_image(
        (float(instance["confidence"]) for instance in selected),
        mildew_area,
        seed_area,
        thresholds,
    )


def save_prediction(
    result: PredictionResult,
    visualization: np.ndarray,
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{stem}_prediction.png"
    json_path = output_dir / f"{stem}_result.json"
    write_image(image_path, visualization)
    write_json(json_path, result.to_dict(include_instances=True))
    return image_path, json_path
