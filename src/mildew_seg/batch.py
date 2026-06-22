from __future__ import annotations

from pathlib import Path
from typing import Any

from .decision import DecisionThresholds
from .inference import MildewPredictor, save_prediction
from .utils import LOGGER, image_files, safe_name, write_csv


def run_batch(
    predictor: MildewPredictor,
    source: Path,
    output_dir: Path,
    config: dict[str, Any],
    thresholds: DecisionThresholds,
    save_images: bool = True,
) -> Path:
    files = image_files(source, config["data"]["image_extensions"])
    if not files:
        raise ValueError(f"No supported images found: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    rows: list[dict[str, Any]] = []
    for path in files:
        try:
            result, visualization = predictor.predict_path(path, thresholds)
            row = result.to_dict(include_instances=False)
            if save_images:
                save_prediction(
                    result,
                    visualization,
                    image_dir,
                    safe_name(path.stem),
                )
        except Exception as exc:
            LOGGER.exception("Prediction failed for %s", path)
            row = {
                "source": str(path.resolve()),
                "label": "error",
                "spot_count": 0,
                "max_confidence": 0.0,
                "mean_confidence": 0.0,
                "mildew_area": 0,
                "seed_area": 0,
                "mildew_area_ratio": 0.0,
                "elapsed_seconds": 0.0,
                "error": repr(exc),
            }
        rows.append(row)
    csv_path = output_dir / "results.csv"
    write_csv(csv_path, rows)
    LOGGER.info("Batch prediction wrote %d rows to %s", len(rows), csv_path)
    return csv_path
