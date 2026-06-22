from __future__ import annotations

from pathlib import Path
from typing import Any

from .decision import DecisionThresholds
from .inference import MildewPredictor, save_prediction
from .metrics import binary_metrics
from .utils import LOGGER, read_csv, safe_name, write_csv, write_json


def run_validation(
    predictor: MildewPredictor,
    data_yaml: Path,
    manifest_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    thresholds: DecisionThresholds,
    split: str = "test",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = config["predict"]
    slice_metrics = predictor.model.val(
        data=str(data_yaml),
        split=split,
        project=str(output_dir),
        name="slice_metrics",
        imgsz=int(settings["imgsz"]),
        batch=int(settings["batch"]),
        device=settings["device"],
        max_det=int(settings["max_det"]),
        plots=True,
        verbose=False,
    )
    slice_results = {
        key: float(value)
        for key, value in getattr(slice_metrics, "results_dict", {}).items()
    }
    write_json(output_dir / "slice_metrics.json", slice_results)

    manifest = [row for row in read_csv(manifest_path) if row["split"] == split]
    if not manifest:
        raise ValueError(f"Manifest split is empty: {split}")
    rows: list[dict[str, Any]] = []
    image_dir = output_dir / "images"
    for source in manifest:
        result, visualization = predictor.predict_path(
            source["source_image"], thresholds
        )
        save_prediction(
            result,
            visualization,
            image_dir,
            safe_name(Path(source["source_image"]).stem),
        )
        rows.append(
            {
                **result.to_dict(include_instances=False),
                "actual": source["category"],
                "gt_spot_count": int(source["gt_spot_count"]),
                "gt_mildew_area_ratio": float(source["gt_mildew_area_ratio"]),
                "count_absolute_error": abs(
                    result.decision.spot_count - int(source["gt_spot_count"])
                ),
                "area_ratio_absolute_error": abs(
                    result.decision.mildew_area_ratio
                    - float(source["gt_mildew_area_ratio"])
                ),
            }
        )
    write_csv(output_dir / "original_predictions.csv", rows)
    classification = binary_metrics(
        (row["actual"] for row in rows),
        (row["label"] for row in rows),
    )
    summary = {
        "split": split,
        "image_count": len(rows),
        "slice_metrics": slice_results,
        "classification": classification,
        "count_mae": sum(row["count_absolute_error"] for row in rows) / len(rows),
        "area_ratio_mae": (
            sum(row["area_ratio_absolute_error"] for row in rows) / len(rows)
        ),
    }
    write_json(output_dir / "validation_summary.json", summary)
    confusion = [
        {
            "actual": actual,
            "predicted": predicted,
            "count": sum(
                row["actual"] == actual and row["label"] == predicted for row in rows
            ),
        }
        for actual in ("negative", "positive")
        for predicted in ("negative", "positive")
    ]
    write_csv(
        output_dir / "confusion_matrix.csv",
        confusion,
    )
    LOGGER.info("Validation summary: %s", summary)
    return summary
