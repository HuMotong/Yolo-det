from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

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
    category_slice_results = run_category_slice_metrics(
        predictor,
        manifest_path.with_name("tiles.csv"),
        output_dir,
        config,
        split,
    )

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
        "slice_metrics_by_category": category_slice_results,
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


def run_category_slice_metrics(
    predictor: MildewPredictor,
    tiles_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    split: str,
) -> dict[str, dict[str, Any]]:
    if not tiles_path.is_file():
        LOGGER.warning("Cannot compute category slice metrics; missing %s", tiles_path)
        return {}

    settings = config["predict"]
    tiles = [row for row in read_csv(tiles_path) if row["split"] == split]
    grouped: dict[str, list[dict[str, str]]] = {"negative": [], "positive": []}
    for row in tiles:
        if row["category"] in grouped:
            grouped[row["category"]].append(row)

    category_dir = output_dir / "slice_metrics_by_category"
    category_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for category, category_tiles in grouped.items():
        if not category_tiles:
            continue
        data_yaml = _write_category_data_yaml(
            category_dir, category, split, category_tiles
        )
        metrics = predictor.model.val(
            data=str(data_yaml),
            split=split,
            project=str(category_dir),
            name=f"{split}_{category}",
            imgsz=int(settings["imgsz"]),
            batch=int(settings["batch"]),
            device=settings["device"],
            max_det=int(settings["max_det"]),
            plots=True,
            verbose=False,
        )
        metrics_dict = {
            key: float(value)
            for key, value in getattr(metrics, "results_dict", {}).items()
        }
        source_count = len({row["source_image"] for row in category_tiles})
        payload = {
            "category": category,
            "split": split,
            "tile_count": len(category_tiles),
            "source_image_count": source_count,
            "metrics": metrics_dict,
        }
        results[category] = payload
        write_json(category_dir / f"{split}_{category}_metrics.json", payload)
        rows.extend(
            {
                "category": category,
                "split": split,
                "tile_count": len(category_tiles),
                "source_image_count": source_count,
                "metric": metric,
                "value": value,
            }
            for metric, value in metrics_dict.items()
        )

    write_json(category_dir / f"{split}_metrics_by_category.json", results)
    write_csv(category_dir / f"{split}_metrics_by_category.csv", rows)
    return results


def _write_category_data_yaml(
    output_dir: Path,
    category: str,
    split: str,
    tiles: list[dict[str, str]],
) -> Path:
    image_list = output_dir / f"{split}_{category}_images.txt"
    image_list.write_text(
        "\n".join(row["tile_image"] for row in tiles) + "\n",
        encoding="utf-8",
    )
    data_yaml = output_dir / f"{split}_{category}.yaml"
    payload = {
        "path": str(output_dir),
        "train": str(image_list),
        "val": str(image_list),
        "test": str(image_list),
        "names": {0: "mildew_spot"},
    }
    with data_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    return data_yaml
