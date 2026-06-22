from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .decision import DecisionThresholds
from .images import read_image
from .inference import MildewPredictor, summarize_instances
from .metrics import binary_metrics
from .roi import locate_seed
from .thresholds import save_thresholds
from .utils import LOGGER, read_csv, write_csv, write_json


def run_calibration(
    predictor: MildewPredictor,
    manifest_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> DecisionThresholds:
    manifest = [row for row in read_csv(manifest_path) if row["split"] == "val"]
    if not manifest:
        raise ValueError("Validation split is empty")
    minimum_confidence = min(
        float(value) for value in config["calibration"]["instance_confidences"]
    )
    original_conf = predictor.config["predict"]["conf"]
    predictor.config["predict"]["conf"] = minimum_confidence
    raw_predictions: list[dict[str, Any]] = []
    try:
        for row in manifest:
            image = read_image(row["source_image"])
            seed = locate_seed(image, config)
            instances = predictor._predict_instances(image, seed)
            raw_predictions.append(
                {
                    "source_image": row["source_image"],
                    "category": row["category"],
                    "instances": instances,
                    "seed_mask": seed.mask,
                    "seed_area": seed.area,
                }
            )
    finally:
        predictor.config["predict"]["conf"] = original_conf

    feature_cache: dict[float, list[dict[str, Any]]] = {}
    for instance_confidence in config["calibration"]["instance_confidences"]:
        threshold = float(instance_confidence)
        features: list[dict[str, Any]] = []
        for item in raw_predictions:
            provisional = DecisionThresholds(threshold, 0.0, 0, 0.0)
            decision = summarize_instances(
                item["instances"],
                item["seed_mask"],
                item["seed_area"],
                provisional,
            )
            features.append(
                {
                    "source_image": item["source_image"],
                    "category": item["category"],
                    "spot_count": decision.spot_count,
                    "mean_confidence": decision.mean_confidence,
                    "mildew_area_ratio": decision.mildew_area_ratio,
                }
            )
        feature_cache[threshold] = features

    candidates: list[dict[str, Any]] = []
    best_thresholds: DecisionThresholds | None = None
    best_key: tuple[float, ...] | None = None
    quantile_count = int(config["calibration"]["count_quantiles"])
    area_quantile_count = int(config["calibration"]["area_quantiles"])
    for instance_confidence, features in feature_cache.items():
        counts = np.asarray([row["spot_count"] for row in features], dtype=float)
        areas = np.asarray([row["mildew_area_ratio"] for row in features], dtype=float)
        count_values = sorted(
            {
                int(round(value))
                for value in np.quantile(counts, np.linspace(0, 1, quantile_count))
            }
        )
        area_values = sorted(
            {
                float(value)
                for value in np.quantile(areas, np.linspace(0, 1, area_quantile_count))
            }
        )
        for mean_confidence in config["calibration"]["mean_confidences"]:
            for count_threshold in count_values:
                for area_threshold in area_values:
                    predictions = [
                        "positive"
                        if row["mean_confidence"] >= float(mean_confidence)
                        and (
                            row["spot_count"] >= count_threshold
                            or row["mildew_area_ratio"] >= area_threshold
                        )
                        else "negative"
                        for row in features
                    ]
                    metrics = binary_metrics(
                        (row["category"] for row in features), predictions
                    )
                    complexity = (
                        float(mean_confidence)
                        + count_threshold / max(1.0, float(counts.max()))
                        + area_threshold / max(1e-12, float(areas.max()))
                    )
                    candidate = {
                        "instance_confidence": instance_confidence,
                        "mean_confidence": float(mean_confidence),
                        "count_threshold": count_threshold,
                        "area_ratio_threshold": area_threshold,
                        "complexity": complexity,
                        **metrics,
                    }
                    candidates.append(candidate)
                    key = (
                        float(metrics["f1"]),
                        float(metrics["recall"]),
                        float(metrics["accuracy"]),
                        -complexity,
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_thresholds = DecisionThresholds(
                            instance_confidence,
                            float(mean_confidence),
                            count_threshold,
                            area_threshold,
                        )

    if best_thresholds is None:
        raise RuntimeError("Calibration did not produce threshold candidates")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_thresholds(output_dir / "thresholds.yaml", best_thresholds)
    sorted_candidates = sorted(
        candidates,
        key=lambda row: (
            -float(row["f1"]),
            -float(row["recall"]),
            -float(row["accuracy"]),
            float(row["complexity"]),
        ),
    )
    write_csv(output_dir / "search_results.csv", sorted_candidates)
    best_features = feature_cache[best_thresholds.instance_confidence]
    validation_rows = []
    for row in best_features:
        predicted = (
            "positive"
            if row["mean_confidence"] >= best_thresholds.mean_confidence
            and (
                row["spot_count"] >= best_thresholds.count_threshold
                or row["mildew_area_ratio"] >= best_thresholds.area_ratio_threshold
            )
            else "negative"
        )
        validation_rows.append({**row, "predicted": predicted})
    write_csv(output_dir / "validation_predictions.csv", validation_rows)
    write_json(
        output_dir / "calibration_summary.json",
        {
            "thresholds": asdict(best_thresholds),
            "best_metrics": sorted_candidates[0],
            "validation_images": len(raw_predictions),
        },
    )
    LOGGER.info("Calibrated thresholds: %s", best_thresholds)
    return best_thresholds
