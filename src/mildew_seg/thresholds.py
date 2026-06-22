from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from .decision import DecisionThresholds


def load_thresholds(
    path: Path | None,
    defaults: DecisionThresholds,
) -> DecisionThresholds:
    if path is None:
        return defaults
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    values = payload.get("decision", payload)
    return DecisionThresholds(
        instance_confidence=float(
            values.get("instance_confidence", defaults.instance_confidence)
        ),
        mean_confidence=float(values.get("mean_confidence", defaults.mean_confidence)),
        count_threshold=int(values.get("count_threshold", defaults.count_threshold)),
        area_ratio_threshold=float(
            values.get("area_ratio_threshold", defaults.area_ratio_threshold)
        ),
    )


def save_thresholds(path: Path, thresholds: DecisionThresholds) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"decision": asdict(thresholds)},
            handle,
            allow_unicode=True,
            sort_keys=False,
        )
