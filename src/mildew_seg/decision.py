from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecisionThresholds:
    instance_confidence: float
    mean_confidence: float
    count_threshold: int
    area_ratio_threshold: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DecisionThresholds:
        values = config["decision"]
        return cls(
            instance_confidence=float(values["instance_confidence"]),
            mean_confidence=float(values["mean_confidence"]),
            count_threshold=int(values["count_threshold"]),
            area_ratio_threshold=float(values["area_ratio_threshold"]),
        )


@dataclass(frozen=True)
class Decision:
    label: str
    spot_count: int
    max_confidence: float
    mean_confidence: float
    mildew_area: int
    seed_area: int
    mildew_area_ratio: float


def classify_image(
    confidences: Iterable[float],
    mildew_area: int,
    seed_area: int,
    thresholds: DecisionThresholds,
) -> Decision:
    selected = [
        float(value)
        for value in confidences
        if float(value) >= thresholds.instance_confidence
    ]
    count = len(selected)
    maximum = max(selected, default=0.0)
    mean = sum(selected) / count if count else 0.0
    area_ratio = mildew_area / seed_area if seed_area > 0 else 0.0
    positive = mean >= thresholds.mean_confidence and (
        count >= thresholds.count_threshold
        or area_ratio >= thresholds.area_ratio_threshold
    )
    return Decision(
        label="positive" if positive else "negative",
        spot_count=count,
        max_confidence=maximum,
        mean_confidence=mean,
        mildew_area=int(mildew_area),
        seed_area=int(seed_area),
        mildew_area_ratio=float(area_ratio),
    )
