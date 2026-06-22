from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np


def render_predictions(
    image: np.ndarray,
    instances: Iterable[dict],
    label: str,
    mask_alpha: float = 0.45,
) -> np.ndarray:
    instances = list(instances)
    output = image.copy()
    overlay = image.copy()
    color = (0, 0, 255) if label == "positive" else (0, 180, 0)
    for instance in instances:
        polygon = np.asarray(instance["polygon"], dtype=np.int32)
        if len(polygon) < 3:
            continue
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(output, [polygon], True, color, 1, cv2.LINE_AA)
    output = cv2.addWeighted(overlay, mask_alpha, output, 1.0 - mask_alpha, 0)
    text = f"{label.upper()} | spots={len(instances)}"
    cv2.rectangle(
        output,
        (12, 12),
        (min(output.shape[1] - 1, 500), 54),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        output,
        text,
        (22, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output
