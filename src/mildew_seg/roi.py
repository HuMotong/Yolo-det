from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .geometry import expand_bbox


class SeedLocalizationError(RuntimeError):
    """Raised when a seed cannot be isolated from the configured background."""


@dataclass
class SeedRegion:
    mask: np.ndarray
    roi: tuple[int, int, int, int]
    area: int
    background_bgr: tuple[float, float, float]


def locate_seed(image: np.ndarray, config: dict[str, Any]) -> SeedRegion:
    if image.ndim != 3 or image.shape[2] != 3:
        raise SeedLocalizationError("Expected a BGR color image")
    height, width = image.shape[:2]
    settings = config["roi"]
    border = max(1, round(min(height, width) * float(settings["border_fraction"])))
    samples = np.concatenate(
        (
            image[:border].reshape(-1, 3),
            image[-border:].reshape(-1, 3),
            image[:, :border].reshape(-1, 3),
            image[:, -border:].reshape(-1, 3),
        ),
        axis=0,
    )
    background = np.median(samples.astype(np.float32), axis=0)
    distance = np.linalg.norm(image.astype(np.float32) - background, axis=2)
    foreground = (distance >= float(settings["color_distance_threshold"])).astype(
        np.uint8
    )
    kernel_size = max(3, int(settings["morphology_kernel"]))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )
    if count <= 1:
        raise SeedLocalizationError("No foreground component found")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component, cv2.CC_STAT_AREA])
    if area < int(settings["min_seed_area"]):
        raise SeedLocalizationError(
            f"Largest foreground area {area} is below roi.min_seed_area"
        )
    mask = (labels == component).astype(np.uint8)
    x = int(stats[component, cv2.CC_STAT_LEFT])
    y = int(stats[component, cv2.CC_STAT_TOP])
    w = int(stats[component, cv2.CC_STAT_WIDTH])
    h = int(stats[component, cv2.CC_STAT_HEIGHT])
    roi = expand_bbox(
        (x, y, x + w - 1, y + h - 1),
        int(config["tiling"]["roi_margin"]),
        width,
        height,
    )
    return SeedRegion(
        mask=mask,
        roi=roi,
        area=area,
        background_bgr=tuple(float(value) for value in background),
    )
