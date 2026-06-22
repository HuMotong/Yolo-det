from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .geometry import Polygon, polygon_area


@dataclass
class Annotation:
    label: str
    points: Polygon
    source_index: int
    area: float


@dataclass
class LabelMeRecord:
    json_path: Path
    image_path: Path
    image_width: int
    image_height: int
    category: str
    mildew_spots: list[Annotation] = field(default_factory=list)
    seed_polygons: list[Annotation] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    seed_inferred: bool = False

    @property
    def image_area(self) -> int:
        return self.image_width * self.image_height

    @property
    def seed_area(self) -> float:
        return sum(annotation.area for annotation in self.seed_polygons)

    @property
    def mildew_area(self) -> float:
        return sum(annotation.area for annotation in self.mildew_spots)


def normalize_label(label: str, config: dict[str, Any]) -> str | None:
    normalized = label.strip().lower()
    seed_label = str(config["data"]["labels"]["seed"]).strip().lower()
    spot_labels = {
        str(value).strip().lower() for value in config["data"]["labels"]["mildew_spot"]
    }
    if normalized == seed_label:
        return "seed"
    if normalized in spot_labels:
        return "mildew_spot"
    return None


def load_labelme_record(
    json_path: Path,
    image_path: Path,
    category: str,
    config: dict[str, Any],
    allow_repair: bool = True,
) -> LabelMeRecord:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    record = LabelMeRecord(
        json_path=json_path,
        image_path=image_path,
        image_width=int(payload.get("imageWidth") or 0),
        image_height=int(payload.get("imageHeight") or 0),
        category=category,
    )
    if record.image_width <= 0 or record.image_height <= 0:
        record.issues.append(
            {
                "code": "invalid_dimensions",
                "severity": "error",
                "message": "Invalid size",
            }
        )

    unknown: list[Annotation] = []
    for index, shape in enumerate(payload.get("shapes") or []):
        label = str(shape.get("label") or "")
        shape_type = str(shape.get("shape_type") or "polygon")
        raw_points = shape.get("points") or []
        points = [(float(point[0]), float(point[1])) for point in raw_points]
        area = polygon_area(points)
        normalized = normalize_label(label, config)
        issue_base = {"shape_index": index, "source_label": label}

        if shape_type != "polygon":
            record.issues.append(
                {
                    **issue_base,
                    "code": "unsupported_shape_type",
                    "severity": "warning",
                    "message": f"Skipped shape type {shape_type}",
                }
            )
            continue
        if len(points) < 3 or area <= 0:
            record.issues.append(
                {
                    **issue_base,
                    "code": "degenerate_polygon",
                    "severity": "warning",
                    "point_count": len(points),
                    "area": area,
                    "message": "Skipped polygon with fewer than 3 points or zero area",
                }
            )
            continue
        if any(
            x < 0 or y < 0 or x > record.image_width or y > record.image_height
            for x, y in points
        ):
            record.issues.append(
                {
                    **issue_base,
                    "code": "out_of_bounds",
                    "severity": "error",
                    "message": "Polygon coordinates exceed image bounds",
                }
            )
            continue

        annotation = Annotation(label, points, index, area)
        if normalized == "seed":
            record.seed_polygons.append(annotation)
        elif normalized == "mildew_spot":
            record.mildew_spots.append(annotation)
        else:
            unknown.append(annotation)
            record.issues.append(
                {
                    **issue_base,
                    "code": "unknown_label",
                    "severity": "warning",
                    "message": f"Skipped unknown label: {label}",
                }
            )

    if not record.seed_polygons and allow_repair:
        _infer_seed_polygon(record, config)
    if not record.seed_polygons:
        record.issues.append(
            {
                "code": "missing_seed",
                "severity": "error",
                "message": "No seed polygon found or inferred",
            }
        )
    return record


def _infer_seed_polygon(record: LabelMeRecord, config: dict[str, Any]) -> None:
    if not record.mildew_spots:
        return
    ordered = sorted(record.mildew_spots, key=lambda item: item.area, reverse=True)
    largest = ordered[0]
    second_area = ordered[1].area if len(ordered) > 1 else 0.0
    repair = config["data"]["repair"]
    min_image_area = float(repair["infer_seed_min_image_ratio"]) * record.image_area
    min_ratio = float(repair["infer_seed_min_area_ratio"])
    ratio = float("inf") if second_area <= 0 else largest.area / second_area
    if largest.area < min_image_area or ratio < min_ratio:
        return
    record.mildew_spots.remove(largest)
    record.seed_polygons.append(
        Annotation("seed_inferred", largest.points, largest.source_index, largest.area)
    )
    record.seed_inferred = True
    record.issues.append(
        {
            "code": "seed_inferred",
            "severity": "warning",
            "shape_index": largest.source_index,
            "area": largest.area,
            "next_largest_ratio": ratio,
            "message": "Inferred seed from the unique oversized polygon",
        }
    )
