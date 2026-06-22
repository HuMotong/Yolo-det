from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

Point = tuple[float, float]
Polygon = list[Point]


@dataclass(frozen=True)
class Tile:
    index: int
    x: int
    y: int
    size: int
    core_left: float
    core_top: float
    core_right: float
    core_bottom: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.size, self.y + self.size

    def owns(self, point: Point) -> bool:
        x, y = point
        return (
            self.core_left <= x < self.core_right
            and self.core_top <= y < self.core_bottom
        )


def polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    values = np.asarray(points, dtype=np.float64)
    return abs(float(cv2.contourArea(values.astype(np.float32))))


def polygon_centroid(points: Sequence[Point]) -> Point:
    values = np.asarray(points, dtype=np.float64)
    if len(values) == 0:
        return 0.0, 0.0
    moments = cv2.moments(values.astype(np.float32))
    if abs(moments["m00"]) > 1e-9:
        return (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )
    return float(values[:, 0].mean()), float(values[:, 1].mean())


def polygon_bbox(points: Sequence[Point]) -> tuple[float, float, float, float]:
    values = np.asarray(points, dtype=np.float64)
    return (
        float(values[:, 0].min()),
        float(values[:, 1].min()),
        float(values[:, 0].max()),
        float(values[:, 1].max()),
    )


def combined_bbox(
    polygons: Iterable[Sequence[Point]],
) -> tuple[float, float, float, float]:
    values = [np.asarray(points, dtype=np.float64) for points in polygons if points]
    if not values:
        raise ValueError("Cannot compute a bounding box without polygons")
    merged = np.concatenate(values, axis=0)
    return (
        float(merged[:, 0].min()),
        float(merged[:, 1].min()),
        float(merged[:, 0].max()),
        float(merged[:, 1].max()),
    )


def expand_bbox(
    box: tuple[float, float, float, float],
    margin: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    return (
        max(0, int(np.floor(left)) - margin),
        max(0, int(np.floor(top)) - margin),
        min(width, int(np.ceil(right)) + margin + 1),
        min(height, int(np.ceil(bottom)) + margin + 1),
    )


def _axis_starts(
    start: int, end: int, limit: int, size: int, overlap: int
) -> list[int]:
    if limit <= size:
        return [0]
    start = max(0, min(start, limit - size))
    last = max(0, min(max(start, end - size), limit - size))
    stride = size - overlap
    values = [start]
    while values[-1] < last:
        candidate = min(values[-1] + stride, last)
        if candidate == values[-1]:
            break
        values.append(candidate)
    return values


def build_tiles(
    roi: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    size: int,
    overlap: int,
) -> list[Tile]:
    left, top, right, bottom = roi
    xs = _axis_starts(left, right, image_width, size, overlap)
    ys = _axis_starts(top, bottom, image_height, size, overlap)

    x_edges = _core_edges(xs, size, image_width)
    y_edges = _core_edges(ys, size, image_height)
    tiles: list[Tile] = []
    index = 0
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            tiles.append(
                Tile(
                    index=index,
                    x=x,
                    y=y,
                    size=size,
                    core_left=x_edges[xi],
                    core_top=y_edges[yi],
                    core_right=x_edges[xi + 1],
                    core_bottom=y_edges[yi + 1],
                )
            )
            index += 1
    return tiles


def _core_edges(starts: list[int], size: int, limit: int) -> list[float]:
    edges = [0.0]
    for previous, current in zip(starts, starts[1:], strict=False):
        edges.append((previous + size + current) / 2.0)
    edges.append(float(limit))
    return edges


def clip_polygon_to_rect(
    points: Sequence[Point],
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> Polygon:
    polygon = list(points)
    for inside, intersect in (
        (
            lambda p: p[0] >= left,
            lambda a, b: _intersect_vertical(a, b, left),
        ),
        (
            lambda p: p[0] <= right,
            lambda a, b: _intersect_vertical(a, b, right),
        ),
        (
            lambda p: p[1] >= top,
            lambda a, b: _intersect_horizontal(a, b, top),
        ),
        (
            lambda p: p[1] <= bottom,
            lambda a, b: _intersect_horizontal(a, b, bottom),
        ),
    ):
        polygon = _clip_edge(polygon, inside, intersect)
        if not polygon:
            break
    return polygon


def _clip_edge(
    polygon: Polygon,
    inside: object,
    intersect: object,
) -> Polygon:
    if not polygon:
        return []
    output: Polygon = []
    previous = polygon[-1]
    previous_inside = inside(previous)  # type: ignore[operator]
    for current in polygon:
        current_inside = inside(current)  # type: ignore[operator]
        if current_inside:
            if not previous_inside:
                output.append(intersect(previous, current))  # type: ignore[operator]
            output.append(current)
        elif previous_inside:
            output.append(intersect(previous, current))  # type: ignore[operator]
        previous = current
        previous_inside = current_inside
    return output


def _intersect_vertical(a: Point, b: Point, x: float) -> Point:
    if abs(b[0] - a[0]) < 1e-12:
        return x, a[1]
    ratio = (x - a[0]) / (b[0] - a[0])
    return x, a[1] + ratio * (b[1] - a[1])


def _intersect_horizontal(a: Point, b: Point, y: float) -> Point:
    if abs(b[1] - a[1]) < 1e-12:
        return a[0], y
    ratio = (y - a[1]) / (b[1] - a[1])
    return a[0] + ratio * (b[0] - a[0]), y


def polygon_to_yolo(points: Sequence[Point], tile: Tile) -> str:
    values: list[str] = ["0"]
    for x, y in points:
        normalized_x = np.clip((x - tile.x) / tile.size, 0.0, 1.0)
        normalized_y = np.clip((y - tile.y) / tile.size, 0.0, 1.0)
        values.extend((f"{normalized_x:.6f}", f"{normalized_y:.6f}"))
    return " ".join(values)
