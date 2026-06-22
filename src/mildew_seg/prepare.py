from __future__ import annotations

import hashlib
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .audit import discover_records, run_audit
from .config import path_from_config
from .geometry import (
    build_tiles,
    clip_polygon_to_rect,
    combined_bbox,
    expand_bbox,
    polygon_area,
    polygon_centroid,
    polygon_to_yolo,
)
from .images import read_image, write_image
from .labelme import LabelMeRecord
from .utils import LOGGER, safe_name, write_csv, write_json


def split_records(
    records: list[LabelMeRecord],
    config: dict[str, Any],
) -> dict[str, list[LabelMeRecord]]:
    settings = config["data"]["split"]
    rng = random.Random(int(settings["seed"]))
    result = {"train": [], "val": [], "test": []}
    for category in ("positive", "negative"):
        items = sorted(
            (record for record in records if record.category == category),
            key=lambda record: str(record.image_path),
        )
        rng.shuffle(items)
        train_count = round(len(items) * float(settings["train"]))
        val_count = round(len(items) * float(settings["val"]))
        if train_count + val_count > len(items):
            val_count = max(0, len(items) - train_count)
        result["train"].extend(items[:train_count])
        result["val"].extend(items[train_count : train_count + val_count])
        result["test"].extend(items[train_count + val_count :])
    for split in result.values():
        split.sort(key=lambda record: (record.category, str(record.image_path)))
    return result


def prepare_dataset(
    config: dict[str, Any],
    output_dir: Path,
    clean: bool = True,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    runs_root = path_from_config(config, "runs").resolve()
    if clean and output_dir.exists():
        if output_dir == runs_root or runs_root not in output_dir.parents:
            raise ValueError(f"Refusing to clean outside runs directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_audit(config, output_dir, allow_repair=True)
    records, global_issues = discover_records(config, allow_repair=True)
    errors = [issue for issue in global_issues if issue.get("severity") == "error"]
    errors.extend(
        {"image_path": str(record.image_path), **issue}
        for record in records
        for issue in record.issues
        if issue.get("severity") == "error"
    )
    if errors:
        raise ValueError(
            f"Data preparation stopped because {len(errors)} audit errors remain. "
            f"See {output_dir / 'audit_issues.csv'}"
        )

    splits = split_records(records, config)
    dataset_root = output_dir / "dataset"
    original_rows: list[dict[str, Any]] = []
    tile_rows: list[dict[str, Any]] = []
    for split_name, split_records_list in splits.items():
        image_dir = dataset_root / "images" / split_name
        label_dir = dataset_root / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for record in split_records_list:
            original_row, generated_tiles = _prepare_record(
                record, split_name, image_dir, label_dir, config
            )
            original_rows.append(original_row)
            tile_rows.extend(generated_tiles)

    manifest_path = output_dir / "manifest.csv"
    tiles_path = output_dir / "tiles.csv"
    write_csv(manifest_path, original_rows)
    write_csv(tiles_path, tile_rows)
    data_yaml = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "mildew_spot"},
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, allow_unicode=True, sort_keys=False)

    summary = {
        "original_count": len(original_rows),
        "tile_count": len(tile_rows),
        "split_counts": dict(Counter(row["split"] for row in original_rows)),
        "category_split_counts": dict(
            Counter(f"{row['split']}/{row['category']}" for row in original_rows)
        ),
        "tile_split_counts": dict(Counter(row["split"] for row in tile_rows)),
        "dataset_root": str(dataset_root),
        "data_yaml": str(output_dir / "data.yaml"),
        "manifest": str(manifest_path),
        "tiles": str(tiles_path),
    }
    write_json(output_dir / "prepare_summary.json", summary)
    LOGGER.info(
        "Prepared %d originals and %d tiles at %s",
        len(original_rows),
        len(tile_rows),
        dataset_root,
    )
    return summary


def _prepare_record(
    record: LabelMeRecord,
    split: str,
    image_dir: Path,
    label_dir: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image = read_image(record.image_path)
    tiling = config["tiling"]
    roi = expand_bbox(
        combined_bbox(annotation.points for annotation in record.seed_polygons),
        int(tiling["roi_margin"]),
        record.image_width,
        record.image_height,
    )
    tiles = build_tiles(
        roi,
        record.image_width,
        record.image_height,
        int(tiling["size"]),
        int(tiling["overlap"]),
    )
    digest = hashlib.sha1(str(record.image_path).encode("utf-8")).hexdigest()[:8]
    source_id = f"{record.category}_{safe_name(record.image_path.stem)}_{digest}"
    tile_rows: list[dict[str, Any]] = []
    assigned = 0

    annotations_by_tile: dict[int, list[str]] = {tile.index: [] for tile in tiles}
    for annotation in record.mildew_spots:
        center = polygon_centroid(annotation.points)
        owner = next((tile for tile in tiles if tile.owns(center)), None)
        if owner is None:
            continue
        clipped = clip_polygon_to_rect(
            annotation.points,
            owner.x,
            owner.y,
            owner.x + owner.size,
            owner.y + owner.size,
        )
        if len(clipped) < 3 or polygon_area(clipped) <= 0:
            continue
        annotations_by_tile[owner.index].append(polygon_to_yolo(clipped, owner))
        assigned += 1

    for tile in tiles:
        tile_name = f"{source_id}_x{tile.x}_y{tile.y}"
        image_path = image_dir / f"{tile_name}.png"
        label_path = label_dir / f"{tile_name}.txt"
        crop = image[tile.y : tile.y + tile.size, tile.x : tile.x + tile.size]
        if crop.shape[:2] != (tile.size, tile.size):
            raise ValueError(
                f"Unexpected tile size for {record.image_path}: {crop.shape}"
            )
        write_image(image_path, crop)
        labels = annotations_by_tile[tile.index]
        label_path.write_text(
            "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
        )
        tile_rows.append(
            {
                "source_id": source_id,
                "source_image": str(record.image_path.resolve()),
                "category": record.category,
                "split": split,
                "tile_image": str(image_path.resolve()),
                "tile_label": str(label_path.resolve()),
                "tile_x": tile.x,
                "tile_y": tile.y,
                "tile_size": tile.size,
                "core_left": tile.core_left,
                "core_top": tile.core_top,
                "core_right": tile.core_right,
                "core_bottom": tile.core_bottom,
                "instance_count": len(labels),
            }
        )

    if assigned != len(record.mildew_spots):
        LOGGER.warning(
            "%s assigned %d/%d mildew polygons to tiles",
            record.image_path,
            assigned,
            len(record.mildew_spots),
        )
    original_row = {
        "source_id": source_id,
        "source_image": str(record.image_path.resolve()),
        "source_json": str(record.json_path.resolve()),
        "category": record.category,
        "split": split,
        "width": record.image_width,
        "height": record.image_height,
        "roi_left": roi[0],
        "roi_top": roi[1],
        "roi_right": roi[2],
        "roi_bottom": roi[3],
        "seed_area": record.seed_area,
        "gt_spot_count": len(record.mildew_spots),
        "gt_mildew_area": record.mildew_area,
        "gt_mildew_area_ratio": (
            record.mildew_area / record.seed_area if record.seed_area else 0.0
        ),
        "seed_inferred": record.seed_inferred,
        "tile_count": len(tiles),
    }
    return original_row, tile_rows
