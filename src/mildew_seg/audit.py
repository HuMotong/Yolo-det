from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .config import path_from_config
from .images import read_image
from .labelme import LabelMeRecord, load_labelme_record
from .utils import LOGGER, write_csv, write_json


def discover_records(
    config: dict[str, Any],
    allow_repair: bool = True,
) -> tuple[list[LabelMeRecord], list[dict[str, Any]]]:
    raw_root = path_from_config(config, "raw_data")
    extensions = {value.lower() for value in config["data"]["image_extensions"]}
    records: list[LabelMeRecord] = []
    global_issues: list[dict[str, Any]] = []
    directories = {
        "positive": raw_root / config["data"]["positive_dir"],
        "negative": raw_root / config["data"]["negative_dir"],
    }

    for category, directory in directories.items():
        if not directory.is_dir():
            global_issues.append(
                {
                    "code": "missing_category_directory",
                    "severity": "error",
                    "category": category,
                    "path": str(directory),
                }
            )
            continue
        images = {
            item.stem: item
            for item in directory.iterdir()
            if item.is_file() and item.suffix.lower() in extensions
        }
        jsons = {item.stem: item for item in directory.glob("*.json")}
        for stem in sorted(images.keys() - jsons.keys()):
            global_issues.append(
                {
                    "code": "missing_json",
                    "severity": "error",
                    "category": category,
                    "path": str(images[stem]),
                }
            )
        for stem in sorted(jsons.keys() - images.keys()):
            global_issues.append(
                {
                    "code": "missing_image",
                    "severity": "error",
                    "category": category,
                    "path": str(jsons[stem]),
                }
            )
        for stem in sorted(images.keys() & jsons.keys()):
            try:
                record = load_labelme_record(
                    jsons[stem],
                    images[stem],
                    category,
                    config,
                    allow_repair=allow_repair,
                )
                image = read_image(images[stem])
                height, width = image.shape[:2]
                if (width, height) != (record.image_width, record.image_height):
                    record.issues.append(
                        {
                            "code": "dimension_mismatch",
                            "severity": "error",
                            "json_size": [record.image_width, record.image_height],
                            "image_size": [width, height],
                            "message": "JSON and image dimensions differ",
                        }
                    )
                records.append(record)
            except Exception as exc:
                global_issues.append(
                    {
                        "code": "record_read_error",
                        "severity": "error",
                        "category": category,
                        "json_path": str(jsons[stem]),
                        "image_path": str(images[stem]),
                        "message": repr(exc),
                    }
                )
    return records, global_issues


def run_audit(
    config: dict[str, Any],
    output_dir: Path,
    allow_repair: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records, global_issues = discover_records(config, allow_repair=allow_repair)
    rows: list[dict[str, Any]] = []
    issue_rows = list(global_issues)
    label_counts: Counter[str] = Counter()

    for record in records:
        label_counts["mildew_spot"] += len(record.mildew_spots)
        label_counts["seed"] += len(record.seed_polygons)
        severities = Counter(issue["severity"] for issue in record.issues)
        rows.append(
            {
                "category": record.category,
                "image_path": str(record.image_path),
                "json_path": str(record.json_path),
                "width": record.image_width,
                "height": record.image_height,
                "mildew_spot_count": len(record.mildew_spots),
                "mildew_area": record.mildew_area,
                "seed_count": len(record.seed_polygons),
                "seed_area": record.seed_area,
                "seed_inferred": record.seed_inferred,
                "warning_count": severities["warning"],
                "error_count": severities["error"],
            }
        )
        for issue in record.issues:
            issue_rows.append(
                {
                    "category": record.category,
                    "image_path": str(record.image_path),
                    "json_path": str(record.json_path),
                    **issue,
                }
            )

    write_csv(output_dir / "audit_files.csv", rows)
    write_csv(output_dir / "audit_issues.csv", issue_rows)
    severity_counts = Counter(issue.get("severity", "unknown") for issue in issue_rows)
    issue_code_counts = Counter(issue.get("code", "unknown") for issue in issue_rows)
    summary = {
        "raw_data": str(path_from_config(config, "raw_data")),
        "record_count": len(records),
        "category_counts": dict(Counter(record.category for record in records)),
        "label_counts": dict(label_counts),
        "severity_counts": dict(severity_counts),
        "issue_code_counts": dict(issue_code_counts),
        "seed_inferred_count": sum(record.seed_inferred for record in records),
        "files": rows,
        "issues": issue_rows,
    }
    write_json(output_dir / "audit_report.json", summary)
    LOGGER.info(
        "Audited %d records: %s; issues=%s",
        len(records),
        summary["category_counts"],
        dict(severity_counts),
    )
    return summary
