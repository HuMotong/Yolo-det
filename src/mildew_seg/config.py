from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the project configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = {"paths", "data", "tiling", "roi", "train", "predict", "decision"}
    missing = required - set(config)
    if missing:
        raise ConfigError(f"Missing config sections: {sorted(missing)}")
    config = deepcopy(config)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(_find_project_root(config_path))
    _validate_config(config)
    return config


def _find_project_root(config_path: Path) -> Path:
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(config["_project_root"]) / path).resolve()


def path_from_config(config: dict[str, Any], key: str) -> Path:
    try:
        value = config["paths"][key]
    except KeyError as exc:
        raise ConfigError(f"Missing paths.{key}") from exc
    return resolve_path(config, value)


def _validate_config(config: dict[str, Any]) -> None:
    size = int(config["tiling"]["size"])
    overlap = int(config["tiling"]["overlap"])
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ConfigError("tiling requires size > overlap >= 0")
    split = config["data"]["split"]
    total = sum(float(split[name]) for name in ("train", "val", "test"))
    if abs(total - 1.0) > 1e-6:
        raise ConfigError("data.split train/val/test values must sum to 1")
    if int(config["predict"]["max_det"]) < 1:
        raise ConfigError("predict.max_det must be positive")
