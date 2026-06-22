from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__:
    from .utils import LOGGER
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mildew_seg.utils import LOGGER


def run_training(
    config: dict[str, Any],
    model_path: Path,
    data_yaml: Path,
    output_dir: Path,
    name: str,
    resume: bool = False,
) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run `pip install -e .` first."
        ) from exc
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    settings = config["train"]
    model = YOLO(str(model_path))
    LOGGER.info("Starting training from %s", model_path)
    return model.train(
        data=str(data_yaml),
        project=str(output_dir),
        name=name,
        imgsz=int(settings["imgsz"]),
        batch=int(settings["batch"]),
        epochs=int(settings["epochs"]),
        patience=int(settings["patience"]),
        mask_ratio=int(settings["mask_ratio"]),
        device=settings["device"],
        workers=int(settings["workers"]),
        amp=bool(settings["amp"]),
        mosaic=float(settings["mosaic"]),
        mixup=float(settings["mixup"]),
        max_det=int(config["predict"]["max_det"]),
        seed=int(settings["seed"]),
        deterministic=bool(settings["deterministic"]),
        resume=resume,
        exist_ok=False,
    )


if __name__ == "__main__":
    from mildew_seg.cli import main

    raise SystemExit(main(["train", *sys.argv[1:]]))
