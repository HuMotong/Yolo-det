"""Single-image mildew spot segmentation test.

Usage in VSCode:
1. Edit IMAGE_PATH below.
2. Open this file.
3. Click "Run Python File" or press F5 with the current-file launcher.

This script intentionally does not use argparse. It is meant as a small,
editable test entrypoint for one image at a time.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Change this to the image you want to test.
IMAGE_PATH = r"mildew\positive\famei (1).png"

# Usually you do not need to change these.
CONFIG_PATH = r"configs\default.yaml"
MODEL_PATH = r"runs\train\yolo11n_seg_20260605_133949\weights\best.pt"

# Optional. If this file exists, it will override the decision thresholds
# from configs/default.yaml. Leave as None to use config thresholds.
THRESHOLDS_PATH: str | None = r"runs\calibration\thresholds.yaml"

# Outputs are written under this folder.
OUTPUT_DIR = r"runs\single_image_test"


def main() -> None:
    from mildew_seg.config import load_config, resolve_path
    from mildew_seg.decision import DecisionThresholds
    from mildew_seg.inference import MildewPredictor, save_prediction
    from mildew_seg.thresholds import load_thresholds
    from mildew_seg.utils import safe_name

    config = load_config(PROJECT_ROOT / CONFIG_PATH)
    image_path = resolve_path(config, IMAGE_PATH)
    model_path = resolve_path(config, MODEL_PATH)
    output_dir = resolve_path(config, OUTPUT_DIR)

    thresholds = DecisionThresholds.from_config(config)
    if THRESHOLDS_PATH:
        threshold_path = resolve_path(config, THRESHOLDS_PATH)
        if threshold_path.is_file():
            thresholds = load_thresholds(threshold_path, thresholds)
        else:
            print(f"[WARN] Threshold file not found, using config values: {threshold_path}")

    predictor = MildewPredictor(model_path, config)
    result, visualization = predictor.predict_path(image_path, thresholds)
    prediction_path, json_path = save_prediction(
        result,
        visualization,
        output_dir,
        safe_name(image_path.stem),
    )

    decision = result.decision
    print("Single-image prediction complete")
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Decision: {decision.label}")
    print(f"Spot count: {decision.spot_count}")
    print(f"Max confidence: {decision.max_confidence:.4f}")
    print(f"Mean confidence: {decision.mean_confidence:.4f}")
    print(f"Mildew area: {decision.mildew_area}")
    print(f"Seed area: {decision.seed_area}")
    print(f"Mildew area ratio: {decision.mildew_area_ratio:.6f}")
    print(f"Elapsed seconds: {result.elapsed_seconds:.3f}")
    print(f"Prediction image: {prediction_path}")
    print(f"Result JSON: {json_path}")


if __name__ == "__main__":
    main()
