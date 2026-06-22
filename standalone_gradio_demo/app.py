"""Standalone Gradio demo for single-image mildew spot segmentation.

Edit the path constants below if your model or config path changes, then run:

    python standalone_gradio_demo/app.py
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Usually only MODEL_PATH needs to change after you train a newer model.
CONFIG_PATH = r"configs\default.yaml"
MODEL_PATH = r"runs\train\yolo11n_seg_20260605_133949\weights\best.pt"

# If this file exists, it overrides thresholds from configs/default.yaml.
# Leave as None to always use configs/default.yaml.
THRESHOLDS_PATH: str | None = r"runs\calibration\thresholds.yaml"

# Demo outputs are saved here.
OUTPUT_DIR = r"runs\standalone_gradio_demo"

# Local web server settings.
SERVER_NAME = "127.0.0.1"
SERVER_PORT = 7861
SHARE = False


def _imports() -> dict[str, Any]:
    from mildew_seg.config import load_config, resolve_path
    from mildew_seg.decision import DecisionThresholds
    from mildew_seg.inference import MildewPredictor, save_prediction
    from mildew_seg.thresholds import load_thresholds
    from mildew_seg.utils import safe_name, timestamp

    return {
        "load_config": load_config,
        "resolve_path": resolve_path,
        "DecisionThresholds": DecisionThresholds,
        "MildewPredictor": MildewPredictor,
        "save_prediction": save_prediction,
        "load_thresholds": load_thresholds,
        "safe_name": safe_name,
        "timestamp": timestamp,
    }


@lru_cache(maxsize=1)
def get_runtime() -> dict[str, Any]:
    modules = _imports()
    config = modules["load_config"](PROJECT_ROOT / CONFIG_PATH)
    model_path = modules["resolve_path"](config, MODEL_PATH)
    output_dir = modules["resolve_path"](config, OUTPUT_DIR)

    defaults = modules["DecisionThresholds"].from_config(config)
    thresholds = defaults
    threshold_path = None
    if THRESHOLDS_PATH:
        candidate = modules["resolve_path"](config, THRESHOLDS_PATH)
        if candidate.is_file():
            threshold_path = candidate
            thresholds = modules["load_thresholds"](candidate, defaults)

    predictor = modules["MildewPredictor"](model_path, config)
    return {
        **modules,
        "config": config,
        "model_path": model_path,
        "output_dir": output_dir,
        "threshold_path": threshold_path,
        "thresholds": thresholds,
        "predictor": predictor,
    }


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _metrics_rows(result: Any) -> list[list[Any]]:
    decision = result.decision
    return [
        ["图像判定", decision.label],
        ["霉点实例数", decision.spot_count],
        ["最大置信度", _round(decision.max_confidence, 4)],
        ["平均置信度", _round(decision.mean_confidence, 4)],
        ["霉点面积(px)", decision.mildew_area],
        ["种子面积(px)", decision.seed_area],
        ["霉点面积占比", _round(decision.mildew_area_ratio, 6)],
        ["推理耗时(s)", _round(result.elapsed_seconds, 3)],
    ]


def _instance_rows(result: Any, limit: int = 50) -> list[list[Any]]:
    rows: list[list[Any]] = []
    instances = sorted(
        result.instances,
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    )
    for index, instance in enumerate(instances[:limit], start=1):
        center = instance.get("center", [0.0, 0.0])
        rows.append(
            [
                index,
                _round(float(instance.get("confidence", 0.0)), 4),
                _round(float(center[0]), 1),
                _round(float(center[1]), 1),
                instance.get("tile_index", ""),
            ]
        )
    return rows


def predict_image(
    rgb_image: np.ndarray | None,
    instance_confidence: float,
    mean_confidence: float,
    count_threshold: int,
    area_ratio_threshold: float,
) -> tuple[np.ndarray | None, str, list[list[Any]], list[list[Any]], dict[str, Any]]:
    runtime = get_runtime()
    if rgb_image is None:
        return None, "请先上传一张图像。", [], [], {}

    thresholds = runtime["DecisionThresholds"](
        float(instance_confidence),
        float(mean_confidence),
        int(count_threshold),
        float(area_ratio_threshold),
    )

    try:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        result, visualization = runtime["predictor"].predict_array(
            bgr_image,
            source="<gradio-upload>",
            thresholds=thresholds,
        )
        stem = runtime["safe_name"](f"demo_{runtime['timestamp']()}")
        prediction_path, json_path = runtime["save_prediction"](
            result,
            visualization,
            runtime["output_dir"],
            stem,
        )
    except Exception as exc:
        error = f"推理失败：{type(exc).__name__}: {exc}"
        return rgb_image, error, [], [], {"error": error}

    decision = result.decision
    summary = (
        f"## 判定结果：`{decision.label}`\n\n"
        f"- 霉点实例数：{decision.spot_count}\n"
        f"- 最大置信度：{decision.max_confidence:.4f}\n"
        f"- 平均置信度：{decision.mean_confidence:.4f}\n"
        f"- 霉点面积占比：{decision.mildew_area_ratio:.6f}\n"
        f"- 推理耗时：{result.elapsed_seconds:.3f} 秒\n"
        f"- 可视化结果：`{prediction_path}`\n"
        f"- JSON 结果：`{json_path}`"
    )
    payload = result.to_dict(include_instances=False)
    payload["model"] = str(runtime["model_path"])
    payload["thresholds"] = {
        "instance_confidence": thresholds.instance_confidence,
        "mean_confidence": thresholds.mean_confidence,
        "count_threshold": thresholds.count_threshold,
        "area_ratio_threshold": thresholds.area_ratio_threshold,
    }
    return (
        cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB),
        summary,
        _metrics_rows(result),
        _instance_rows(result),
        payload,
    )


def build_app() -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is not installed. Install project dependencies first.") from exc

    runtime = get_runtime()
    defaults = runtime["thresholds"]
    threshold_source = (
        str(runtime["threshold_path"])
        if runtime["threshold_path"] is not None
        else "configs/default.yaml"
    )

    with gr.Blocks(title="种子霉点识别分割演示") as app:
        gr.Markdown(
            "# 种子霉点 YOLO11-seg 识别分割演示\n"
            "上传单张蓝色背景种子图像，系统会自动定位种子区域、分割霉点实例，"
            "并根据数量、置信度和面积占比给出 positive/negative 判定。\n\n"
            f"- 当前模型：`{runtime['model_path']}`\n"
            f"- 阈值来源：`{threshold_source}`\n"
            f"- 输出目录：`{runtime['output_dir']}`"
        )

        with gr.Row():
            input_image = gr.Image(type="numpy", label="上传图像")
            output_image = gr.Image(type="numpy", label="分割可视化结果")

        with gr.Accordion("判定阈值", open=False):
            instance_confidence = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=defaults.instance_confidence,
                step=0.01,
                label="实例置信度阈值",
            )
            mean_confidence = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=defaults.mean_confidence,
                step=0.01,
                label="平均置信度阈值",
            )
            count_threshold = gr.Number(
                value=defaults.count_threshold,
                precision=0,
                label="霉点数量阈值",
            )
            area_ratio_threshold = gr.Number(
                value=defaults.area_ratio_threshold,
                label="霉点面积占比阈值",
            )

        run_button = gr.Button("开始识别分割", variant="primary")
        summary = gr.Markdown(label="结果摘要")
        metrics = gr.Dataframe(
            headers=["指标", "数值"],
            label="单图推理统计",
            interactive=False,
        )
        instances = gr.Dataframe(
            headers=["序号", "置信度", "中心X", "中心Y", "切片编号"],
            label="霉点实例明细（按置信度前 50 个）",
            interactive=False,
        )
        result_json = gr.JSON(label="JSON 摘要")

        run_button.click(
            predict_image,
            inputs=[
                input_image,
                instance_confidence,
                mean_confidence,
                count_threshold,
                area_ratio_threshold,
            ],
            outputs=[output_image, summary, metrics, instances, result_json],
        )

    return app


if __name__ == "__main__":
    build_app().launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=SHARE,
    )
