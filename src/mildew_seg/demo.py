from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .decision import DecisionThresholds
from .inference import MildewPredictor, save_prediction
from .utils import safe_name, timestamp


def launch_demo(
    predictor: MildewPredictor,
    output_dir: Path,
    config: dict[str, Any],
    server_name: str,
    server_port: int,
    share: bool,
) -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is not installed. Run `pip install -e .` first."
        ) from exc

    defaults = DecisionThresholds.from_config(config)

    def predict(
        rgb_image: np.ndarray | None,
        instance_confidence: float,
        mean_confidence: float,
        count_threshold: int,
        area_ratio_threshold: float,
    ) -> tuple[np.ndarray | None, str, list[list[Any]]]:
        if rgb_image is None:
            return None, "请先上传图像。", []
        thresholds = DecisionThresholds(
            float(instance_confidence),
            float(mean_confidence),
            int(count_threshold),
            float(area_ratio_threshold),
        )
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        try:
            result, visualization = predictor.predict_array(
                bgr_image,
                source="<gradio>",
                thresholds=thresholds,
            )
            stem = safe_name(f"demo_{timestamp()}")
            save_prediction(result, visualization, output_dir, stem)
            decision = result.decision
            markdown = (
                f"## 判定：`{decision.label}`\n"
                f"- 霉点实例数：{decision.spot_count}\n"
                f"- 平均置信度：{decision.mean_confidence:.4f}\n"
                f"- 最大置信度：{decision.max_confidence:.4f}\n"
                f"- 霉点面积占比：{decision.mildew_area_ratio:.6f}\n"
                f"- 推理耗时：{result.elapsed_seconds:.3f} 秒"
            )
            table = [
                [
                    decision.label,
                    decision.spot_count,
                    decision.mean_confidence,
                    decision.max_confidence,
                    decision.mildew_area,
                    decision.seed_area,
                    decision.mildew_area_ratio,
                ]
            ]
            return cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB), markdown, table
        except Exception as exc:
            return rgb_image, f"## 推理失败\n`{exc}`", []

    with gr.Blocks(title="种子霉点实例分割 DEMO") as app:
        gr.Markdown(
            "# 基于 YOLO11-seg 的种子霉点实例分割\n"
            "上传固定蓝色背景、单颗种子的图像。模型先分割霉点实例，"
            "再根据数量、置信度和面积占比判断 positive/negative。"
        )
        with gr.Row():
            input_image = gr.Image(type="numpy", label="输入图像")
            output_image = gr.Image(type="numpy", label="分割结果")
        with gr.Accordion("判定阈值", open=False):
            instance_confidence = gr.Slider(
                0.0,
                1.0,
                value=defaults.instance_confidence,
                step=0.01,
                label="实例置信度阈值",
            )
            mean_confidence = gr.Slider(
                0.0,
                1.0,
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
        button = gr.Button("开始推理", variant="primary")
        summary = gr.Markdown()
        table = gr.Dataframe(
            headers=[
                "判定",
                "实例数",
                "平均置信度",
                "最大置信度",
                "霉点面积",
                "种子面积",
                "面积占比",
            ],
            label="结果统计",
            interactive=False,
        )
        button.click(
            predict,
            inputs=[
                input_image,
                instance_confidence,
                mean_confidence,
                count_threshold,
                area_ratio_threshold,
            ],
            outputs=[output_image, summary, table],
        )
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )
