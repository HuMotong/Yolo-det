# 独立 Gradio 演示说明

这个目录提供一个不依赖命令行参数的单文件 Gradio 演示：

- `app.py`：启动网页演示，上传单张图像并显示识别分割结果。

它不会修改项目其他文件，推理结果会保存到：

```text
runs/standalone_gradio_demo/
```

## 使用方法

在项目根目录运行：

```powershell
D:\Anaconda3\envs\torch2.9py310\python.exe standalone_gradio_demo\app.py
```

或者在 VSCode 中打开 `standalone_gradio_demo/app.py`，点击 `Run Python File`。

启动后浏览器打开：

```text
http://127.0.0.1:7861
```

如果没有自动打开，手动复制上面的地址到浏览器。

## 需要修改的路径

通常只需要修改 `app.py` 顶部这几项：

```python
CONFIG_PATH = r"configs\default.yaml"
MODEL_PATH = r"runs\train\yolo11n_seg_20260605_133949\weights\best.pt"
THRESHOLDS_PATH = r"runs\calibration\thresholds.yaml"
OUTPUT_DIR = r"runs\standalone_gradio_demo"
```

如果 `THRESHOLDS_PATH` 指向的文件不存在，程序会自动使用 `configs/default.yaml` 中的判定阈值。

## 页面展示内容

页面会展示：

- 上传图像
- 霉点 mask 分割可视化图
- positive/negative 判定
- 霉点实例数
- 最大置信度
- 平均置信度
- 霉点面积
- 种子面积
- 霉点面积占比
- 推理耗时
- 前 50 个霉点实例的置信度和中心点
- JSON 摘要

注意：这是单图推理演示，没有人工标注真值，因此页面显示的是单图推理统计指标，不是验证集 mAP、Precision、Recall 这类评估指标。
