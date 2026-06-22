# 种子霉点 YOLO11-seg DEMO

本项目使用 Ultralytics YOLO11-seg 训练单类别 `mildew_spot` 实例分割模型。
`positive` 和 `negative` 只用于原图级分层划分、阈值校准和最终评估，不作为
YOLO 训练类别。推理时先分割霉点，再根据实例数量、置信度和面积占比判断图像
为 positive 或 negative。

## 项目流程

1. 检查 LabelMe JSON、图像配对、标签、polygon 和尺寸。
2. 将 `median`、`meidian` 统一为 `mildew_spot`。
3. 使用 `seed` polygon 定位 ROI，生成原始分辨率 `320×320` 重叠切片。
4. 从 `checkpoints/yolo11n-seg.pt` 微调 YOLO11n-seg。
5. 在原图上自动定位种子、切片推理、恢复坐标并计算判定规则。
6. 使用验证集校准阈值，在测试集上输出无泄漏评估。

原始 `mildew/` 数据不会被修改。所有生成内容保存到 `runs/`。

## 环境安装

建议使用 Python 3.10 或 3.11。RTX 5070 需要支持 Blackwell 的新版 CUDA
PyTorch。先在 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)
选择 Windows、Pip 和当前推荐 CUDA 版本安装 PyTorch，再安装本项目：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 先执行 PyTorch 官网针对当前 CUDA 给出的安装命令
python -m pip install -e ".[dev]"
```

检查环境：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import ultralytics, gradio, cv2; print(ultralytics.__version__, gradio.__version__, cv2.__version__)"
mildew-seg --version
```

## 数据准备

默认配置位于 `configs/default.yaml`。相对路径以项目根目录解析。

```powershell
mildew-seg check
mildew-seg prepare
```

主要输出：

- `runs/data/audit_report.json`
- `runs/data/audit_files.csv`
- `runs/data/audit_issues.csv`
- `runs/data/manifest.csv`
- `runs/data/tiles.csv`
- `runs/data/data.yaml`
- `runs/data/dataset/images/{train,val,test}`
- `runs/data/dataset/labels/{train,val,test}`

少于 3 点或零面积 polygon 会被跳过。缺少 `seed` 时，仅当唯一超大 polygon
满足配置中的面积与倍数条件才推断为 seed；修复只出现在生成数据和审计报告中。

## 训练与验证

```powershell
mildew-seg train
```

默认从 `checkpoints/yolo11n-seg.pt` 开始，使用 `imgsz=1024`、`batch=4`、
`epochs=150`、`mask_ratio=2`。训练结果保存在 `runs/train/`。

训练完成后校准规则：

```powershell
mildew-seg calibrate --model runs/train/<run>/weights/best.pt
```

验证集用于搜索规则阈值，结果写入
`runs/calibration/thresholds.yaml`。独立测试集评估：

```powershell
mildew-seg validate --model runs/train/<run>/weights/best.pt --split test
```

验证同时输出 Ultralytics 切片级 box/mask 指标和原图级分类、数量误差、面积占比
误差。未传 `--model` 时，命令会优先使用 `runs/train/` 中最新的 `best.pt`。

## 推理

单图：

```powershell
mildew-seg predict --source "mildew/positive/famei (1).png"
```

批量：

```powershell
mildew-seg batch --source mildew/positive
```

结果分别写入 `runs/predict/` 和 `runs/batch/`。批量 CSV 包含图像路径、判定、
实例数、最大/平均置信度、霉点面积、种子面积、面积占比、耗时和错误信息。

默认判定公式：

```text
mean_confidence >= threshold
AND
(spot_count >= count_threshold OR mildew_area_ratio >= area_threshold)
```

## Gradio

```powershell
mildew-seg demo
```

打开 `http://127.0.0.1:7860`。界面支持上传图像、调整四个判定阈值、查看 mask
可视化和统计结果。输入假设为固定蓝色背景、单颗种子；种子定位失败会明确报错，
不会静默判为 negative。

## 配置覆盖

所有命令可指定其他 YAML：

```powershell
mildew-seg train --config configs/default.yaml
```

训练、预测和校准参数应优先在 YAML 中维护。单次输入路径和模型路径使用命令行
参数覆盖，避免在代码中写死路径。

## 测试

```powershell
python -m pytest
ruff check .
ruff format --check .
```

VSCode 的 Run and Debug 中提供数据检查、数据准备、训练和 Gradio 启动项。
