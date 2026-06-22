# 项目结构说明

这份文档用于解释本项目每个主要文件和文件夹的作用。目标读者是假设完全没有接触过该项目的人，读完后能理解项目由哪些部分组成、数据如何流动、训练和推理分别由哪些代码负责。

## 1. 项目做什么

本项目是一个基于 Ultralytics YOLO11-seg 的种子霉点实例分割项目。

它不是训练一个 `positive/negative` 图像分类器，而是训练一个单类别分割模型：

```text
0: mildew_spot
```

也就是说，模型只负责找出图像中的霉点实例。图像最终被判断为 `positive` 或 `negative`，是推理后根据规则计算出来的：

```text
平均置信度达到阈值
并且
霉点数量达到阈值 或 霉点面积占比达到阈值
```

项目中的 `seed` 标注不作为 YOLO 训练类别，它只用于定位种子区域、裁剪 ROI、计算霉点面积占比。

## 2. 整体数据流

项目的主流程如下：

```text
mildew/ 原始图像和 LabelMe JSON
        |
        | mildew-seg check
        v
runs/data/audit_*.csv / audit_report.json
        |
        | mildew-seg prepare
        v
runs/data/dataset/ YOLO-seg 切片数据集
        |
        | mildew-seg train
        v
runs/train/.../weights/best.pt
        |
        | mildew-seg validate / calibrate / predict / batch / demo
        v
验证指标、单图预测、批量预测、Gradio 展示
```

训练阶段使用切片图像训练 YOLO11-seg。推理阶段也会先自动定位种子，再切成小块推理，最后把切片结果恢复到原图坐标系。

## 3. 根目录文件

### `README.md`

项目使用说明。主要面向使用者，说明如何安装环境、准备数据、训练、验证、推理和启动 Gradio。

如果只想知道“怎么跑”，优先看这个文件。

### `PROJECT_STRUCTURE.md`

也就是当前文档。主要面向新接手项目的人，解释项目文件结构和每个模块的职责。

如果想知道“代码怎么组织”，看这个文件。

### `pyproject.toml`

Python 项目的安装和工具配置文件。

它定义了：

- 项目名：`mildew-seg`
- Python 版本要求
- 依赖包，例如 `ultralytics`、`opencv-python`、`PyYAML`、`gradio`
- 开发依赖，例如 `pytest`、`ruff`
- 命令行入口：

```toml
mildew-seg = "mildew_seg.cli:main"
```

这表示安装项目后，可以在终端运行：

```powershell
mildew-seg check
mildew-seg train
mildew-seg predict
```

### `.gitignore`

Git 忽略规则。用于避免把缓存、虚拟环境、训练输出、预测结果等本地文件提交到仓库。

重要忽略项包括：

- `__pycache__/`
- `.pytest_cache/`
- `.ruff_cache/`
- `.venv/`
- `runs/`

### `.editorconfig`

编辑器通用格式配置。用于约束缩进、换行等基础代码风格。

### `.gitattributes`

Git 属性配置。通常用于控制文本文件换行、二进制文件处理等。

## 4. VSCode 配置

### `.vscode/settings.json`

VSCode 工作区设置。

主要作用：

- 指定 Python 解释器路径
- 设置 Pylance 类型检查模式
- 设置 Python 文件保存时用 Ruff 格式化
- 把 `src/` 加入 Python 分析路径，方便 VSCode 识别 `mildew_seg` 包

### `.vscode/launch.json`

VSCode 调试配置。

里面包含几类启动方式：

- `Python: Current File`：直接运行当前打开的 Python 文件
- `Python: Current File (subprocess)`：运行当前文件并支持子进程调试
- `Mildew: Check Data`：运行 `mildew-seg check`
- `Mildew: Prepare Dataset`：运行 `mildew-seg prepare`
- `Mildew: Train`：运行 `mildew-seg train`
- `Mildew: Gradio Demo`：运行 `mildew-seg demo`

这些配置都设置了 `PYTHONPATH=${workspaceFolder}/src`，所以即使还没执行 `pip install -e .`，VSCode 也能找到源码包。

## 5. 配置目录

### `configs/default.yaml`

项目默认配置文件。绝大部分路径、训练参数、推理参数、阈值都在这里配置。

主要配置块：

- `paths`：原始数据、预训练权重、运行输出目录
- `data`：数据集目录名、标签映射、划分比例、异常修复规则
- `tiling`：切片大小、重叠宽度、ROI 外扩边距
- `roi`：推理时自动定位种子的参数
- `train`：YOLO 训练参数
- `predict`：推理参数
- `decision`：positive/negative 判定阈值
- `calibration`：自动校准阈值时的搜索范围

如果需要改训练轮数、batch size、模型权重路径、输入数据路径，优先改这里。

## 6. 权重目录

### `checkpoints/`

存放预训练模型权重。

当前主要文件：

- `checkpoints/yolo11n-seg.pt`：用于 YOLO11-seg 微调的初始权重
- `checkpoints/yolo11n.pt`：普通检测模型权重，不是本项目分割训练的默认权重

### `yolo11n.pt`

根目录下也存在一个 `yolo11n.pt`。它不是当前配置中的默认训练权重。默认使用的是：

```text
checkpoints/yolo11n-seg.pt
```

## 7. 原始数据目录

### `mildew/`

原始数据目录。它是输入数据，不应该被训练、转换、推理代码直接修改。

结构大致是：

```text
mildew/
  positive/
    *.png
    *.json
  negative/
    *.png
    *.json
```

其中：

- `positive/`：原始图像级标签为 positive 的种子图像
- `negative/`：原始图像级标签为 negative 的种子图像
- `.png`：原始图像
- `.json`：LabelMe 标注文件

LabelMe JSON 中可能出现的标签：

- `seed`：种子轮廓，只用于 ROI 和面积归一化，不作为 YOLO 类别
- `median`
- `meidian`
- `mildew_spot`

其中 `median`、`meidian`、`mildew_spot` 都会统一映射为 YOLO 类别 `mildew_spot`。

数据文件很多，所以这里不逐个解释每张图像和 JSON。理解这个目录时，只需要记住：它是原始数据源，`positive/negative` 是图像级真值，真正训练的实例类别只有霉点。

## 8. 源码目录

源码位于：

```text
src/mildew_seg/
```

这是一个标准 Python 包。下面逐个解释每个源码文件。

### `src/mildew_seg/__init__.py`

包初始化文件。

当前主要定义项目版本号：

```python
__version__ = "0.1.0"
```

### `src/mildew_seg/cli.py`

命令行入口。

安装项目后，`mildew-seg` 命令最终会进入这个文件的 `main()` 函数。

它负责解析命令行参数，并把不同子命令分发给对应模块：

- `check` -> 数据检查
- `prepare` -> 数据转换和切片
- `train` -> 训练
- `validate` -> 验证
- `predict` -> 单图推理
- `batch` -> 批量推理
- `calibrate` -> 阈值校准
- `demo` -> Gradio DEMO

如果要新增一个命令，通常需要改这个文件。

### `src/mildew_seg/config.py`

配置读取与路径解析。

主要职责：

- 读取 YAML 配置文件
- 校验配置字段是否存在
- 把相对路径解析为项目根目录下的绝对路径
- 检查切片大小、划分比例等配置是否合法

例如 `configs/default.yaml` 中的：

```yaml
paths:
  raw_data: mildew
```

会被解析成当前项目下的绝对路径。

### `src/mildew_seg/utils.py`

通用工具函数。

包含：

- 日志初始化
- 时间戳生成
- 文件名安全化
- JSON 写入
- CSV 读写
- 查找图像文件
- 支持 NumPy 类型序列化的 JSON encoder

这个文件不关心业务逻辑，只提供各模块共用的小工具。

### `src/mildew_seg/images.py`

图像读写工具。

使用 OpenCV 读取和保存图像，并通过 `np.fromfile` / `tofile` 兼容 Windows 中文路径。

如果直接用 `cv2.imread()` 读取中文路径，有时会失败；这个文件就是为了解决这类路径兼容问题。

### `src/mildew_seg/geometry.py`

几何计算模块。

主要负责：

- polygon 面积计算
- polygon 中心点计算
- polygon 外接框计算
- ROI 外扩
- 根据 ROI 生成重叠切片
- 判断一个实例属于哪个切片核心区
- 把 polygon 裁剪到切片范围内
- 把 polygon 转成 YOLO-seg 标签格式

这是数据转换和切片推理都依赖的基础模块。

### `src/mildew_seg/labelme.py`

LabelMe JSON 解析模块。

主要负责：

- 读取 LabelMe JSON
- 解析图像尺寸
- 解析 `shapes`
- 标准化标签名
- 把 `median`、`meidian` 统一为 `mildew_spot`
- 把 `seed` 单独保存
- 跳过少于 3 点或零面积 polygon
- 在特殊情况下自动推断缺失的 `seed`

这个文件不会修改原始 JSON，只是在内存中生成转换后的标注对象。

### `src/mildew_seg/audit.py`

数据检查模块。

对应命令：

```powershell
mildew-seg check
```

主要检查：

- 图像和 JSON 是否一一配对
- JSON 是否能读取
- 图像尺寸和 JSON 中尺寸是否一致
- 标签是否合法
- polygon 是否退化
- polygon 坐标是否越界
- 是否缺少 `seed`
- 是否触发 seed 自动推断

输出在：

```text
runs/data/audit_report.json
runs/data/audit_files.csv
runs/data/audit_issues.csv
```

### `src/mildew_seg/prepare.py`

数据准备和格式转换模块。

对应命令：

```powershell
mildew-seg prepare
```

主要负责：

- 调用数据检查
- 按原图进行 train/val/test 分层划分
- 根据 `seed` polygon 得到种子 ROI
- 生成重叠切片
- 把 LabelMe polygon 转成 YOLO-seg `.txt` 标签
- 生成 Ultralytics 能直接训练的 `data.yaml`
- 输出 manifest 和切片索引

重要输出：

```text
runs/data/dataset/images/{train,val,test}/
runs/data/dataset/labels/{train,val,test}/
runs/data/data.yaml
runs/data/manifest.csv
runs/data/tiles.csv
```

这里最重要的设计是：先按原图划分，再切片。这样可以避免同一张原图的不同切片同时出现在训练集和验证集里，造成数据泄漏。

### `src/mildew_seg/train.py`

训练模块。

对应命令：

```powershell
mildew-seg train
```

主要负责：

- 加载 `checkpoints/yolo11n-seg.pt`
- 读取 `runs/data/data.yaml`
- 调用 Ultralytics `YOLO.train()`
- 把训练结果写入 `runs/train/`

这个文件也支持被 VSCode 直接运行。直接运行时会自动转到 CLI 的 `train` 子命令。

### `src/mildew_seg/roi.py`

推理阶段自动定位种子区域的模块。

它假设输入图像是固定蓝色背景、单颗种子。

主要流程：

1. 从图像边缘估计背景颜色
2. 计算每个像素和背景颜色的距离
3. 提取非背景区域
4. 通过形态学操作去噪
5. 找最大连通域作为种子
6. 输出种子 mask、ROI 和种子面积

如果无法定位种子，会抛出明确异常，不会静默判定为 negative。

### `src/mildew_seg/inference.py`

核心推理模块。

对应单图、批量和 Gradio 都会用到它。

主要负责：

- 加载训练好的 YOLO 模型
- 读取一张图像
- 自动定位种子 ROI
- 按 ROI 生成切片
- 对切片调用 YOLO 推理
- 把切片坐标恢复到原图坐标
- 只保留属于切片核心区的实例，减少重叠切片造成的重复预测
- 只保留落在种子 mask 内的实例
- 计算霉点 mask 并集面积
- 根据规则生成 positive/negative 判定
- 返回可视化图和 JSON 结果

这是项目推理阶段最核心的文件。

### `src/mildew_seg/decision.py`

图像级判定规则模块。

模型输出的是霉点实例，不直接输出 positive/negative。这个文件负责把实例统计量转换为图像级结果。

主要内容：

- `DecisionThresholds`：判定阈值
- `Decision`：判定结果
- `classify_image()`：执行判定公式

判定公式：

```text
mean_confidence >= mean_confidence_threshold
AND
(spot_count >= count_threshold OR mildew_area_ratio >= area_ratio_threshold)
```

### `src/mildew_seg/thresholds.py`

阈值文件读写模块。

主要负责：

- 从 YAML 中读取校准后的判定阈值
- 保存校准后的阈值到 YAML

校准结果默认保存到：

```text
runs/calibration/thresholds.yaml
```

### `src/mildew_seg/calibrate.py`

阈值校准模块。

对应命令：

```powershell
mildew-seg calibrate
```

主要作用：

- 在验证集原图上运行推理
- 搜索不同的实例置信度、平均置信度、数量阈值、面积占比阈值
- 以 F1 分数作为主要目标选择最优阈值
- 保存搜索结果和最佳阈值

输出目录：

```text
runs/calibration/
```

### `src/mildew_seg/validate.py`

验证模块。

对应命令：

```powershell
mildew-seg validate --split test
```

它做两类验证：

1. 切片级 YOLO box/mask 指标
2. 原图级 positive/negative 判定指标

切片级指标来自 Ultralytics `model.val()`，包括：

- Box Precision
- Box Recall
- Box mAP50
- Box mAP50-95
- Mask Precision
- Mask Recall
- Mask mAP50
- Mask mAP50-95

另外它还会按原始图像类别分别统计：

- positive 样本的 box/mask 指标
- negative 样本的 box/mask 指标

相关输出：

```text
runs/val/test/slice_metrics.json
runs/val/test/validation_summary.json
runs/val/test/original_predictions.csv
runs/val/test/confusion_matrix.csv
runs/val/test/slice_metrics_by_category/
```

### `src/mildew_seg/metrics.py`

指标计算模块。

当前主要实现二分类指标计算：

- TP
- TN
- FP
- FN
- Precision
- Recall
- F1
- Accuracy

用于评估原图级 positive/negative 判定结果。

### `src/mildew_seg/batch.py`

批量推理模块。

对应命令：

```powershell
mildew-seg batch --source <图像目录>
```

主要负责：

- 查找目录下所有支持的图像文件
- 对每张图调用推理流程
- 保存每张图的可视化结果
- 汇总所有图像的判定结果到 CSV

输出目录：

```text
runs/batch/
```

### `src/mildew_seg/demo.py`

Gradio 网页 DEMO 模块。

对应命令：

```powershell
mildew-seg demo
```

主要负责：

- 创建 Gradio 页面
- 上传单张图像
- 调整判定阈值
- 显示分割可视化图
- 显示 positive/negative 判定和统计表
- 把 DEMO 推理结果保存到 `runs/demo/`

### `src/mildew_seg/visualize.py`

可视化模块。

主要负责把预测到的霉点 polygon 画回原图：

- positive 用红色
- negative 用绿色
- 在图像左上角绘制判定结果和实例数量

### `src/mildew_seg/__pycache__/`

Python 自动生成的字节码缓存目录。

它不是源码，不需要阅读，也不应该手工修改。通常会被 `.gitignore` 忽略。

## 9. 测试目录

测试位于：

```text
tests/
```

这些测试用于保证项目关键逻辑不会被改坏。

### `tests/test_geometry.py`

测试几何模块。

覆盖：

- polygon 面积
- polygon 中心点
- polygon 裁剪
- 重叠切片核心区归属

### `tests/test_labelme.py`

测试 LabelMe 解析模块。

覆盖：

- 标签别名映射
- 退化 polygon 跳过
- 缺失 seed 时的超大 polygon 自动推断

### `tests/test_roi_decision.py`

测试种子 ROI 定位和图像级判定规则。

覆盖：

- 蓝色背景下定位种子
- 没有前景时明确失败
- positive/negative 判定公式

### `tests/test_prepare_integration.py`

测试数据准备流程。

它会临时创建一个小型合成 LabelMe 数据集，然后验证：

- 能生成 manifest
- 能生成 tiles
- YOLO 标签类别都是 `0`
- 能生成 `data.yaml`

### `tests/test_real_data_regression.py`

真实数据回归测试。

如果本地存在 `mildew/` 数据目录，它会检查当前真实数据中的已知情况：

- 总记录数为 148
- 发现 10 个退化 polygon
- `famei (89).json` 触发一次 seed 推断

这个测试用于防止数据检查逻辑被改坏。

### `tests/test_inference.py`

测试推理流程。

它使用假模型模拟 YOLO 输出，不依赖真实权重。

覆盖：

- 切片推理结果恢复到原图坐标
- 实例中心点是否正确
- 图像级判定是否正确
- 可视化输出尺寸是否正确

### `tests/test_train_entrypoint.py`

测试 `src/mildew_seg/train.py` 能否被直接执行。

这是为了解决 VSCode 直接运行包内文件时可能出现的相对导入问题。

### `tests/test_validate.py`

测试验证阶段的正负样本分组 box/mask 指标统计。

它使用假模型模拟 `model.val()` 输出，确认：

- positive 和 negative 会分别生成评估数据
- 会写出分组 YAML
- 会写出分组 CSV

### `tests/__pycache__/`

测试运行后生成的缓存目录，不是源码。

## 10. 运行产物目录

### `runs/`

所有命令生成的输出都放在这里。

这个目录通常不提交到 Git，因为它包含训练结果、预测图、日志、缓存和中间数据。

### `runs/check/`

`mildew-seg check` 的日志目录。

常见文件：

- `check.log`

### `runs/prepare/`

`mildew-seg prepare` 的日志目录。

常见文件：

- `prepare.log`

### `runs/data/`

数据检查和数据转换后的核心输出目录。

重要文件：

- `audit_report.json`：完整审计报告
- `audit_files.csv`：每张原图的检查摘要
- `audit_issues.csv`：所有 warning/error 明细
- `manifest.csv`：原图级索引，记录每张原图属于 train/val/test 哪一类
- `tiles.csv`：切片级索引，记录每个切片来自哪张原图、属于哪个 split/category
- `data.yaml`：Ultralytics 训练和验证使用的数据配置
- `prepare_summary.json`：数据准备摘要

### `runs/data/dataset/`

转换后的 YOLO-seg 数据集。

结构：

```text
runs/data/dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

其中：

- `images/*`：切片图像
- `labels/*`：YOLO-seg 标签文件
- `*.cache`：Ultralytics 自动生成的数据缓存

### `runs/train/`

训练输出目录。

每次训练会生成一个类似下面的子目录：

```text
runs/train/yolo11n_seg_20260605_133949/
```

常见文件：

- `args.yaml`：本次训练参数
- `results.csv`：每轮训练指标
- `results.png`：训练曲线图
- `labels.jpg`：标签分布可视化
- `Box*.png`：box 指标曲线
- `Mask*.png`：mask 指标曲线
- `confusion_matrix.png`：混淆矩阵
- `train_batch*.jpg`：训练 batch 可视化
- `val_batch*_labels.jpg`：验证 batch 真值图
- `val_batch*_pred.jpg`：验证 batch 预测图
- `weights/best.pt`：最佳模型权重，推理时通常使用这个
- `weights/last.pt`：最后一轮模型权重

### `runs/val/`

验证输出目录。

例如：

```text
runs/val/test/
```

常见文件：

- `slice_metrics.json`：全体切片级 box/mask 指标
- `validation_summary.json`：原图级验证摘要
- `original_predictions.csv`：每张原图的预测统计
- `confusion_matrix.csv`：positive/negative 判定混淆矩阵
- `images/`：原图级预测可视化和 JSON 结果
- `slice_metrics/`：Ultralytics 全量切片验证输出
- `slice_metrics_by_category/`：positive/negative 分开统计的 box/mask 指标

### `runs/val/test/slice_metrics_by_category/`

正负样本分组的切片级指标目录。

常见文件：

- `test_negative_metrics.json`
- `test_positive_metrics.json`
- `test_metrics_by_category.csv`
- `test_negative/`
- `test_positive/`

其中 `test_negative/` 和 `test_positive/` 是 Ultralytics 分别对 negative 和 positive 切片运行 `model.val()` 的完整输出目录。

### `runs/predict/`

单图推理输出目录。

执行：

```powershell
mildew-seg predict --source <图片路径>
```

会在这里生成：

- `<图片名>_prediction.png`
- `<图片名>_result.json`

### `runs/batch/`

批量推理输出目录。

执行：

```powershell
mildew-seg batch --source <图片目录>
```

会在这里生成：

- `results.csv`
- `images/` 下的每张图可视化和 JSON 结果

### `runs/calibration/`

阈值校准输出目录。

常见文件：

- `thresholds.yaml`：最佳阈值
- `search_results.csv`：所有候选阈值组合的评估结果
- `validation_predictions.csv`：验证集预测明细
- `calibration_summary.json`：校准摘要

### `runs/demo/`

Gradio DEMO 的推理输出目录。

上传图像并推理后，结果会保存在这里。

## 11. 常用命令和对应代码

### 数据检查

```powershell
mildew-seg check
```

入口：

```text
src/mildew_seg/cli.py
src/mildew_seg/audit.py
```

### 数据准备

```powershell
mildew-seg prepare
```

入口：

```text
src/mildew_seg/cli.py
src/mildew_seg/prepare.py
```

### 训练

```powershell
mildew-seg train
```

入口：

```text
src/mildew_seg/cli.py
src/mildew_seg/train.py
```

### 验证

```powershell
mildew-seg validate --split test
```

入口：

```text
src/mildew_seg/cli.py
src/mildew_seg/validate.py
```

### 单图推理

```powershell
mildew-seg predict --source "C:\path\to\image.png"
```

入口：

```text
src/mildew_seg/cli.py
src/mildew_seg/inference.py
src/mildew_seg/visualize.py
```

### 批量推理

```powershell
mildew-seg batch --source "C:\path\to\image_folder"
```

入口：

```text
src/mildew_seg/cli.py
src/mildew_seg/batch.py
src/mildew_seg/inference.py
```

### 阈值校准

```powershell
mildew-seg calibrate
```

入口：

```text
src/mildew_seg/cli.py
src/mildew_seg/calibrate.py
src/mildew_seg/thresholds.py
```

### Gradio DEMO

```powershell
mildew-seg demo
```

入口：

```text
src/mildew_seg/cli.py
src/mildew_seg/demo.py
src/mildew_seg/inference.py
```

## 12. 新人应该先看哪些文件

如果只是使用项目，建议顺序：

1. `README.md`
2. `configs/default.yaml`
3. `runs/data/manifest.csv`
4. `runs/train/.../weights/best.pt`

如果要理解代码，建议顺序：

1. `src/mildew_seg/cli.py`
2. `src/mildew_seg/prepare.py`
3. `src/mildew_seg/labelme.py`
4. `src/mildew_seg/geometry.py`
5. `src/mildew_seg/inference.py`
6. `src/mildew_seg/decision.py`
7. `src/mildew_seg/validate.py`

如果要改训练参数，优先看：

```text
configs/default.yaml
src/mildew_seg/train.py
```

如果要改推理逻辑，优先看：

```text
src/mildew_seg/roi.py
src/mildew_seg/inference.py
src/mildew_seg/decision.py
src/mildew_seg/visualize.py
```

如果要改数据转换逻辑，优先看：

```text
src/mildew_seg/labelme.py
src/mildew_seg/geometry.py
src/mildew_seg/prepare.py
```

## 13. 关键设计点

### 只训练一个类别

YOLO 模型只训练 `mildew_spot`，不训练 `positive` 或 `negative`。

`positive/negative` 是原图级标签，用于：

- 数据划分时保持类别比例
- 校准 positive/negative 判定阈值
- 最终评估图像级判定效果

### `seed` 不进入 YOLO 标签

`seed` 只用于：

- 找种子位置
- 生成 ROI
- 计算霉点面积占比

YOLO 标签文件里不会出现 `seed` 类别。

### 为什么要切片

原图尺寸较大，霉点非常小。如果直接把整张图缩放到 YOLO 输入尺寸，霉点会变得更小，难以训练和识别。

因此项目先根据种子 ROI 生成 `320x320` 重叠切片，再以较大的 `imgsz=1024` 训练和推理。

### 为什么要核心区归属

切片之间有重叠，同一个霉点可能出现在多个切片里。

项目通过“切片核心区”规则，只让每个实例归属于一个切片，减少重复标注和重复预测。

### 为什么推理时要自动定位种子

最终用户输入的是完整图像，不一定会提供 LabelMe JSON 或 seed polygon。

因此推理阶段通过蓝色背景和最大连通域自动找到种子区域，再执行切片推理。

## 14. 维护建议

- 不要直接修改 `mildew/` 原始数据，除非你明确是在修正标注。
- 不要把 `runs/` 下的大量训练结果提交到 Git。
- 改动数据转换逻辑后，运行：

```powershell
python -m pytest tests/test_prepare_integration.py tests/test_real_data_regression.py
```

- 改动推理逻辑后，运行：

```powershell
python -m pytest tests/test_inference.py tests/test_roi_decision.py
```

- 改动验证指标逻辑后，运行：

```powershell
python -m pytest tests/test_validate.py
```

- 提交前建议运行：

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
```
