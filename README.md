# EMDS7_min 数据集测试指南

## 概述

本指南介绍如何测试基于改进YOLOv7的微藻轻量级检测方法项目，包括数据集验证、模型测试和性能评估。

## 项目结构

```
基于改进YOLOv7的微藻轻量级检测方法/
├── yolov7/                    # YOLOv7 核心代码
│   ├── models/
│   │   ├── common.py         # 包含 GhostNetBackbone, CBAM, DSConv
│   │   ├── ghostnet.py       # GhostNet 实现
│   │   └── yolo.py           # YOLO 模型定义
│   ├── data/
│   │   └── emds7_min.yaml    # EMDS7_min 数据集配置
│   └── utils/                # 工具函数
├── EMDS7_min/                # 数据集目录
│   ├── images/
│   │   ├── train/           # 训练图片
│   │   ├── val/             # 验证图片
│   │   └── test/            # 测试图片
│   └── labels_txt/
│       ├── train/           # 训练标签
│       ├── val/             # 验证标签
│       └── test/            # 测试标签
├── test_emds7.py            # 完整测试脚本
├── test_emds7_simple.py     # 简化测试脚本
└── TESTING_GUIDE.md         # 本指南
```

## 改进的模型组件

### 1. GhostNetBackbone
- 替换原始YOLOv7的backbone
- 提供更轻量级的特征提取
- 位置：`yolov7/models/common.py`

### 2. CBAM (Convolutional Block Attention Module)
- 通道和空间注意力机制
- 提升特征表示能力
- 位置：`yolov7/models/common.py`

### 3. DSConv (Depthwise Separable Convolution)
- 深度可分离卷积
- 减少计算量和参数数量
- 位置：`yolov7/models/common.py`

## 测试步骤

### 步骤 1: 环境检查

确保安装了必要的依赖：

```bash
pip install torch torchvision
pip install opencv-python
pip install pyyaml
pip install tqdm
pip install matplotlib
pip install seaborn
```

### 步骤 2: 基础测试

运行简化测试脚本验证基本配置：

```bash
python test_emds7_simple.py
```

这个脚本会检查：
- ✅ 数据集配置文件
- ✅ 模型模块导入
- ✅ GhostNetBackbone 功能
- ✅ CBAM 模块功能
- ✅ DSConv 模块功能
- ✅ 数据集加载

### 步骤 3: 完整测试

如果基础测试通过，运行完整测试：

```bash
# 基本测试
python test_emds7.py --weights yolov7.pt --data yolov7/data/emds7_min.yaml

# 详细测试（显示每个类别结果）
python test_emds7.py --weights yolov7.pt --data yolov7/data/emds7_min.yaml --verbose

# 保存预测结果
python test_emds7.py --weights yolov7.pt --data yolov7/data/emds7_min.yaml --save-txt

# 自定义参数
python test_emds7.py \
    --weights yolov7.pt \
    --data yolov7/data/emds7_min.yaml \
    --batch-size 8 \
    --img-size 640 \
    --conf-thres 0.25 \
    --iou-thres 0.45 \
    --device 0 \
    --save-dir runs/test_emds7_custom
```

## 测试参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | yolov7.pt | 模型权重文件路径 |
| `--data` | yolov7/data/emds7_min.yaml | 数据集配置文件 |
| `--batch-size` | 16 | 批次大小 |
| `--img-size` | 640 | 输入图片尺寸 |
| `--conf-thres` | 0.25 | 置信度阈值 |
| `--iou-thres` | 0.45 | NMS IOU阈值 |
| `--device` | 0 | 设备选择 (0,1,2,3 或 cpu) |
| `--save-txt` | False | 保存预测结果到txt文件 |
| `--save-conf` | False | 保存置信度 |
| `--verbose` | False | 详细输出每个类别结果 |
| `--trace` | False | 使用TorchScript追踪模型 |
| `--half` | True | 使用半精度推理 |

## 数据集信息

### EMDS7_min 数据集
- **类别数量**: 6
- **类别名称**: 
  - 0: Oscillatoria (颤藻)
  - 1: Microcystis (微囊藻)
  - 2: sphaerocystis (球囊藻)
  - 3: tribonema (黄丝藻)
  - 4: brachionus (臂尾轮虫)
  - 5: Ankistrodesmus (纤维藻)

### 数据集结构
```
EMDS7_min/
├── images/
│   ├── train/     # 训练图片
│   ├── val/       # 验证图片
│   └── test/      # 测试图片
└── labels_txt/
    ├── train/     # 训练标签 (YOLO格式)
    ├── val/       # 验证标签
    └── test/      # 测试标签
```

### 标签格式
YOLO格式：`class_id x_center y_center width height`
- 所有值都是相对于图片尺寸的归一化值 (0-1)
- 每行一个目标

## 预期输出

### 测试结果示例
```
EMDS7_min 数据集测试结果
================================================================================
                Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95
                   all        225        450      0.856      0.823      0.847      0.456

各类别详细结果:
--------------------------------------------------------------------------------
          Oscillatoria        225         75      0.892      0.867      0.891      0.523
           Microcystis        225        150      0.845      0.813      0.834      0.445
        sphaerocystis        225         50      0.823      0.780      0.812      0.423
           tribonema        225         50      0.867      0.840      0.856      0.478
          brachionus        225         75      0.834      0.800      0.825      0.432
       Ankistrodesmus        225         50      0.856      0.820      0.843      0.456

速度: 12.3/2.1/14.4 ms 推理/NMS/总计 per 640x640 图片 at batch-size 16
```

### 保存的文件
- `runs/test_emds7/` - 测试结果目录
  - `confusion_matrix.png` - 混淆矩阵
  - `test_batch0_labels.jpg` - 真实标签可视化
  - `test_batch0_pred.jpg` - 预测结果可视化
  - `labels/` - 预测标签文件 (如果使用 --save-txt)

## 常见问题

### 1. 模块导入错误
**问题**: `ModuleNotFoundError: No module named 'models'`
**解决**: 确保在项目根目录运行脚本，或检查 `sys.path.append('yolov7')` 是否正确

### 2. 数据集路径错误
**问题**: 找不到数据集文件
**解决**: 检查 `yolov7/data/emds7_min.yaml` 中的路径是否正确

### 3. 模型权重不存在
**问题**: 找不到权重文件
**解决**: 确保权重文件存在，或先训练模型

### 4. CUDA内存不足
**问题**: `RuntimeError: CUDA out of memory`
**解决**: 减小 `--batch-size` 或使用 `--device cpu`

### 5. 图片格式不支持
**问题**: 无法加载某些图片
**解决**: 确保图片格式为 jpg, png, jpeg

## 性能优化建议

1. **使用GPU**: 设置 `--device 0` 使用GPU加速
2. **半精度推理**: 默认启用，可显著减少内存使用
3. **批量大小**: 根据GPU内存调整 `--batch-size`
4. **图片尺寸**: 较小的 `--img-size` 可提高速度但可能影响精度

## 模型训练

在测试之前，需要先训练模型：

```bash
# 训练命令示例
python yolov7/train.py \
    --data yolov7/data/emds7_min.yaml \
    --cfg yolov7/cfg/training/yolov7.yaml \
    --weights yolov7/weights/yolov7.pt \
    --batch-size 16 \
    --epochs 100 \
    --device 0
```

## 联系信息

如有问题，请检查：
1. 数据集是否正确准备
2. 模型权重是否存在
3. 环境依赖是否安装完整
4. 路径配置是否正确

---

**注意**: 本测试脚本专门为EMDS7_min数据集和改进的YOLOv7模型设计，确保与项目架构完全兼容。 