# 基于改进YOLOv7的微藻轻量级检测方法

## 项目简介
本项目旨在实现高通量微藻细胞检测，针对小目标、运动模糊、多尺度、复杂背景等实际挑战，基于YOLOv7/YOLOv8等主流目标检测框架，结合轻量化网络（如GhostNet、CBAM）进行算法优化。项目最初基于 EMDS7_min 数据集，后适配了 Vision Meets Algae 2023 (VisAlgae2023) 数据集。

---


## 项目结构

```
基于改进YOLOv7的微藻轻量级检测方法/
├── yolov7/                    # YOLOv7 主代码（含自定义模块）
│   ├── models/
│   │   ├── common.py         # GhostNetBackbone, CBAM, DSConv 等自定义模块
│   │   ├── ghostnet.py       # GhostNet 轻量主干实现
│   │   └── yolo.py           # YOLO 检测头与主结构
│   ├── data/
│   │   └── emds7_min.yaml    # 数据集配置
│   ├── utils/                # 工具函数与数据增强
│   ├── train.py              # 官方训练脚本
│   ├── test.py               # 官方测试脚本
│   └── detect.py             # 推理脚本
├── 2023_dataset/                # 微藻数据集
│   ├── images/               # 图片（train/test）
│   └── labels/               # 标签（YOLO格式，train/test）
├── train_emds7.py            # 定制化训练脚本
├── test_emds7_final.py       # 定制化测试脚本
├── README.md                 # 项目说明（本文件）
└── runs/                     # 训练与测试输出
```

## 数据集说明
#### 数据集配置文件示例（yolov7/data/visalgae2023.yaml）
```yaml
train: 2023_dataset/train/images
val: 2023_dataset/test/images
nc: 6
names: [Platymonas, Chlorella, "Dunaliella salina", Effrenium, Porphyridium, Haematococcus]
```
- **VisAlgae2023**：6类微藻，YOLO格式标注，图片与标签一一对应。
  - 目录结构示例：
    ```
    2023_dataset/
      ├── train/
      │   ├── images/
      │   └── labels/
      └── test/
          ├── images/
          └── labels/
    ```
- **标签格式**：每行 `class x_center y_center width height`，均为归一化浮点数。
- **类别名**：
  0: Platymonas  1: Chlorella  2: Dunaliella salina  3: Effrenium  4: Porphyridium  5: Haematococcus

---

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


## 模型结构

以 YOLOv7 为主干，支持 GhostNet、CBAM 等轻量化和注意力机制模块。
训练脚本（如 train_emds7.py）支持灵活配置模型结构、超参数、数据集路径。

训练流程
训练流程标准化，支持命令行参数配置，自动保存最优权重（best.pt）和最后权重（last.pt）。
训练日志、超参数、配置参数均自动保存，便于复现和对比实验。
训练过程自动生成 loss、mAP、Precision、Recall 等曲线图（results.png），并支持 TensorBoard 可视化。

标签与数据增强检查
增强了标签检查脚本，确保所有图片和标签一一对应，标签内容无格式错误。
检查并修正了数据增强、标签格式等潜在问题。





#### 训练
```bash
python yolov7/train.py --data yolov7/data/visalgae2023.yaml --cfg yolov7/cfg/training/yolov7.yaml --weights '' --device 0
```

#### 测试
```bash
python yolov7/test.py --data yolov7/data/visalgae2023.yaml --weights runs/train/exp/weights/best.pt --device 0
```

#### 推理
```bash
python yolov7/detect.py --weights runs/train/exp/weights/best.pt --source 2023_dataset/test/images --img 640 --conf 0.25 --device 0
```


##训练与评估结果
最新训练结果（2025-07-10）
训练集：700张图片，测试集：300张图片，6个类别。
训练50个epoch，模型成功收敛，权重文件已保存。
验证集评估结果（部分指标）

     Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95
     all         300        1627       0.505       0.242       0.198       0.091

    结果文件说明
    results.png：训练过程曲线图。
    results.txt：每个epoch详细指标。
    train_batch*.jpg：训练样本可视化。
    weights/best.pt：最优模型权重。
    opt.yaml/hyp.yaml：本次实验参数与超参数。


## 结果分析
- 训练日志、loss/mAP曲线：`runs/train/exp*/results.png`
- 最优模型权重：`runs/train/exp*/weights/best.pt`
- 详细指标：`runs/train/exp*/results.txt`









