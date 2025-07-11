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
最新训练结果（2025-07-11）
训练集：700张图片，测试集：300张图片，6个类别。
训练300个epoch，模型成功收敛，权重文件已保存。
验证集评估结果（部分指标）

![train_batch2](https://github.com/user-attachments/assets/bfcda473-b667-4913-8b0a-5411c1bf9747)
![train_batch1](https://github.com/user-attachments/assets/b4075346-5cf3-4f15-a97b-fcf7e91a872c)
![train_batch0](https://github.com/user-attachments/assets/022789fc-9aef-41ed-9c2c-097591091b41)
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/54220459-27f7-4926-ba03-4cae118f707c" />


[results.txt](https://github.com/user-attachments/files/21173680/results.txt)
    

   | 指标                           | 变化趋势                      | 说明                            |
| ---------------------------- | ------------------------- | ----------------------------- |
| **Box Loss**                 | 逐步下降，趋于稳定                 | 模型对目标边界框预测逐步精确                |
| **Objectness Loss**          | 开始急剧下降，后期略上升              | 表示网络识别物体与否逐步优化，但后期可能过拟合或学习率过高 |
| **Classification Loss**      | 明显下降                      | 模型对类别判别能力增强                   |
| **Precision / Recall**       | 明显上升                      | 检测效果变好，召回率有提升                 |
| **mAP\@0.5 / mAP\@0.5:0.95** | 呈现明显上升趋势，最终约为 0.13 和 0.15 | 模型检测准确率逐步提高，但精度仍偏低            |


##图片可视化分析（推理检测结果）
从 train_batch0.jpg 到 train_batch2.jpg 可以观察到：

背景光照复杂，图像对比度低，目标较小。

模型基本能识别出不同种类微藻，用不同颜色框标注。

部分图片中出现误检或漏检（如目标密集时框重叠、粘连），说明检测能力尚可，但仍有提升空间。


##训练日志分析（results.txt）
查看部分关键指标随 epoch 变化：

项目	说明
Box Loss 从 0.1685 降到约 0.095 左右	说明定位能力有所提升
Precision/Recall 分别达到 0.5+/0.55+	初步具备目标识别能力
mAP@0.5:0.95 最终大约 0.15	在多类别和复杂背景下，识别效果尚可，但与工业应用标准（如 0.3~0.5）仍有差距
未使用验证集进行 early stopping 或 checkpoint 策略	这可能影响最终收敛质量



## 结果分析
- 训练日志、loss/mAP曲线：`runs/train/exp*/results.png`
- 最优模型权重：`runs/train/exp*/weights/best.pt`
- 详细指标：`runs/train/exp*/results.txt`









