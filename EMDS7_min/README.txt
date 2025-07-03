EMDS7_min 微藻检测数据集说明文档
=====================================

数据集概述
----------
EMDS7_min 是一个用于微藻检测的轻量级数据集，包含6种主要微藻类别的图像和标注数据。
该数据集专为基于YOLOv7的微藻轻量级检测方法设计。

数据集结构
----------
EMDS7_min/
├── images/
│   ├── train/          # 训练集图片 (80%)
│   ├── val/            # 验证集图片 (10%)
│   └── test/           # 测试集图片 (10%)
└── labels_txt/
    ├── train/          # 训练集标签 (YOLO格式)
    ├── val/            # 验证集标签 (YOLO格式)
    └── test/           # 测试集标签 (YOLO格式)

类别信息
--------
数据集包含以下6种微藻类别：

0: Oscillatoria     - 颤藻属
1: Microcystis      - 微囊藻属  
2: sphaerocystis    - 球囊藻属
3: tribonema        - 丝藻属
4: brachionus       - 臂尾轮虫属
5: Ankistrodesmus   - 纤维藻属

标签格式
--------
每个图片对应一个同名的.txt标签文件，采用YOLO格式：
class_id x_center y_center width height

其中：
- class_id: 类别编号 (0-5)
- x_center, y_center: 目标中心点坐标 (归一化到0-1)
- width, height: 目标宽度和高度 (归一化到0-1)

文件命名规则
-----------
图片和标签文件采用统一命名格式：
EMDS7-G{类别编号}-{图片编号}-0400.png
EMDS7-G{类别编号}-{图片编号}-0400.txt

例如：
- EMDS7-G012-113-0400.png (图片)
- EMDS7-G012-113-0400.txt (对应标签)

使用方法
--------
1. 在YOLOv7项目中使用：
   - 配置文件：yolov7/data/emds7_min.yaml
   - 训练命令：python yolov7/train.py --data yolov7/data/emds7_min.yaml --cfg yolov7/cfg/training/yolov7.yaml

2. 数据集划分比例：
   - 训练集：80%
   - 验证集：10% 
   - 测试集：10%

注意事项
--------
1. 确保图片和标签文件一一对应，文件名完全一致
2. 标签坐标已归一化，无需额外处理
3. 图片格式为PNG，分辨率统一
4. 每个标签文件包含一个目标实例

数据统计
--------
- 总图片数量：根据实际划分结果
- 类别数量：6类
- 图片格式：PNG
- 标签格式：YOLO (.txt)

维护信息
--------
- 创建日期：2024年
- 版本：1.0
- 用途：微藻检测模型训练与评估
- 适用框架：YOLOv7
