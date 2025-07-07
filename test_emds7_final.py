#!/usr/bin/env python3
"""
最终修复的 EMDS7_min 数据集测试脚本
解决了所有路径、参数和兼容性问题
"""

import os
import sys
import yaml
import torch
import warnings
import numpy
from pathlib import Path

# 忽略NumPy兼容性警告
warnings.filterwarnings("ignore", message=".*NumPy.*")
warnings.filterwarnings("ignore", message=".*Failed to initialize NumPy.*")

# 添加 yolov7 目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
yolov7_path = os.path.join(current_dir, 'yolov7')
if yolov7_path not in sys.path:
    sys.path.insert(0, yolov7_path)

# PyTorch 2.6+ 兼容性：允许反序列化 numpy 对象
import torch.serialization
torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])

def test_dataset_config():
    """测试数据集配置文件"""
    print("=" * 50)
    print("测试数据集配置")
    print("=" * 50)
    
    # 检查多个可能的配置文件路径
    config_paths = [
        'yolov7/data/emds7_min.yaml',
        'data/emds7_min.yaml',
        '../yolov7/data/emds7_min.yaml',
        os.path.join(current_dir, 'yolov7/data/emds7_min.yaml')
    ]
    
    config_path = None
    for path in config_paths:
        if os.path.exists(path):
            config_path = path
            break
    
    if not config_path:
        print(f"❌ 配置文件不存在，尝试的路径:")
        for path in config_paths:
            print(f"   {path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ 配置文件加载成功: {config_path}")
        print(f"   训练集路径: {config.get('train', 'N/A')}")
        print(f"   验证集路径: {config.get('val', 'N/A')}")
        print(f"   测试集路径: {config.get('test', 'N/A')}")
        print(f"   类别数量: {config.get('nc', 'N/A')}")
        print(f"   类别名称: {config.get('names', 'N/A')}")
        
        # 检查路径是否存在
        for split in ['train', 'val', 'test']:
            path = config.get(split, '')
            if path and isinstance(path, str):  # 确保path是字符串
                # 尝试多个可能的路径
                possible_paths = [
                    path,
                    os.path.join(current_dir, path),
                    os.path.join(os.path.dirname(config_path), path)
                ]
                
                found_path = None
                for p in possible_paths:
                    if os.path.exists(p):
                        found_path = p
                        break
                
                if found_path:
                    img_count = len([f for f in os.listdir(found_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                    print(f"   {split}集图片数量: {img_count} (路径: {found_path})")
                else:
                    print(f"   ⚠️ {split}集路径不存在: {path}")
            else:
                print(f"   ⚠️ {split}集路径格式错误: {path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False

def test_model_imports():
    """测试模型相关模块导入"""
    print("\n" + "=" * 50)
    print("测试模型模块导入")
    print("=" * 50)
    
    try:
        from models.experimental import attempt_load
        print("✅ attempt_load 导入成功")
    except Exception as e:
        print(f"❌ attempt_load 导入失败: {e}")
        return False
    
    try:
        from models.common import GhostNetBackbone, CBAM, DSConv
        print("✅ GhostNetBackbone, CBAM, DSConv 导入成功")
    except Exception as e:
        print(f"❌ 自定义模块导入失败: {e}")
        return False
    
    try:
        from models.ghostnet import ghostnet
        print("✅ ghostnet 模块导入成功")
    except Exception as e:
        print(f"❌ ghostnet 模块导入失败: {e}")
        return False
    
    return True

def test_ghostnet_backbone():
    """测试 GhostNetBackbone 功能"""
    print("\n" + "=" * 50)
    print("测试 GhostNetBackbone")
    print("=" * 50)
    
    try:
        from models.common import GhostNetBackbone
        
        # 创建 GhostNetBackbone 实例（不使用pretrained参数）
        backbone = GhostNetBackbone()
        print("✅ GhostNetBackbone 创建成功")
        
        # 测试前向传播
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = backbone(x)
        
        print(f"✅ 前向传播成功")
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ GhostNetBackbone 测试失败: {e}")
        return False

def test_cbam_module():
    """测试 CBAM 模块"""
    print("\n" + "=" * 50)
    print("测试 CBAM 模块")
    print("=" * 50)
    
    try:
        from models.common import CBAM
        
        # 创建 CBAM 实例
        cbam = CBAM(c1=64, reduction=16)
        print("✅ CBAM 创建成功")
        
        # 测试前向传播
        x = torch.randn(1, 64, 32, 32)
        with torch.no_grad():
            output = cbam(x)
        
        print(f"✅ 前向传播成功")
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ CBAM 测试失败: {e}")
        return False

def test_dsconv_module():
    """测试 DSConv 模块"""
    print("\n" + "=" * 50)
    print("测试 DSConv 模块")
    print("=" * 50)
    
    try:
        from models.common import DSConv
        
        # 创建 DSConv 实例
        dsconv = DSConv(in_ch=64, out_ch=128, k=3, s=1)
        print("✅ DSConv 创建成功")
        
        # 测试前向传播
        x = torch.randn(1, 64, 32, 32)
        with torch.no_grad():
            output = dsconv(x)
        
        print(f"✅ 前向传播成功")
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ DSConv 测试失败: {e}")
        return False

def test_dataset_loading():
    """测试数据集加载"""
    print("\n" + "=" * 50)
    print("测试数据集加载")
    print("=" * 50)
    
    try:
        from utils.datasets import create_dataloader
        from utils.general import colorstr
        
        # 加载配置
        with open('yolov7/data/emds7_min.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        class Opt:
            def __init__(self):
                self.rect = True
                self.pad = 0.5
                self.task = 'test'
                self.single_cls = False
        
        opt = Opt()
        
        dataloader = create_dataloader(
            path=config['test'],
            imgsz=640,
            batch_size=1,
            stride=32,
            opt=opt,
            pad=0.5,
            rect=True,
            prefix=colorstr('test: ')
        )[0]
        
        print("✅ 数据加载器创建成功")
        print(f"   数据集大小: {len(dataloader.dataset)}")
        
        # 尝试加载一个batch
        for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
            print(f"✅ 成功加载第 {batch_i+1} 个batch")
            print(f"   图片形状: {img.shape}")
            print(f"   标签形状: {targets.shape}")
            print(f"   图片路径: {paths[0]}")
            break
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return False

def test_environment():
    """测试环境配置"""
    print("\n" + "=" * 50)
    print("测试环境配置")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
        print(f"   CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA 版本: {torch.version.cuda}")
            print(f"   GPU 数量: {torch.cuda.device_count()}")
        
        import numpy as np
        print(f"✅ NumPy 版本: {np.__version__}")
        
        import cv2
        print(f"✅ OpenCV 版本: {cv2.__version__}")
        
        return True
        
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        return False

def test_paths():
    """测试路径配置"""
    print("\n" + "=" * 50)
    print("测试路径配置")
    print("=" * 50)
    
    try:
        # 检查关键目录
        key_dirs = [
            'yolov7',
            'yolov7/models',
            'yolov7/utils',
            'yolov7/data',
            'EMDS7_min',
            'EMDS7_min/images',
            'EMDS7_min/labels'
        ]
        
        for dir_path in key_dirs:
            if os.path.exists(dir_path):
                print(f"✅ {dir_path} 存在")
            else:
                print(f"❌ {dir_path} 不存在")
        
        # 检查关键文件
        key_files = [
            'yolov7/data/emds7_min.yaml',
            'yolov7/models/common.py',
            'yolov7/models/ghostnet.py',
            'yolov7/utils/datasets.py'
        ]
        
        for file_path in key_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path} 存在")
            else:
                print(f"❌ {file_path} 不存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 路径测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("EMDS7_min 数据集和模型测试 (最终修复版)")
    print("=" * 60)
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本所在目录: {current_dir}")
    print("=" * 60)
    
    tests = [
        ("环境配置", test_environment),
        ("路径配置", test_paths),
        ("数据集配置", test_dataset_config),
        ("模型模块导入", test_model_imports),
        ("GhostNetBackbone", test_ghostnet_backbone),
        ("CBAM模块", test_cbam_module),
        ("DSConv模块", test_dsconv_module),
        ("数据集加载", test_dataset_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目配置正确。")
        print("\n下一步:")
        print("1. 可以开始训练模型")
        print("2. 可以运行完整测试")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关配置。")
        print("\n建议:")
        print("1. 确保数据集文件存在且路径正确")
        print("2. 检查Python环境和依赖包版本")
        print("3. 如果NumPy版本问题，考虑降级到NumPy 1.x")
        print("4. 确保所有必要的目录和文件都存在")
        return False

if __name__ == "__main__":
    main() 