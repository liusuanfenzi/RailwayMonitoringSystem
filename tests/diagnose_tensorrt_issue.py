# diagnose_tensorrt_issue.py
import torch
import tensorrt as trt
import os
import subprocess
import sys

def diagnose_tensorrt_issue():
    print("🔍 TensorRT问题诊断")
    print("=" * 50)
    
    # 1. 检查版本兼容性
    print("1. 版本兼容性检查:")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   PyTorch CUDA版本: {torch.version.cuda}")
    print(f"   TensorRT版本: {trt.__version__}")
    
    # 检查CUDA工具包版本
    try:
        cuda_version = subprocess.check_output(["nvcc", "--version"]).decode()
        cuda_lines = cuda_version.split('\n')
        for line in cuda_lines:
            if "release" in line:
                print(f"   系统CUDA版本: {line.strip()}")
                break
    except:
        print("   无法获取系统CUDA版本")
    
    # 2. 检查TensorRT安装状态
    print("\n2. TensorRT安装状态:")
    try:
        # 检查TensorRT库路径
        trt_path = trt.__file__
        print(f"   TensorRT Python包路径: {trt_path}")
        
        # 检查TensorRT插件
        plugin_paths = [
            '/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so',
            '/usr/local/cuda/lib64/libnvinfer_plugin.so'
        ]
        for path in plugin_paths:
            if os.path.exists(path):
                print(f"   ✅ TensorRT插件存在: {path}")
            else:
                print(f"   ❌ TensorRT插件缺失: {path}")
                
    except Exception as e:
        print(f"   TensorRT检查失败: {e}")
    
    # 3. 检查PyTorch-TensorRT集成
    print("\n3. PyTorch-TensorRT集成:")
    try:
        import torch_tensorrt
        print(f"   torch_tensorrt版本: {torch_tensorrt.__version__}")
    except ImportError:
        print("   ❌ torch_tensorrt未安装")
    except Exception as e:
        print(f"   torch_tensorrt错误: {e}")
    
    # 4. 检查Ultralytics的TensorRT配置
    print("\n4. Ultralytics TensorRT配置:")
    try:
        from ultralytics import YOLO
        # 检查YOLO是否尝试自动使用TensorRT
        model = YOLO('yolov8n.pt')
        print(f"   模型设备: {model.device}")
        print(f"   模型类型: {type(model.model)}")
    except Exception as e:
        print(f"   Ultralytics检查失败: {e}")
    
    # 5. 检查环境变量
    print("\n5. 环境变量检查:")
    tensorrt_vars = {k: v for k, v in os.environ.items() if 'TRT' in k or 'TENSORRT' in k}
    for k, v in tensorrt_vars.items():
        print(f"   {k}: {v}")
    
    if not tensorrt_vars:
        print("   未找到TensorRT相关环境变量")
    
    # 6. 检查Jetson特定配置
    print("\n6. Jetson特定检查:")
    try:
        # 检查JetPack版本
        jetpack_info = subprocess.check_output(["cat", "/etc/nv_tegra_release"]).decode()
        print(f"   JetPack信息: {jetpack_info.split()[0] if jetpack_info else '未知'}")
    except:
        print("   无法获取JetPack信息")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    diagnose_tensorrt_issue()