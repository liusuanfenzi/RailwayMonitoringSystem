# check_python_tensorrt.py
import tensorrt as trt
import torch
import os

def check_python_tensorrt():
    print("🔍 Python环境TensorRT检查")
    print("=" * 50)
    
    # TensorRT信息
    print(f"Python TensorRT版本: {trt.__version__}")
    print(f"TensorRT路径: {trt.__file__}")
    
    # 检查构建器
    logger = trt.Logger(trt.Logger.WARNING)
    try:
        builder = trt.Builder(logger)
        print(f"TensorRT Builder: ✅ 可用")
        
        # 检查插件
        registry = trt.get_plugin_registry()
        print(f"已注册插件数量: {registry.num_plugins}")
        
    except Exception as e:
        print(f"TensorRT Builder: ❌ 不可用 - {e}")
    
    # PyTorch信息
    print(f"\nPyTorch版本: {torch.__version__}")
    print(f"PyTorch CUDA版本: {torch.version.cuda}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # 检查环境变量
    print(f"\n环境变量:")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '未设置')}")
    print(f"TENSORRT_PATH: {os.environ.get('TENSORRT_PATH', '未设置')}")

if __name__ == "__main__":
    check_python_tensorrt()