# final_check.py
import torch
import psutil

def final_system_check():
    print("🎯 最终系统检查")
    print("=" * 40)
    
    # 1. 内存检查
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"✅ 物理内存: {memory.available / (1024**3):.1f}GB 可用")
    print(f"✅ 交换空间: {swap.total / (1024**3):.1f}GB 总量")
    
    # 2. GPU检查
    if torch.cuda.is_available():
        print(f"✅ GPU可用: {torch.cuda.get_device_name()}")
        print(f"✅ CUDA内存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    else:
        print("❌ GPU不可用")
    
    # 3. TensorRT检查
    try:
        import tensorrt
        print(f"✅ TensorRT: {tensorrt.__version__}")
    except ImportError:
        print("❌ TensorRT导入失败")
    
    # 4. 建议
    if memory.available < 1.0:  # 1GB
        print("⚠️ 警告: 物理内存较低，将依赖交换空间")
    else:
        print("✅ 系统状态良好，可以运行TensorRT demo")

if __name__ == "__main__":
    final_system_check()