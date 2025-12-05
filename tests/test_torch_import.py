# test_torch_import.py
import sys
print(f"Python路径: {sys.executable}")
print(f"Python版本: {sys.version}")

# 检查 torch
try:
    import torch
    print(f"✅ torch 版本: {torch.__version__}")
    print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ 设备: {torch.cuda.get_device_name()}")
except ImportError as e:
    print(f"❌ torch 导入失败: {e}")

# 检查 torchvision
try:
    import torchvision
    print(f"✅ torchvision 版本: {torchvision.__version__}")
except ImportError as e:
    print(f"❌ torchvision 导入失败: {e}")

# 检查 ultralytics
try:
    import ultralytics
    print(f"✅ ultralytics 版本: {ultralytics.__version__}")
except ImportError as e:
    print(f"❌ ultralytics 导入失败: {e}")