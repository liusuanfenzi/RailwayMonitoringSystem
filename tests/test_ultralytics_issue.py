# test_ultralytics_issue.py
import os
import torch
from ultralytics import YOLO

print("=== Ultralytics YOLO TensorRT问题验证 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 方法1: 尝试各种禁用TensorRT的方式
print("\n1. 尝试禁用TensorRT...")
os.environ['TRT_ENGINE_CACHE_PATH'] = ''
os.environ['TORCH_BACKEND'] = 'PYTORCH'

model = YOLO('yolov8n.pt')
print(f"模型设备: {model.device}")

# 方法2: 强制使用CPU
print("\n2. 强制使用CPU...")
try:
    results = model.predict(torch.randn(1, 3, 640, 640), device='cpu', verbose=True)
    print("✅ CPU模式成功")
except Exception as e:
    print(f"❌ CPU模式失败: {e}")

# 方法3: 检查模型内部状态
print("\n3. 检查模型内部...")
print(f"模型类型: {type(model.model)}")
print(f"模型设备: {next(model.model.parameters()).device}")