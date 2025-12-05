# 创建一个简单的测试脚本 test_tensorrt.py
import tensorrt as trt
print(f"TensorRT版本: {trt.__version__}")

import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")

# 测试基本的TensorRT功能
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
print("✅ TensorRT Builder创建成功")