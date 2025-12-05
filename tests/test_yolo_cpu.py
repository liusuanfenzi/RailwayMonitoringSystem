# test_yolo_cpu.py
from ultralytics import YOLO
import cv2
import numpy as np

print("测试YOLO CPU模式...")
model = YOLO('yolov8n.pt')

# 强制使用CPU
test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

print("开始CPU推理测试...")
try:
    results = model(test_img, verbose=True, device='cpu')  # 强制CPU
    print("✅ CPU模式YOLO推理成功")
    print(f"检测结果: {len(results)}")
except Exception as e:
    print(f"❌ CPU模式也失败: {e}")