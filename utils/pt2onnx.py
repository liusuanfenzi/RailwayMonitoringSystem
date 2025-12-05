# create_onnx_model_jetson.py
import os
import sys

print("🔄 在Jetson上导出动态ONNX模型...")

try:
    from ultralytics import YOLO
    
    # 检查模型文件
    if not os.path.exists('yolov8n.pt'):
        print("📥 下载YOLOv8n模型...")
        model = YOLO('yolov8n.pt')
    else:
        print("📁 加载现有模型...")
        model = YOLO('yolov8n.pt')
    
    # 导出动态ONNX
    print("🔧 导出动态ONNX模型...")
    model.export(
        format='onnx', 
        dynamic=True,
        imgsz=[480, 640],
        half=False,
        device='cpu'
    )
    
    print("✅ 动态ONNX导出成功！")
    print("📁 文件: yolov8n.onnx")
    print("📐 支持动态尺寸: 480x480 和 640x640")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("💡 请安装 ultralytics: pip install ultralytics")
except Exception as e:
    print(f"❌ 导出失败: {e}")