#!/usr/bin/env python3
"""
环境验证脚本
"""

def verify_environment():
    """验证所有依赖是否正常"""
    print("🔍 验证开发环境...")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        
        from ultralytics import YOLO
        print("✅ YOLOv8: 导入成功")
        
        import sklearn
        print(f"✅ scikit-learn: {sklearn.__version__}")
        
        print("🎉 所有依赖验证通过！")
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False

if __name__ == "__main__":
    verify_environment()