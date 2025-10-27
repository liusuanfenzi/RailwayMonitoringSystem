"""
YOLO检测器测试
"""

import cv2
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detectors.yolo_detector import YOLODetector

def test_yolo_detector():
    """测试YOLO检测器"""
    print("🧪 测试YOLO检测器...")
    
    # 创建检测器
    detector = YOLODetector()
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # 测试检测
    detections = detector.detect(test_image)
    print(f"检测到 {len(detections)} 个目标")
    
    # 测试姿态检测
    poses = detector.detect_poses(test_image)
    print(f"检测到 {len(poses)} 个人体姿态")
    
    print("✅ YOLO检测器测试完成")

if __name__ == "__main__":
    test_yolo_detector()
