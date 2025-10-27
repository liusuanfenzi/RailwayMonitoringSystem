#!/usr/bin/env python3
"""
自动创建完整版铁路监控项目目录结构
"""

import os
from pathlib import Path

def create_complete_project_structure():
    """创建完整版项目目录结构"""
    
    # 基础目录
    base_dir = Path("D:/railway_monitoring")
    
    # 完整目录结构
    directories = [
        # 模型权重目录
        "weights/pretrained",
        "weights/custom",
        "weights/deepsort",
        
        # 模型定义目录
        "models/detectors",
        "models/trackers", 
        "models/background_subtractors",
        "models/pose_estimators",
        
        # 工具函数目录
        "utils/video",
        "utils/image",
        "utils/geometry",
        "utils/logger",
        "utils/notifications",
        
        # 配置文件目录
        "config/cameras",
        "config/models",
        "config/system",
        
        # 测试相关目录
        "tests/unit",
        "tests/integration",
        "tests/data",
        
        # 数据目录
        "data/test_videos",
        "data/training",
        "data/validation", 
        "data/annotations",
        
        # 输出目录
        "outputs/detections",
        "outputs/tracks",
        "outputs/alerts",
        "outputs/logs",
        "outputs/exports",
        "outputs/visualizations",
        
        # 文档目录
        "docs/api",
        "docs/tutorials",
        "docs/images",
        
        # 脚本目录
        "scripts/deployment",
        "scripts/training",
        "scripts/utilities",
        
        # 源代码目录
        "src/core",
        "src/processing",
        "src/analysis",
        "src/interface",
        
        # 其他目录
        "notebooks",
        "assets",
        "backups",
        "docker"
    ]
    
    print("🚀 开始创建完整版铁路监控项目结构...")
    print("=" * 60)
    
    # 创建所有目录
    for dir_path in directories:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建目录: {dir_path}")
    
    # 创建基础Python文件
    python_files = {
        # 根目录文件
        "main.py": """#!/usr/bin/env python3
\"\"\"
铁路监控系统主程序
\"\"\"

import argparse
from src.core.system_controller import SystemController

def main():
    parser = argparse.ArgumentParser(description='铁路智能监控系统')
    parser.add_argument('--config', default='config/system/main.yaml', help='配置文件路径')
    parser.add_argument('--camera', help='指定相机ID')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 启动系统
    controller = SystemController(args.config)
    controller.start()
    
    print(\"🚄 铁路监控系统启动成功！\")

if __name__ == \"__main__\":
    main()
""",

        "requirements.txt": """# 铁路监控系统依赖包

# 基础科学计算
numpy>=1.24.3
opencv-python>=4.8.1.78
matplotlib>=3.7.2
scipy>=1.11.3
scikit-learn>=1.3.0
pandas>=2.0.3

# 深度学习框架
torch>=2.0.1
torchvision>=0.15.2
ultralytics>=8.0.186

# 图像处理
Pillow>=10.0.0
scikit-image>=0.21.0

# 目标跟踪
filterpy>=1.4.5
deep-sort-realtime>=1.3.2

# 配置管理
PyYAML>=6.0
configparser>=5.3.0

# 日志和工具
loguru>=0.7.0
tqdm>=4.65.0
click>=8.1.0

# Web界面
flask>=2.3.0
flask-socketio>=5.3.0
flask-cors>=4.0.0

# 测试和开发
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
jupyter>=1.0.0
""",

        "README.md": """# 铁路智能监控系统

## 项目简介
基于计算机视觉的铁路场景智能监控系统，实现列车进出站分析、道口杂物检测、人员手势识别等功能。

## 项目结构
railway_monitoring/
├── weights/ # 模型权重文件
├── models/ # 模型定义
├── utils/ # 工具函数
├── config/ # 配置文件
├── src/ # 源代码
├── data/ # 数据文件
├── outputs/ # 输出结果
├── tests/ # 测试代码
├── docs/ # 文档
├── scripts/ # 工具脚本
└── notebooks/ # Jupyter笔记本


## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 配置相机参数：编辑 `config/cameras/` 下的配置文件
3. 运行系统：`python main.py --config config/system/main.yaml`

## 功能模块
- 列车进出站检测
- 道口杂物检测  
- 人员手势识别
- 人员/车辆停留检测
- 违规使用手机检测
""",

        # 配置文件
        "config/system/main.yaml": """# 系统主配置

system:
  name: "铁路智能监控系统"
  version: "1.0.0"
  debug: false
  
logging:
  level: "INFO"
  file: "outputs/logs/system.log"
  max_size: "10MB"
  backup_count: 5

cameras:
  - id: "camera_1"
    enabled: true
    type: "gesture_detection"
  - id: "camera_2" 
    enabled: true
    type: "train_detection"

modules:
  train_detection:
    enabled: true
    confidence_threshold: 0.7
  gesture_detection:
    enabled: true  
    gesture_sequence: ["left", "right", "forward"]
""",

        "config/models/yolo.yaml": """# YOLO模型配置

yolov8:
  detection:
    model: "yolov8n.pt"
    confidence: 0.5
    iou_threshold: 0.45
    classes: [0, 1, 2, 3, 5]  # person, bicycle, car, motorcycle, bus
    
  pose:
    model: "yolov8n-pose.pt"
    confidence: 0.6
    keypoints_confidence: 0.5

training:
  img_size: 640
  batch_size: 16
  epochs: 100
  patience: 10
""",

        # 模型文件
        "models/__init__.py": """\"\"\"
模型定义包
\"\"\"

from .detectors.yolo_detector import YOLODetector
from .trackers.sort_tracker import SORTTracker
from .background_subtractors.gmm_model import GMMBackgroundSubtractor

__all__ = [
    'YOLODetector',
    'SORTTracker', 
    'GMMBackgroundSubtractor'
]
""",

        "models/detectors/yolo_detector.py": """\"\"\"
YOLO目标检测器封装
\"\"\"

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any

class YOLODetector:
    \"\"\"YOLO目标检测器\"\"\"
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.class_names = self.model.names
        
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        \"\"\"检测图像中的目标\"\"\"
        results = self.model(image, conf=self.confidence, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy(),  # [x1, y1, x2, y2]
                        'confidence': box.conf[0].cpu().numpy(),
                        'class_id': int(box.cls[0].cpu().numpy()),
                        'class_name': self.class_names[int(box.cls[0].cpu().numpy())]
                    }
                    detections.append(detection)
                    
        return detections
    
    def detect_poses(self, image: np.ndarray) -> List[Dict[str, Any]]:
        \"\"\"人体姿态检测\"\"\"
        results = self.model(image, conf=self.confidence, verbose=False)
        
        poses = []
        for result in results:
            keypoints = result.keypoints
            if keypoints is not None:
                for kp in keypoints:
                    pose = {
                        'keypoints': kp.xy[0].cpu().numpy(),
                        'confidences': kp.conf[0].cpu().numpy()
                    }
                    poses.append(pose)
                    
        return poses
""",

        "models/trackers/sort_tracker.py": """\"\"\"
SORT多目标跟踪器
\"\"\"

import numpy as np
from sort import Sort
from typing import List, Dict, Any

class SORTTracker:
    \"\"\"SORT多目标跟踪器封装\"\"\"
    
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        
    def update(self, detections: List[np.ndarray]) -> List[Dict[str, Any]]:
        \"\"\"更新跟踪器状态\"\"\"
        if len(detections) == 0:
            detections = np.empty((0, 5))
        else:
            detections = np.array(detections)
            
        tracks = self.tracker.update(detections)
        
        results = []
        for track in tracks:
            result = {
                'track_id': int(track[4]),
                'bbox': track[:4],  # [x1, y1, x2, y2]
                'class_id': int(track[5]) if len(track) > 5 else 0
            }
            results.append(result)
            
        return results
""",

        # 工具文件
        "utils/__init__.py": """\"\"\"
工具函数包
\"\"\"

from .video.video_utils import VideoReader, VideoWriter
from .image.roi_manager import ROIManager
from .visualization.visualizer import DetectionVisualizer

__all__ = [
    'VideoReader',
    'VideoWriter', 
    'ROIManager',
    'DetectionVisualizer'
]
""",

        "utils/video/video_utils.py": """\"\"\"
视频处理工具函数
\"\"\"

import cv2
import numpy as np
from typing import Optional, Tuple

class VideoReader:
    \"\"\"视频读取器\"\"\"
    
    def __init__(self, source: str):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f\"无法打开视频源: {source}\")
            
    def read_frame(self) -> Optional[np.ndarray]:
        \"\"\"读取一帧\"\"\"
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
        
    def get_properties(self) -> Tuple[int, int, int]:
        \"\"\"获取视频属性\"\"\"
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        return width, height, fps
        
    def release(self):
        \"\"\"释放资源\"\"\"
        self.cap.release()

class VideoWriter:
    \"\"\"视频写入器\"\"\"
    
    def __init__(self, output_path: str, width: int, height: int, fps: int = 30):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
    def write_frame(self, frame: np.ndarray):
        \"\"\"写入一帧\"\"\"
        self.writer.write(frame)
        
    def release(self):
        \"\"\"释放资源\"\"\"
        self.writer.release()
""",

        "utils/image/roi_manager.py": """\"\"\"
ROI区域管理
\"\"\"

import cv2
import numpy as np
from typing import List, Tuple

class ROIManager:
    \"\"\"ROI区域管理器\"\"\"
    
    def __init__(self):
        self.rois = {}
        
    def add_roi(self, name: str, points: List[Tuple[int, int]]):
        \"\"\"添加ROI区域\"\"\"
        self.rois[name] = points
        
    def draw_rois(self, image: np.ndarray) -> np.ndarray:
        \"\"\"在图像上绘制ROI区域\"\"\"
        result = image.copy()
        for name, points in self.rois.items():
            if len(points) == 2:  # 矩形
                cv2.rectangle(result, points[0], points[1], (0, 255, 0), 2)
            else:  # 多边形
                pts = np.array(points, np.int32)
                cv2.polylines(result, [pts], True, (0, 255, 0), 2)
            # 添加标签
            cv2.putText(result, name, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return result
        
    def is_point_in_roi(self, point: Tuple[int, int], roi_name: str) -> bool:
        \"\"\"判断点是否在ROI内\"\"\"
        if roi_name not in self.rois:
            return False
            
        points = self.rois[roi_name]
        if len(points) == 2:  # 矩形判断
            x1, y1 = points[0]
            x2, y2 = points[1]
            x, y = point
            return x1 <= x <= x2 and y1 <= y <= y2
        else:  # 多边形判断
            # 简化版本，实际可以使用cv2.pointPolygonTest
            return True
""",

        # 测试文件
        "tests/__init__.py": "",
        "tests/test_yolo_detector.py": """\"\"\"
YOLO检测器测试
\"\"\"

import cv2
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detectors.yolo_detector import YOLODetector

def test_yolo_detector():
    \"\"\"测试YOLO检测器\"\"\"
    print(\"🧪 测试YOLO检测器...\")
    
    # 创建检测器
    detector = YOLODetector()
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # 测试检测
    detections = detector.detect(test_image)
    print(f\"检测到 {len(detections)} 个目标\")
    
    # 测试姿态检测
    poses = detector.detect_poses(test_image)
    print(f\"检测到 {len(poses)} 个人体姿态\")
    
    print(\"✅ YOLO检测器测试完成\")

if __name__ == \"__main__\":
    test_yolo_detector()
""",

        # 脚本文件
        "scripts/utilities/setup_environment.py": """#!/usr/bin/env python3
\"\"\"
环境设置脚本
\"\"\"

import subprocess
import sys

def main():
    \"\"\"主函数\"\"\"
    print(\"🚀 设置铁路监控系统环境...\")
    
    # 安装依赖
    subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"-r\", \"../requirements.txt\"])
    
    print(\"✅ 环境设置完成\")

if __name__ == \"__main__\":
    main()
""",

        # Git忽略文件
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
railway_vision_pc/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data files
outputs/
*.mp4
*.avi
*.jpg
*.png
data/training/
data/validation/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Weights (large files)
weights/*.pt
weights/*.pth
!weights/.gitkeep
""",

        # 空文件标记
        "weights/.gitkeep": "",
        "data/.gitkeep": "",
        "outputs/.gitkeep": ""
    }
    
    # 创建所有文件
    for file_path, content in python_files.items():
        full_path = base_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📄 创建文件: {file_path}")
    
    print("=" * 60)
    print("🎉 完整版项目结构创建完成！")
    print("📍 项目位置: D:/railway_monitoring")
    print("📋 下一步:")
    print("   1. 检查项目结构")
    print("   2. 安装依赖: pip install -r requirements.txt") 
    print("   3. 运行测试: python tests/test_yolo_detector.py")
    print("   4. 开始开发! 🚀")

if __name__ == "__main__":
    create_complete_project_structure()