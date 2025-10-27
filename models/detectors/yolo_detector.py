import torch
import numpy as np
import os

class YOLODetector:
    """YOLO目标检测器封装"""
    
    def __init__(self, model_path='yolov5su.pt', conf_threshold=0.5, 
                 target_classes=None, use_gpu=True):
        """
        初始化YOLO检测器
        
        Args:
            model_path: 模型文件路径
            conf_threshold: 置信度阈值
            target_classes: 目标类别列表，None表示使用默认类别
            use_gpu: 是否使用GPU
        """
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.conf_threshold = conf_threshold
        
        # 设置目标类别
        self.target_classes = ['person', 'car']
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        print(f"✅ YOLO检测器初始化完成 - 设备: {self.device}")
        print(f"🎯 目标类别: {self.target_classes}")
    
    def _load_model(self, model_path):
        """加载YOLO模型"""
        try:
            from ultralytics import YOLO
            
            if os.path.exists(model_path):
                print(f"📁 从本地加载模型: {model_path}")
                model = YOLO(model_path)
            else:
                print("❌ 模型文件不存在，使用内置模型")
                model = YOLO('yolov8n.pt')
            
            # 设置参数
            model.conf = self.conf_threshold
            model.iou = 0.45
            
            return model
            
        except Exception as e:
            print(f"❌ YOLO加载失败: {e}")
            return self._create_mock_detector()
    
    def detect(self, frame):
        """
        检测单帧图像中的目标
        
        Args:
            frame: 输入图像
            
        Returns:
            detections: 检测结果 [[x1, y1, x2, y2, confidence, class_id], ...]
        """
        try:
            if hasattr(self.model, 'predict'):
                results = self.model.predict(frame, verbose=False, imgsz=640)
                return self._parse_detections(results)
            else:
                return np.empty((0, 6)) # 如果没有predict方法，返回空数组
                
        except Exception as e:
            print(f"❌ 检测失败: {e}")
            return np.empty((0, 6))
    
    def _parse_detections(self, results):
        """解析YOLO检测结果"""
        detections = []
        
        if len(results) == 0:
            return detections
        
        result = results[0]
        
        # 处理不同版本的输出格式
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes_data = result.boxes.data.cpu().numpy()
            
            for det in boxes_data:
                if len(det) >= 6:
                    x1, y1, x2, y2, confidence, class_id = det[:6]
                    class_id = int(class_id)
                    
                    class_name = self._get_class_name(class_id)
                    
                    if class_name in self.target_classes and confidence > self.conf_threshold:
                        detections.append([x1, y1, x2, y2, confidence, class_id])
        
        return np.array(detections) if detections else np.empty((0, 6))
    
    def _get_class_name(self, class_id):
        """根据类别ID获取类别名称"""
        try:
            if hasattr(self.model, 'names'):
                return self.model.names.get(class_id, f'class_{class_id}')
            else:
                coco_classes = {0: 'person', 2: 'car', 5: 'bus', 7: 'truck'}
                return coco_classes.get(class_id, f'class_{class_id}')
        except:
            return f'class_{class_id}'