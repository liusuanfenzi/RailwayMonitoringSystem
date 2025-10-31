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

        # ROI相关
        self.roi_points = None
        self.roi_active = False

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

    def set_roi(self, points):
        """设置ROI区域"""
        if len(points) == 2:
            self.roi_points = points
            self.roi_active = True
            print(f"🎯 设置YOLO检测ROI: {points}")
        else:
            print("⚠️ ROI点必须是两个点 [(x1,y1), (x2,y2)]")

    def disable_roi(self):
        """禁用ROI检测"""
        self.roi_active = False
        print("🔓 禁用ROI检测，使用全图检测")

    def detect(self, frame):
        """
        检测单帧图像中的目标（可选ROI区域）

        Args:
            frame: 输入图像

        Returns:
            detections: 检测结果 [[x1, y1, x2, y2, confidence, class_id], ...]
        """
        try:
            if self.roi_active and self.roi_points:
                # 在ROI区域内检测
                return self._detect_in_roi(frame)
            else:
                # 全图检测
                return self._detect_full_frame(frame)

        except Exception as e:
            print(f"❌ 检测失败: {e}")
            return np.empty((0, 6))

    def _detect_in_roi(self, frame):
        """在ROI区域内进行检测"""
        x1, y1 = self.roi_points[0]
        x2, y2 = self.roi_points[1]

        # 确保坐标在图像范围内
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # 裁剪ROI区域
        roi_frame = frame[y1:y2, x1:x2]

        if roi_frame.size == 0:
            print("⚠️ ROI区域为空，跳过检测")
            return np.empty((0, 6))

        # 在ROI区域上检测
        if hasattr(self.model, 'predict'):
            results = self.model.predict(roi_frame, verbose=False, imgsz=640)
            roi_detections = self._parse_detections(results)

            # 将坐标映射回原图
            detections = []
            for det in roi_detections:
                if len(det) >= 6:
                    # 坐标转换: ROI坐标 -> 原图坐标
                    x1_roi, y1_roi, x2_roi, y2_roi, confidence, class_id = det
                    x1_orig = x1_roi + x1
                    y1_orig = y1_roi + y1
                    x2_orig = x2_roi + x1
                    y2_orig = y2_roi + y1

                    detections.append(
                        [x1_orig, y1_orig, x2_orig, y2_orig, confidence, class_id])

            return np.array(detections) if detections else np.empty((0, 6))
        else:
            return np.empty((0, 6))

    def _detect_full_frame(self, frame):
        """全图检测"""
        if hasattr(self.model, 'predict'):
            results = self.model.predict(frame, verbose=False, imgsz=640)
            return self._parse_detections(results)
        else:
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
                        detections.append(
                            [x1, y1, x2, y2, confidence, class_id])

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
