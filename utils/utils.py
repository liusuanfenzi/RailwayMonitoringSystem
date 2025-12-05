# utils/jetson_utils.py
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any


class ROIManager:
    """Jetson优化的ROI区域管理器"""

    def __init__(self):
        self.rois = {}

    def add_roi(self, name: str, points: List[Tuple[int, int]]):
        """添加ROI区域"""
        if len(points) != 2:
            raise ValueError("ROI点必须是两个点 [(x1,y1), (x2,y2)]")
        self.rois[name] = points
        print(f"🎯 设置ROI区域 {name}: {points}")

    def crop_roi(self, image: np.ndarray, roi_name: str) -> np.ndarray:
        """裁剪ROI区域"""
        if roi_name not in self.rois:
            raise ValueError(f"ROI '{roi_name}' 不存在")

        points = self.rois[roi_name]
        x1, y1 = points[0]
        x2, y2 = points[1]
        return image[y1:y2, x1:x2]

    def point_in_roi(self, x: int, y: int, roi_name: str) -> bool:
        """检查点是否在指定ROI内"""
        if roi_name not in self.rois:
            return False

        points = self.rois[roi_name]
        x1, y1 = points[0]
        x2, y2 = points[1]
        return x1 <= x <= x2 and y1 <= y <= y2

    def get_roi_names(self):
        """获取所有ROI名称"""
        return list(self.rois.keys())


class PerformanceMonitor:
    """Jetson性能监控器"""

    def __init__(self):
        self.processing_times = []
        self.memory_usage = []

    def start_timing(self):
        """开始计时"""
        return cv2.getTickCount()

    def end_timing(self, start_time, operation_name=""):
        """结束计时并记录"""
        end_time = cv2.getTickCount()
        time_ms = (end_time - start_time) * 1000 / cv2.getTickFrequency()
        self.processing_times.append(time_ms)

        # 保持最近100次记录
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        if operation_name:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            fps = 1000 / avg_time if avg_time > 0 else 0
            print(f"⏱️ {operation_name}: {time_ms:.1f}ms, 平均FPS: {fps:.1f}")

        return time_ms

    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        if not self.processing_times:
            return {"avg_time": 0, "fps": 0}

        avg_time = sum(self.processing_times) / len(self.processing_times)
        fps = 1000 / avg_time if avg_time > 0 else 0
        return {"avg_time": avg_time, "fps": fps}
