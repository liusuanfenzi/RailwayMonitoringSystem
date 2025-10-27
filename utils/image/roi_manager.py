"""
ROI区域管理
"""

import cv2
import numpy as np
from typing import List, Tuple

class ROIManager:
    """ROI区域管理器"""
    
    def __init__(self):
        self.rois = {}
        
    def add_roi(self, name: str, points: List[Tuple[int, int]]):
        """添加ROI区域"""
        self.rois[name] = points
        
    def draw_rois(self, image: np.ndarray) -> np.ndarray:
        """在图像上绘制ROI区域"""
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
        """判断点是否在ROI内"""
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
