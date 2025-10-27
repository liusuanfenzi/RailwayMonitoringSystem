"""
视频处理工具函数
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Generator
from pathlib import Path

class VideoReader:
    """视频读取器"""
    
    def __init__(self, source: str):
        # 检查文件是否存在（如果是文件路径）
        if not str(source).isdigit():  # 不是摄像头ID
            video_path = Path(source)
            if not video_path.exists():
                raise ValueError(f"视频文件不存在: {source}")
        
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频源: {source}")
            
    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧"""
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
        
    def get_properties(self) -> Tuple[int, int, int, int]:
        """获取视频属性"""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return width, height, fps, frame_count
        
    def stream_generator(self) -> Generator[np.ndarray, None, None]:
        """生成器方式读取视频流"""
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame
        
    def release(self):
        """释放资源"""
        self.cap.release()

class VideoWriter:
    """视频写入器"""
    
    def __init__(self, output_path: str, width: int, height: int, fps: int = 30):
        # 确保输出目录存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
    def write_frame(self, frame: np.ndarray):
        """写入一帧"""
        self.writer.write(frame)
        
    def release(self):
        """释放资源"""
        self.writer.release()

class ROIManager:
    """ROI区域管理器"""
    
    def __init__(self):
        self.rois = {}
        
    def add_roi(self, name: str, points: list):
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
        
    def crop_roi(self, image: np.ndarray, roi_name: str) -> np.ndarray:
        """裁剪ROI区域"""
        if roi_name not in self.rois:
            raise ValueError(f"ROI '{roi_name}' 不存在")
            
        points = self.rois[roi_name]
        if len(points) == 2:  # 矩形裁剪
            x1, y1 = points[0]
            x2, y2 = points[1]
            return image[y1:y2, x1:x2]
        else:
            # 多边形裁剪（简化版，返回边界矩形）
            pts = np.array(points, np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            return image[y:y+h, x:x+w]
    
    def point_in_roi(self, x: int, y: int, roi_name: str) -> bool:
        """检查点是否在指定ROI内"""
        if roi_name not in self.rois:
            return False
            
        points = self.rois[roi_name]
        if len(points) == 2:  # 矩形检查
            x1, y1 = points[0]
            x2, y2 = points[1]
            return x1 <= x <= x2 and y1 <= y <= y2
        else:  # 多边形检查
            pts = np.array(points, np.int32)
            return cv2.pointPolygonTest(pts, (x, y), False) >= 0
    
    def get_roi_names(self):
        """获取所有ROI名称"""
        return list(self.rois.keys())