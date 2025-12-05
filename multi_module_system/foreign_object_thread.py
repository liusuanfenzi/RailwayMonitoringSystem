# foreign_object_thread.py
import time
import cv2
import numpy as np
from .base_thread import BaseThread
from models.detector.foreign_object_detector import MotionDetector, ForeignObjectDetector

class ForeignObjectThread(BaseThread):
    """异物检测线程"""
    
    def __init__(self, name, frame_buffer=None, result_manager=None, stop_event=None, config=None):
        super().__init__(name, frame_buffer, result_manager, stop_event, config)
        
        # 配置参数
        self.roi_coords = config.get('foreign_object_roi', [(550, 400, 400, 300)])
        self.min_static_duration = config.get('foreign_object_min_static_duration', 2.0)
        self.threshold = config.get('foreign_object_threshold', 200)
        self.min_area = config.get('foreign_object_min_area', 100)
        self.alert_dir = config.get('foreign_object_alert_dir', "alerts/foreign_object_detection")
        
        # 运动检测器参数
        self.motion_threshold = config.get('foreign_object_motion_threshold', 800)
        self.background_frames = config.get('foreign_object_background_frames', 15)
        self.difference_threshold = config.get('foreign_object_difference_threshold', 50)
        
        # 检测器实例（延迟初始化）
        self.motion_detector = None
        self.detector = None
        
        self.initialized = False
        self.last_frame = None  # 添加这个属性
        print(f"✅ {self.name} 初始化完成 - ROI: {self.roi_coords}")

    def _run_impl(self):
        """初始化检测器，然后调用父类的主循环"""
        print(f"🚀 {self.name} 正在初始化...")
        
        try:
            # 第一步：初始化背景模型
            if not self.initialize_background_model():
                print(f"❌ {self.name} 背景模型初始化失败")
                self.video_ended = True  # 标记视频结束，防止继续尝试
                return
            
            print(f"✅ {self.name} 背景模型初始化完成，开始正常检测循环")
            
            # 第二步：调用父类的主循环
            super()._run_impl()
            
        except Exception as e:
            print(f"❌ {self.name} 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.video_ended = True  # 标记视频结束，防止继续尝试

    def initialize_background_model(self):
        """初始化背景模型"""
        print(f"🚀 {self.name} 正在初始化背景模型...")
        
        # 创建运动检测器
        self.motion_detector = MotionDetector(
            roi_coords=self.roi_coords,
            motion_threshold=self.motion_threshold,
            background_frames=self.background_frames,
            difference_threshold=self.difference_threshold
        )
        
        # 从缓冲区构建背景模型
        if not self.motion_detector.build_background_from_buffer(self.frame_buffer, self.stop_event):
            return False
        
        # 创建异物检测器
        self.detector = ForeignObjectDetector(
            roi_coords=self.roi_coords,
            min_static_duration=self.min_static_duration,
            threshold=self.threshold,
            min_area=self.min_area,
            alert_dir=self.alert_dir
        )
        
        # 初始化异物检测器
        if not self.detector.initialize(self.motion_detector):
            return False
        
        self.initialized = True
        print(f"✅ {self.name} 背景模型初始化完成")
        return True

    def process_frame(self, frame, frame_count, timestamp):
        """处理帧的抽象方法实现"""
        if frame is None or not self.initialized:
            # 返回一个包含帧的占位结果
            return {
                'frame': frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                'frame_id': frame_count,
                'timestamp': timestamp,
                'thread_name': self.name,
                'status': 'not_initialized'
            }
        
        # 记录最后处理的帧
        self.last_frame = frame.copy()
        
        # 确保帧是 BGR 格式
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # 处理当前帧
        result = self.detector.process_frame(frame)
        
        if result is None:
            result = {}
        
        # 确保结果中包含原始帧
        result['frame'] = frame.copy()
        result['frame_id'] = frame_count
        result['timestamp'] = timestamp
        result['thread_name'] = self.name
        
        # 添加ROI信息
        result['roi_coords'] = self.roi_coords
        
        # 添加性能统计
        result['fps'] = self.fps
        
        # 添加警报信息
        if result.get('alert_info'):
            result['alert'] = result['alert_info']
            print(f"🚨 {self.name} 警报: {result['alert_info']}")
        
        return result

    def get_specific_stats(self):
        """获取特定模块的统计信息"""
        if self.detector and hasattr(self.detector, 'frame_count'):
            return {
                'detected_frames': self.detector.frame_count,
                'static_regions': len([r for r in self.detector.static_candidates.values() 
                                     if r['duration'] >= self.detector.min_static_duration * 25]),
                'total_alerts': len(self.detector.alerted_regions),
                'roi_area': f"{self.roi_coords[0][2]}x{self.roi_coords[0][3]}" if self.roi_coords else "N/A"
            }
        return {}

    def cleanup(self):
        """清理资源"""
        if self.detector and hasattr(self.detector, 'cleanup'):
            self.detector.cleanup()
        
        print(f"🧹 {self.name} 资源已清理")