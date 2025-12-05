# person_vehicle_detector.py
import time
import numpy as np
from .base_thread import BaseThread

class PersonVehicleDetectionThread(BaseThread):
    """人车检测线程 - 基于BaseThread"""
    
    def __init__(self, name, frame_buffer, result_manager, stop_event, config):
        super().__init__(name, frame_buffer, result_manager, stop_event, config)
        self.detector = None
        self.tracker = None
        self.stay_detector = None
        self.last_frame = None  # 添加这个属性
        
    def _run_impl(self):
        """初始化检测器，然后调用父类的主循环"""
        print("🚗 初始化人车检测器")
        
        # 添加CUDA上下文调试信息
        import pycuda.driver as cuda
        
        try:
            # 使用统一的配置键名
            engine_path = self.config.get('person_vehicle_engine_path', 'yolov8n.engine')
            confidence = self.config.get('person_vehicle_confidence', 0.6)
            
            from models.detector.yolo_detector import YOLODetector
            from models.tracker.multi_object_tracker import MultiObjectTracker
            from models.detector.stay_detector import StayDetector
            
            print("🔄 正在创建YOLODetector实例...")
            # 初始化检测器 - 这里会触发autoinit
            self.detector = YOLODetector(
                engine_path,
                conf_threshold=confidence,
                target_classes=['person', 'car']
            )
            
            # 验证CUDA上下文已创建
            try:
                ctx = cuda.Context.get_current()
                print(f"✅ 检测器创建后CUDA上下文: {ctx}")
            except:
                print("⚠️ 检测器创建后无法获取CUDA上下文")
            
            self.tracker = MultiObjectTracker(
                max_age=self.config.get('person_vehicle_max_age', 50),
                min_hits=self.config.get('person_vehicle_min_hits', 2),
                iou_threshold=self.config.get('person_vehicle_iou_threshold', 0.3)
            )
            self.stay_detector = StayDetector(
                stay_threshold=self.config.get('person_vehicle_stay_threshold', 10.0),
                movement_threshold=self.config.get('person_vehicle_movement_threshold', 15.0),
                min_frames=5  # 硬编码或添加配置
            )
            
            # 设置ROI
            roi_points = self.config.get('person_vehicle_detection_roi')
            if roi_points:
                self.detector.set_roi(roi_points)
                self.tracker.set_roi(roi_points)
            
            print("✅ 人车检测器初始化成功")
            
            # 现在调用父类的_run_impl方法，它会处理主循环
            super()._run_impl()
            
        except ImportError as e:
            print(f"❌ 导入模块失败: {e}")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"❌ 人车检测器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            # 初始化失败，标记视频结束，防止继续尝试
            self.video_ended = True
    
    def process_frame(self, frame, frame_count, timestamp):
        """处理单帧进行人车检测"""
        if self.detector is None or frame is None:
            # 返回一个包含帧的占位结果
            return {
                'frame': frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                'tracked_objects': [],
                'staying_objects': [],
                'detections': [],
                'timestamp': timestamp,
                'frame_count': frame_count,
                'thread_name': self.name,
                'status': 'detector_not_ready'
            }
        
        try:
            # 记录最后处理的帧
            self.last_frame = frame.copy()
            
            # 执行检测
            detections = self.detector.detect(frame)
            
            # 过滤检测结果
            valid_detections = []
            for det in detections:
                if len(det) >= 6:
                    class_id = int(det[5])
                    if class_id in [0, 2]:  # person, car
                        valid_detections.append(det)
            
            # 跟踪
            tracked_objects = []
            if len(valid_detections) > 0:
                tracked_objects = self.tracker.update(valid_detections, frame)
            
            # 停留检测
            self.stay_detector.update(tracked_objects, timestamp, frame)
            staying_objects = self.stay_detector.get_staying_objects()
            
            # 创建包含原始帧的结果
            result = {
                'frame': frame.copy(),  # 确保保存原始帧
                'tracked_objects': tracked_objects,
                'staying_objects': staying_objects,
                'detections': valid_detections,
                'timestamp': timestamp,
                'frame_count': frame_count,
                'thread_name': self.name
            }
            
            return result
            
        except Exception as e:
            print(f"⚠️ 人车检测处理失败: {e}")
            return {
                'frame': frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                'tracked_objects': [],
                'staying_objects': [],
                'detections': [],
                'timestamp': timestamp,
                'error': str(e),
                'thread_name': self.name
            }
    
    def get_specific_stats(self):
        """获取人车检测特定统计"""
        return {
            'objects_tracked': self.tracker.track_count if self.tracker else 0,
            'staying_objects': len(self.stay_detector.staying_objects) if self.stay_detector else 0
        }
    
    def cleanup(self):
        """清理资源"""
        if self.detector:
            self.detector.cleanup()
        print(f"🧹 {self.name} 已清理资源")