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
        """初始化检测器后运行主循环"""
        print("🚗 初始化人车检测器")
        
        # 添加CUDA上下文调试信息
        import pycuda.driver as cuda
        # try:
        #     # 检查当前线程的CUDA上下文状态
        #     ctx = cuda.Context.get_current()
        #     print(f"🔍 检测线程启动时CUDA上下文: {ctx}")
        # except:
        #     print("⚠️ 检测线程启动时无CUDA上下文（将由autoinit自动创建）")
        
        try:
            # 使用统一的配置键名
            engine_path = self.config.get('person_vehicle_engine_path', 'yolov8n.engine')
            confidence = self.config.get('person_vehicle_confidence', 0.6)
            target_fps = self.config.get('person_vehicle_target_fps', 20)
            
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
            
            # 添加帧计数器
            frames_processed = 0
            
            # 调用父类的主循环，但添加更多调试信息
            while not self.stop_event.is_set():
                try:
                    # 获取帧数据
                    frame_data = self.get_frame_data()
                    if frame_data is None:
                        print(f"⚠️ {self.name}: 帧缓冲区为空，等待...")
                        time.sleep(0.1)
                        continue
                    
                    frame, frame_count, timestamp = frame_data
                    frames_processed += 1
                    
                    # 每5帧打印一次
                    # if frames_processed % 15 == 0:
                    #     print(f"🎯 {self.name} 正在处理第 {frames_processed} 帧，形状: {frame.shape}")
                    
                    # 处理帧
                    start_time = time.time()
                    result = self.process_frame(frame, frame_count, timestamp)
                    processing_time = time.time() - start_time
                    
                    # 更新性能统计
                    self.update_performance_stats(processing_time)
                    
                    # 保存结果
                    if result is not None:
                        # 确保结果包含原始帧
                        if isinstance(result, dict) and 'frame' not in result:
                            result['frame'] = frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                        
                        # 保存结果
                        saved = self.save_result(result)
                        if not saved:
                            print(f"⚠️ {self.name} 保存结果失败")
                    else:
                        # 即使没有结果也保存一个空结果
                        empty_result = {
                            'frame': frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                            'timestamp': timestamp,
                            'frame_count': frame_count,
                            'thread_name': self.name,
                            'status': 'no_result'
                        }
                        self.save_result(empty_result)
                    
                    # 控制处理频率
                    self.control_processing_rate()
                    
                except Exception as e:
                    print(f"⚠️ {self.name} 处理帧时异常: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)
                    
        except ImportError as e:
            print(f"❌ 导入模块失败: {e}")
        except Exception as e:
            print(f"❌ 人车检测器初始化失败: {e}")
            import traceback
            traceback.print_exc()
    
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