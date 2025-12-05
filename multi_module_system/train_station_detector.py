# train_station_detector.py
import time
import numpy as np
from .base_thread import BaseThread

class TrainStationDetectionThread(BaseThread):
    """列车检测线程 - 基于BaseThread"""
    
    def __init__(self, name,frame_buffer, result_manager, stop_event, config):
        super().__init__(name, frame_buffer, result_manager, stop_event, config)
        self.detector = None
        self.bg_subtractor = None
        
        # 背景模型预热
        self.warmup_complete = False
        self.warmup_frames = 0
        self.target_warmup_frames = self.config.get('warmup_frames', 15)
        
    def _run_impl(self):
        """初始化检测器后运行主循环"""
        print("🚆 初始化列车检测器")
        
        try:
            self.initialize_detector()
            print("✅ 列车检测器初始化成功")
            
            # 调用父类的主循环
            super()._run_impl()
            
        except Exception as e:
            print(f"❌ 列车检测器初始化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def initialize_detector(self):
        """初始化列车检测器"""
        from models.detector.train_detector import TrainDetector
        from models.background_subtractor.gmm_model import GMMBackgroundSubtractor
        
        # 使用统一的配置键名
        learning_rate = self.config.get('train_station_bg_learning_rate', 0.01)
        history = self.config.get('train_station_bg_history', 500)
        var_threshold = self.config.get('train_station_bg_var_threshold', 16)
        detect_shadows = self.config.get('train_station_bg_detect_shadows', True)
        spatial_threshold = self.config.get('train_station_spatial_threshold', 0.05)
        temporal_frames = self.config.get('train_station_temporal_frames', 50)
        temporal_threshold = self.config.get('train_station_temporal_threshold', 45)
        print_interval = self.config.get('train_station_print_interval', 10)
        
        self.bg_subtractor = GMMBackgroundSubtractor(
            learning_rate=learning_rate,
            history=history,
            var_threshold=var_threshold,
            detect_shadows=detect_shadows
        )
        
        self.detector = TrainDetector(
            spatial_threshold=spatial_threshold,
            temporal_frames=temporal_frames,
            temporal_threshold=temporal_threshold,
            print_interval=print_interval
        )
        
        # 设置ROI
        roi_points = self.config.get('train_station_roi')
        if roi_points and hasattr(self.bg_subtractor, 'roi_manager'):
            self.bg_subtractor.roi_manager.add_roi("train_roi", roi_points)
    
    def process_frame(self, frame, frame_count, timestamp):
        """处理单帧进行列车检测"""
        if self.bg_subtractor is None or self.detector is None:
            return None
        
        # 背景模型预热
        if not self.warmup_complete:
            learning_rate = 0.1  # 预热阶段使用较高的学习率
            self.warmup_frames += 1
            if self.warmup_frames >= self.target_warmup_frames:
                self.warmup_complete = True
                print("✅ 背景模型预热完成")
        else:
            learning_rate = self.config.get('bg_learning_rate', 0.01)
        
        try:
            # 应用背景减除
            if hasattr(self.bg_subtractor, 'apply_with_roi_analysis'):
                bg_results = self.bg_subtractor.apply_with_roi_analysis(frame, learning_rate=learning_rate)
            else:
                # 后备方法
                bg_results = self.bg_subtractor.apply(frame, learning_rate=learning_rate)
                bg_results = self._format_bg_results(bg_results, frame)
            
            # 检测列车事件
            events = self.detector.detect_events(bg_results, frame, frame_count)
            
            # 添加背景减除结果到返回数据
            events['bg_results'] = bg_results
            events['warmup_complete'] = self.warmup_complete
            events['warmup_progress'] = f"{self.warmup_frames}/{self.target_warmup_frames}"
            
            # 获取车站状态
            station_status = self.get_station_status(events)
            
            return {
                'frame': frame,
                'train_detections': events,
                'station_status': station_status,
                'timestamp': timestamp,
                'frame_count': frame_count
            }
            
        except Exception as e:
            print(f"⚠️ 列车检测处理异常: {e}")
            return {
                'frame': frame,
                'train_detections': {
                    'confidence': 0.0,
                    'spatial_detected': False,
                    'current_state': 'unknown',
                    'event_triggered': False,
                    'error': str(e)
                },
                'station_status': {
                    'state': 'unknown',
                    'confidence': 0.0,
                    'event_triggered': False,
                    'trains_detected': 0,
                    'warmup_complete': self.warmup_complete
                },
                'timestamp': timestamp
            }
    
    def _format_bg_results(self, bg_result, frame):
        """格式化背景减除结果为统一格式"""
        if isinstance(bg_result, dict):
            return bg_result
        else:
            # 假设bg_result是前景掩码
            return {
                'full_frame': {
                    'mask': bg_result,
                    'foreground_ratio': np.sum(bg_result > 0) / (bg_result.size if hasattr(bg_result, 'size') else 1)
                }
            }
    
    def get_station_status(self, train_results):
        """根据检测结果分析车站状态"""
        if not train_results:
            return {
                'state': 'unknown',
                'confidence': 0.0,
                'event_triggered': False,
                'event_type': None,
                'trains_detected': 0,
                'warmup_complete': self.warmup_complete
            }
        
        # 从检测结果中提取状态信息
        state = train_results.get('current_state', 'unknown')
        confidence = train_results.get('confidence', 0.0)
        event_triggered = train_results.get('event_triggered', False)
        event_type = train_results.get('event_type', None)
        
        # 判断是否有列车检测
        spatial_detected = train_results.get('spatial_detected', False)
        trains_detected = 1 if spatial_detected and confidence > 0.1 else 0
        
        return {
            'state': str(state),  # 确保state是字符串
            'confidence': confidence,
            'event_triggered': event_triggered,
            'event_type': event_type,
            'trains_detected': trains_detected,
            'warmup_complete': self.warmup_complete,
            'warmup_progress': f"{self.warmup_frames}/{self.target_warmup_frames}"
        }
    
    def get_specific_stats(self):
        """获取列车检测特定统计"""
        detector_status = {}
        if self.detector and hasattr(self.detector, 'get_detection_status'):
            detector_status = self.detector.get_detection_status()
        
        return {
            'warmup_complete': self.warmup_complete,
            'warmup_progress': f"{self.warmup_frames}/{self.target_warmup_frames}",
            **detector_status
        }
    
    def cleanup(self):
        """清理资源"""
        if self.detector and hasattr(self.detector, 'reset_detector'):
            try:
                self.detector.reset_detector()
            except:
                pass
        
        if self.bg_subtractor and hasattr(self.bg_subtractor, 'release'):
            try:
                self.bg_subtractor.release()
            except:
                pass