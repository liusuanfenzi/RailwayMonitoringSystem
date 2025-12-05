# base_thread.py
import threading
import time
import logging
import numpy as np
from abc import ABC, abstractmethod
from collections import deque

class BaseThread(threading.Thread, ABC):
    """所有检测线程的基类"""
    
    def __init__(self, name, frame_buffer=None, result_manager=None, stop_event=None, config=None):
        super().__init__()
        self.name = name
        self.frame_buffer = frame_buffer
        self.result_manager = result_manager
        self.stop_event = stop_event
        self.config = config or {}
        self.daemon = True
        
        # 性能统计
        self.frame_count = 0
        self.processing_times = deque(maxlen=30)
        self.last_stats_time = time.time()
        self.fps = 0
        
        # 设置模块名称
        self.module_name = name.lower().replace('thread', '').replace('_', ' ')
        
        # 视频结束标志
        self.video_ended = False
    
    def run(self):
        """线程主循环 - 提供统一的错误处理"""
        print(f"🚀 启动 {self.name}")
        
        try:
            self._run_impl()
        except Exception as e:
            print(f"❌ {self.name} 异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
            print(f"🛑 停止 {self.name}")
    
    def _run_impl(self):
        """线程主循环实现 - 子类可重写"""
        print(f"🔁 {self.name} 进入主循环")
        
        while not self.stop_event.is_set() and not self.video_ended:
            try:
                # 获取帧数据
                frame_data = self.get_frame_data()
                if frame_data is None:
                    # 检查是否是因为帧缓冲区收到结束信号
                    if self.frame_buffer and hasattr(self.frame_buffer, 'has_end_signal') and self.frame_buffer.has_end_signal():
                        print(f"🎬 {self.name}: 帧缓冲区已收到结束信号，线程正常退出")
                        self.video_ended = True
                        break
                    # 否则，只是缓冲区暂时为空，继续等待
                    time.sleep(0.01)
                    continue
                
                frame, frame_count, timestamp = frame_data
                
                # 检查是否收到视频结束信号（frame 为 None）
                if frame is None:
                    print(f"🎬 {self.name}: 收到视频结束信号，线程正常退出")
                    self.video_ended = True
                    break
                
                # 处理帧
                start_time = time.time()
                result = self.process_frame(frame, frame_count, timestamp)
                processing_time = time.time() - start_time
                
                # 更新性能统计
                self.update_performance_stats(processing_time)
                
                # 保存结果 - 重要：必须调用save_result
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
    
    def get_frame_data(self):
        """获取帧数据 - 支持结束信号"""
        if self.frame_buffer:
            # 使用新的get_frame_data方法，支持超时
            if hasattr(self.frame_buffer, 'get_frame_data'):
                frame_data = self.frame_buffer.get_frame_data(timeout=0.1)
            else:
                # 向后兼容
                frame_data = self.frame_buffer.get_frame_data()
                
            if frame_data:
                # 检查是否为结束信号
                if frame_data.get('is_end_signal', False):
                    return None, frame_data.get('frame_id', 0), frame_data.get('timestamp', time.time())
                
                return (
                    frame_data.get('frame'), 
                    frame_data.get('frame_id', 0),
                    frame_data.get('timestamp', time.time())
                )
        return None
    
    def get_frame_data_blocking(self, timeout=None):
        """阻塞方式获取帧数据"""
        if self.frame_buffer and hasattr(self.frame_buffer, 'get_frame_data'):
            frame_data = self.frame_buffer.get_frame_data(timeout=timeout)
            if frame_data:
                if frame_data.get('is_end_signal', False):
                    return None, frame_data.get('frame_id', 0), frame_data.get('timestamp', time.time())
                return (
                    frame_data.get('frame'), 
                    frame_data.get('frame_id', 0),
                    frame_data.get('timestamp', time.time())
                )
        return None
    
    @abstractmethod
    def process_frame(self, frame, frame_count, timestamp):
        """处理帧的抽象方法 - 子类必须实现"""
        pass
    
    def save_result(self, result):
        """保存处理结果到结果管理器 - 基类实现"""
        if self.result_manager is not None and result is not None:
            # 获取标准化的模块键名
            module_key = self.get_module_key()
            
            # 确保result是字典
            if not isinstance(result, dict):
                result = {'frame': result, 'thread_name': self.name}
            
            # 添加线程名称到结果中
            result['thread_name'] = self.name
            
            # 保存到结果管理器
            self.result_manager.put_result(module_key, result)
            
            # 更新性能统计
            stats = {
                'fps': self.fps,
                'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
                'module': module_key
            }
            self.result_manager.update_performance(module_key, stats)
            
            # 调试信息（可选）
            # if 'frame' in result and result['frame'] is not None:
            #     if self.frame_count % 30 == 0:  # 每30帧输出一次
            #         print(f"✅ {self.name} 保存结果到键: {module_key}, 帧形状: {result['frame'].shape}")
            # else:
            #     if self.frame_count % 30 == 0:
            #         print(f"⚠️ {self.name} 保存结果到键: {module_key}, 无帧")
            
            return True
        return False
    
    def get_module_key(self):
        """获取标准化模块键名"""
        # 根据线程名称返回标准化的键名
        name_lower = self.name.lower()
        
        if 'person' in name_lower or 'personvehicledetection' in name_lower:
            return 'personvehicledetection'
        elif 'foreign' in name_lower or 'foreignobjectdetection' in name_lower:
            return 'foreignobjectdetection'
        elif 'train' in name_lower or 'trainstationdetection' in name_lower:
            return 'trainstationdetection'
        elif 'video' in name_lower or 'videocapture' in name_lower:
            return 'videocapture'
        elif 'display' in name_lower:
            return 'display'
        else:
            # 默认：小写并移除特殊字符
            return name_lower.replace(' ', '').replace('_', '').replace('-', '')
    
    def update_performance_stats(self, processing_time):
        """更新性能统计"""
        self.frame_count += 1
        self.processing_times.append(processing_time)
        
        # 保持最近30个时间样本
        if len(self.processing_times) > 30:
            self.processing_times.popleft()
        
        # 每秒更新一次性能统计
        current_time = time.time()
        if current_time - self.last_stats_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_stats_time)
            self.frame_count = 0
            self.last_stats_time = current_time
    
    def control_processing_rate(self):
        """控制处理速率"""
        target_fps_key = f"{self.get_module_key()}_target_fps"
        target_fps = self.config.get(target_fps_key, self.config.get('target_fps', 30))
        
        if target_fps > 0:
            avg_time = np.mean(self.processing_times) if self.processing_times else 0
            sleep_time = max(0, (1.0 / target_fps) - avg_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_performance_stats(self):
        """获取性能统计"""
        avg_time = np.mean(self.processing_times) if self.processing_times else 0
        
        stats = {
            'module': self.module_name,
            'fps': self.fps,
            'avg_processing_time': avg_time,
            'frame_count': self.frame_count,
            'video_ended': self.video_ended  # 添加视频结束状态
        }
        
        # 添加特定模块的统计信息
        specific_stats = self.get_specific_stats()
        if specific_stats:
            stats.update(specific_stats)
            
        return stats
    
    def get_specific_stats(self):
        """获取特定模块的统计信息 - 子类可重写"""
        return {}
    
    def is_video_ended(self):
        """检查视频是否已结束"""
        return self.video_ended
    
    def cleanup(self):
        """清理资源 - 子类可重写"""
        pass