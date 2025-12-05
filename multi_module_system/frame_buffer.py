# frame_buffer.py
import threading
from collections import deque
import numpy as np
import time

class ThreadSafeFrameBuffer:
    """线程安全的帧缓冲区 - 支持结束信号"""
    
    def __init__(self, max_size=10, name="unnamed"):
        self.max_size = max_size  # 修复：需要保存max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)  # 添加条件变量
        self.latest_frame = None
        self.frame_count = 0
        self.timestamp = 0
        self.name = name
        self._end_signal_received = False  # 标记是否收到结束信号
        
    def put_frame(self, frame, timestamp=None):
        """放入新帧 - 支持结束信号（None）"""
        with self.condition:
            # 检查是否已经收到结束信号
            if self._end_signal_received:
                print(f"⚠️ {self.name}: 缓冲区已收到结束信号，不再接收新帧")
                return
            
            # 如果是结束信号
            if frame is None:
                print(f"📭 {self.name}: 收到结束信号")
                self._end_signal_received = True
                # 清空缓冲区并放入结束标记
                self.buffer.clear()
                self.buffer.append({
                    'frame': None,
                    'timestamp': timestamp or time.time(),
                    'frame_id': self.frame_count,
                    'is_end_signal': True  # 标记为结束信号
                })
            else:
                # 正常帧的处理
                # 如果缓冲区已满，移除最老的帧（使用popleft，而不是pop(0)）
                if len(self.buffer) >= self.max_size:
                    self.buffer.popleft()  # 修改这里：pop(0) -> popleft()

                # 放入新帧    
                self.buffer.append({
                    'frame': frame.copy(),
                    'timestamp': timestamp or time.time(),
                    'frame_id': self.frame_count,
                    'is_end_signal': False
                })
                self.latest_frame = frame.copy()
                self.frame_count += 1
            
            # 通知等待的线程
            self.condition.notify_all()
            
            # 每10帧打印一次调试信息
            # if self.frame_count % 10 == 0 and frame is not None:
            #     print(f"🔄 缓冲区 '{self.name}' 已放入 {self.frame_count} 帧，当前大小: {len(self.buffer)}")

    def get_frame_data(self, timeout=None):
        """获取帧数据 - 支持超时等待"""
        with self.condition:
            # 如果没有数据且未超时，则等待
            if not self.buffer and timeout:
                self.condition.wait(timeout)
            
            if not self.buffer:
                return None  # 超时或没有数据
            
            # 获取最老的帧
            data = self.buffer.popleft()
            
            # 如果是结束信号，清空缓冲区
            if data.get('is_end_signal', False):
                print(f"📭 {self.name}: 转发结束信号")
                self.buffer.clear()
                return data
            
            # 打印调试信息（频率降低）
            # if len(self.buffer) % 5 == 0:
            #     print(f"📥 从缓冲区 '{self.name}' 获取帧，剩余: {len(self.buffer)}")
            
            return data
            
    def get_latest_frame(self):
        """获取最新帧"""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
            
    def get_frame_data_non_blocking(self):
        """非阻塞方式获取帧数据"""
        with self.lock:
            if self.buffer:
                data = self.buffer.popleft()
                
                # 如果是结束信号，清空缓冲区
                if data.get('is_end_signal', False):
                    print(f"📭 {self.name}: 转发结束信号（非阻塞）")
                    self.buffer.clear()
                    return data
                
                if len(self.buffer) % 5 == 0:
                    print(f"📥 从缓冲区 '{self.name}' 获取帧（非阻塞），剩余: {len(self.buffer)}")
                return data
            else:
                return None
            
    def has_end_signal(self):
        """检查是否收到结束信号"""
        with self.lock:
            return self._end_signal_received
            
    def clear(self):
        """清空缓冲区"""
        with self.condition:
            self.buffer.clear()
            self.latest_frame = None
            print(f"🧹 {self.name}: 缓冲区已清空")
            
    def get_frame_count(self):
        """获取帧计数"""
        with self.lock:
            return self.frame_count
            
    def get_buffer_size(self):
        """获取当前缓冲区大小"""
        with self.lock:
            return len(self.buffer)