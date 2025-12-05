# frame_buffer.py
import threading
from collections import deque
import numpy as np
import time

class ThreadSafeFrameBuffer:
    """线程安全的帧缓冲区 - 添加调试信息"""
    
    def __init__(self, max_size=10, name="unnamed"):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_count = 0
        self.timestamp = 0
        self.name = name  # 添加缓冲区名称
        
    def put_frame(self, frame, timestamp=None):
        """放入新帧"""
        with self.lock:
            if frame is None:
                return
                
            self.buffer.append({
                'frame': frame.copy(),
                'timestamp': timestamp or time.time(),
                'frame_id': self.frame_count
            })
            self.latest_frame = frame.copy()
            self.frame_count += 1
            
            # 每10帧打印一次调试信息
            # if self.frame_count % 10 == 0:
            #     print(f"🔄 缓冲区 '{self.name}' 已放入 {self.frame_count} 帧，当前大小: {len(self.buffer)}")
            
    def get_latest_frame(self):
        """获取最新帧"""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
            
    def get_frame_data(self):
        """获取完整的帧数据"""
        with self.lock:
            if self.buffer:
                data = self.buffer[-1].copy()
                # 打印调试信息（频率降低）
                # if len(self.buffer) % 5 == 0:
                #     print(f"📥 从缓冲区 '{self.name}' 获取帧，剩余: {len(self.buffer)}")
                return data
            else:
                # 缓冲区为空时打印警告
                print(f"⚠️ 缓冲区 '{self.name}' 为空")
                return None
            
    def get_frame_count(self):
        """获取帧计数"""
        with self.lock:
            return self.frame_count
            
    def get_buffer_size(self):
        """获取当前缓冲区大小"""
        with self.lock:
            return len(self.buffer)