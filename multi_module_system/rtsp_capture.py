# rtsp_capture.py
import cv2
import time
import numpy as np
from .base_thread import BaseThread

class RTSPCaptureThread(BaseThread):
    """RTSP流捕获线程 - 专门处理海康等网络摄像头"""
    
    def __init__(self, rtsp_url, frame_buffer, result_manager, stop_event, config):
        # 为RTSP线程设置特殊名称
        name = f"RTSPCapture_{rtsp_url.split('@')[-1].split('/')[0]}" if '@' in rtsp_url else f"RTSPCapture_{rtsp_url}"
        super().__init__(name, frame_buffer, result_manager, stop_event, config)
        self.rtsp_url = rtsp_url
        self.cap = None
        self.last_frame = None
        self.frame_counter = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = config.get('rtsp_max_reconnect_attempts', 10)
        self.reconnect_delay = config.get('rtsp_reconnect_delay', 3.0)
        
        # RTSP参数配置
        self.rtsp_timeout = config.get('rtsp_timeout', 5000)  # 5秒超时
        self.rtsp_buffer_size = config.get('rtsp_buffer_size', 1)  # 缓冲区大小，降低延迟
        self.rtsp_frame_width = config.get('rtsp_frame_width', 1920)  # 期望宽度
        self.rtsp_frame_height = config.get('rtsp_frame_height', 1080)  # 期望高度
        
    def _create_capture(self):
        """创建RTSP捕获对象"""
        try:
            # 对于海康摄像头，我们可能需要添加一些参数
            if 'hikvision' in self.rtsp_url.lower() or '192.168' in self.rtsp_url:
                # 海康摄像头特殊参数
                cap = cv2.VideoCapture(self.rtsp_url)
                if not cap.isOpened():
                    return None
                
                # 设置RTSP参数
                cap.set(cv2.CAP_PROP_BUFFERSIZE, self.rtsp_buffer_size)
                cap.set(cv2.CAP_PROP_FPS, 30)  # 尝试设置帧率
                
                # 设置超时（毫秒）
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.rtsp_timeout)
                
                # 尝试设置分辨率
                if self.rtsp_frame_width and self.rtsp_frame_height:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.rtsp_frame_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.rtsp_frame_height)
                
                return cap
            else:
                # 通用RTSP流
                cap = cv2.VideoCapture(self.rtsp_url)
                if not cap.isOpened():
                    return None
                
                cap.set(cv2.CAP_PROP_BUFFERSIZE, self.rtsp_buffer_size)
                return cap
                
        except Exception as e:
            print(f"❌ 创建RTSP捕获对象失败: {e}")
            return None
    
    def _run_impl(self):
        """RTSP捕获线程主循环"""
        print(f"📡 启动RTSP流捕获: {self.rtsp_url}")
        
        # 连接RTSP流
        self.cap = self._create_capture()
        
        if self.cap is None or not self.cap.isOpened():
            print(f"❌ 无法连接RTSP流: {self.rtsp_url}")
            self.video_ended = True
            return
        
        # 获取流信息
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ RTSP连接成功: {width}x{height}, FPS: {fps:.1f}")
        
        # RTSP捕获主循环
        print(f"📹 {self.name} 开始捕获RTSP流")
        
        last_frame_time = time.time()
        no_frame_count = 0
        max_no_frame_count = 30  # 连续30帧无数据则尝试重连
        
        while not self.stop_event.is_set() and not self.video_ended:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print(f"⚠️ {self.name}: RTSP帧读取失败 (尝试 {no_frame_count + 1}/{max_no_frame_count})")
                    no_frame_count += 1
                    
                    if no_frame_count >= max_no_frame_count:
                        print(f"🔁 {self.name}: 连续帧读取失败，尝试重新连接...")
                        self._reconnect()
                        no_frame_count = 0
                        continue
                    
                    time.sleep(0.1)
                    continue
                
                # 重置无帧计数器
                no_frame_count = 0
                
                # 计算实际FPS
                current_time = time.time()
                if current_time - last_frame_time > 0:
                    current_fps = 1.0 / (current_time - last_frame_time)
                else:
                    current_fps = 0
                last_frame_time = current_time
                
                # 将帧放入帧缓冲区
                if frame is not None and self.frame_buffer is not None:
                    try:
                        # 如果配置了缩放，则调整帧大小
                        target_width = self.config.get('frame_width', width)
                        target_height = self.config.get('frame_height', height)
                        
                        if target_width != width or target_height != height:
                            frame = cv2.resize(frame, (target_width, target_height))
                        
                        self.frame_buffer.put_frame(frame=frame, timestamp=time.time())
                        self.frame_counter += 1
                        
                        # 每30帧打印一次调试信息
                        if self.frame_counter % 30 == 0:
                            print(f"📹 {self.name} 已捕获 {self.frame_counter} 帧, 实际FPS: {current_fps:.1f}, 帧形状: {frame.shape}")
                        
                        # 更新性能统计
                        self.frame_count += 1
                        if current_time - self.last_stats_time >= 1.0:
                            self.fps = self.frame_count / (current_time - self.last_stats_time)
                            self.frame_count = 0
                            self.last_stats_time = current_time
                            
                    except Exception as e:
                        print(f"🚨 {self.name}: 放入缓冲区失败: {e}")
                
                # 控制捕获速度（如果FPS太高）
                target_fps = self.config.get('target_fps', 30)
                if target_fps > 0 and current_fps > target_fps * 1.2:
                    sleep_time = max(0, (1.0 / target_fps) - 0.001)
                    time.sleep(sleep_time)
                    
            except cv2.error as e:
                print(f"🚨 {self.name} OpenCV错误: {e}")
                time.sleep(0.5)
                self._reconnect()
            except Exception as e:
                print(f"🚨 {self.name} 未知错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)
                self._reconnect()
        
        print(f"🛑 {self.name} RTSP线程结束")
    
    def _reconnect(self):
        """重新连接RTSP流"""
        self.reconnect_attempts += 1
        
        if self.reconnect_attempts > self.max_reconnect_attempts:
            print(f"❌ {self.name}: 达到最大重连次数 {self.max_reconnect_attempts}，线程退出")
            self.video_ended = True
            return
        
        print(f"🔁 {self.name}: 尝试重新连接 ({self.reconnect_attempts}/{self.max_reconnect_attempts})")
        
        # 释放旧的捕获对象
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 等待一段时间再重连
        time.sleep(self.reconnect_delay)
        
        # 重新连接
        self.cap = self._create_capture()
        if self.cap and self.cap.isOpened():
            print(f"✅ {self.name}: 重连成功")
            self.reconnect_attempts = 0  # 重置重连计数
        else:
            print(f"❌ {self.name}: 重连失败")
    
    def process_frame(self, frame, frame_count, timestamp):
        """RTSP捕获线程不需要处理帧，直接返回"""
        return None
    
    def get_performance_stats(self):
        """获取性能统计"""
        stats = super().get_performance_stats()
        stats.update({
            'frames_captured': self.frame_counter,
            'video_source': 'RTSP Stream',
            'reconnect_attempts': self.reconnect_attempts
        })
        return stats
    
    def cleanup(self):
        """清理资源"""
        if self.cap:
            self.cap.release()
            print(f"📹 {self.name} RTSP资源已释放")