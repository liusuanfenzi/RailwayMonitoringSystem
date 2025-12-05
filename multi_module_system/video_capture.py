# video_capture.py
import cv2
import time
import numpy as np
from .base_thread import BaseThread

class VideoCaptureThread(BaseThread):
    """视频捕获线程 - 完全重写版本"""
    
    def __init__(self, video_source, frame_buffer, result_manager, stop_event, config):
        # 即使视频捕获线程不需要result_manager，我们也传递它，因为BaseThread需要
        super().__init__(f"VideoCapture_{video_source}", frame_buffer, result_manager, stop_event, config)
        self.video_source = video_source
        self.cap = None
        self.is_camera = isinstance(video_source, int) or (isinstance(video_source, str) and video_source.isdigit())
        self.last_frame = None
        self.frame_counter = 0
        
    def _run_impl(self):
        """视频捕获线程主循环 - 直接实现，不调用父类"""
        print(f"🎬 启动视频捕获: {self.video_source}")
        
        max_reconnect_attempts = 3
        reconnect_delay = 2.0
        
        for attempt in range(max_reconnect_attempts):
            try:
                # 打开视频源
                if self.is_camera:
                    self.cap = cv2.VideoCapture(int(self.video_source), cv2.CAP_V4L2)
                else:
                    self.cap = cv2.VideoCapture(self.video_source)
                    
                if not self.cap.isOpened():
                    raise RuntimeError(f"无法打开视频源: {self.video_source}")
                
                # 设置视频参数
                if self.config.get('frame_width'):
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['frame_width'])
                if self.config.get('frame_height'):
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['frame_height'])
                
                print(f"✅ 视频源连接成功 (尝试 {attempt + 1}/{max_reconnect_attempts})")
                break
                
            except Exception as e:
                print(f"❌ 连接视频源失败 (尝试 {attempt + 1}/{max_reconnect_attempts}): {e}")
                if attempt < max_reconnect_attempts - 1:
                    time.sleep(reconnect_delay)
                else:
                    print(f"❌ 无法连接视频源，停止系统")
                    self.stop_event.set()
                    return
        
        # 视频捕获主循环
        print(f"📹 {self.name} 开始捕获循环")
        
        while not self.stop_event.is_set():
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("⚠️ 视频帧读取失败")
                    
                    if not self.is_camera:
                        # 视频文件结束
                        if self.config.get('loop_video', False):
                            print("🔄 重新开始播放视频")
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            print("📹 视频播放结束")
                            self.stop_event.set()
                            break
                    else:
                        # 摄像头故障，尝试重新初始化
                        print("🔁 尝试重新连接摄像头...")
                        self.cap.release()
                        time.sleep(1.0)
                        self.cap = cv2.VideoCapture(int(self.video_source), cv2.CAP_V4L2)
                        if not self.cap.isOpened():
                            print("❌ 摄像头重连失败")
                            break
                        continue
                
                # 将帧放入帧缓冲区
                if frame is not None and self.frame_buffer is not None:
                    self.frame_buffer.put_frame(frame=frame, timestamp=time.time())
                    self.frame_counter += 1
                    
                    # 每10帧打印一次调试信息
                    # if self.frame_counter % 10 == 0:
                    #     print(f"📹 {self.name} 已捕获 {self.frame_counter} 帧，帧形状: {frame.shape}")
                    
                    # 更新性能统计
                    self.frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_stats_time >= 1.0:
                        self.fps = self.frame_count / (current_time - self.last_stats_time)
                        self.frame_count = 0
                        self.last_stats_time = current_time
                
                # 控制捕获速度
                target_fps = self.config.get('target_fps', 30)
                if target_fps > 0:
                    sleep_time = max(0, (1.0 / target_fps) - 0.001)
                    time.sleep(sleep_time)
                    
            except cv2.error as e:
                print(f"🚨 OpenCV错误: {e}")
                time.sleep(0.1)
            except Exception as e:
                print(f"🚨 未知错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
    
    def process_frame(self, frame, frame_count, timestamp):
        """视频捕获线程不需要处理帧，直接返回"""
        # 这个方法不会被调用，因为我们已经重写了_run_impl
        return None
    
    def cleanup(self):
        """清理资源"""
        if self.cap:
            self.cap.release()
            print(f"📹 {self.name} 资源已释放")