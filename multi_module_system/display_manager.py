import cv2
import numpy as np
import time
import threading
from collections import defaultdict

class UnifiedDisplayManager:
    """统一的显示管理器 - 在主线程中管理所有窗口"""
    
    def __init__(self, result_manager, stop_event, config):
        self.result_manager = result_manager
        self.stop_event = stop_event
        self.config = config
        self.windows = {}  # 窗口名 -> 配置
        self.window_frames = {}  # 窗口名 -> 最新帧
        self.window_lock = threading.Lock()
        
    def add_window(self, window_name, module_key, position=None, size=(800, 600)):
        """添加显示窗口"""
        with self.window_lock:
            self.windows[window_name] = {
                'module_key': module_key,
                'position': position or (100, 100),
                'size': size,
                'created': False
            }
            self.window_frames[window_name] = None
            print(f"✅ 添加显示窗口: {window_name} -> 模块: {module_key}")
    
    def update_window(self, window_name, frame):
        """更新窗口帧"""
        with self.window_lock:
            if window_name in self.window_frames:
                self.window_frames[window_name] = frame
    
    def run(self):
        """主显示循环 - 在主线程中运行"""
        print("🖥️ 启动统一显示管理器")
        
        # 创建所有窗口
        self._create_windows()
        
        try:
            while not self.stop_event.is_set():
                # 更新所有窗口
                self._update_all_windows()
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("⏹️ 收到停止信号")
                    self.stop_event.set()
                    break
                elif key == ord('f'):
                    self._toggle_fullscreen()
                
                time.sleep(0.033)  # ~30FPS
                
        except Exception as e:
            print(f"❌ 显示管理器异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
            print("🛑 显示管理器停止")
    
    def _create_windows(self):
        """创建所有窗口"""
        for window_name, config in self.windows.items():
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                
                # 设置位置
                x, y = config['position']
                cv2.moveWindow(window_name, x, y)
                
                # 设置大小
                width, height = config['size']
                cv2.resizeWindow(window_name, width, height)
                
                config['created'] = True
                print(f"✅ 创建窗口: {window_name} 位置: ({x}, {y}) 大小: {width}x{height}")
                
            except Exception as e:
                print(f"❌ 创建窗口 {window_name} 失败: {e}")
    
    def _update_all_windows(self):
        """更新所有窗口"""
        all_results = self.result_manager.get_all_results()
        performance_stats = self.result_manager.get_performance_stats()
        
        for window_name, config in self.windows.items():
            #调试窗口名称
            #print(f"🔄 更新窗口: {window_name}")
            if not config['created']:
                continue
                
            module_key = config['module_key']
            
            # 获取该窗口对应的结果
            result = all_results.get(module_key)
            
            # 创建显示帧
            if result is not None:
                display_frame = self._create_display_frame(window_name, result, performance_stats)
            else:
                display_frame = self._create_default_frame(window_name, module_key)
            
            # 更新窗口
            if display_frame is not None and display_frame.size > 0:
                try:
                    cv2.imshow(window_name, display_frame)
                except Exception as e:
                    print(f"❌ 显示窗口 {window_name} 失败: {e}")
                    # 尝试重新创建窗口
                    try:
                        cv2.destroyWindow(window_name)
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        x, y = config['position']
                        cv2.moveWindow(window_name, x, y)
                        cv2.resizeWindow(window_name, *config['size'])
                        cv2.imshow(window_name, display_frame)
                    except Exception as e2:
                        print(f"❌ 重新创建窗口 {window_name} 失败: {e2}")
    
    def _create_display_frame(self, window_name, result, performance_stats):
        """为指定窗口创建显示帧"""
        if not isinstance(result, dict) or 'frame' not in result:
            return self._create_default_frame(window_name, result)
        
        frame = result['frame']
        if frame is None or not hasattr(frame, 'shape'):
            return self._create_default_frame(window_name, result)
        
        # 确保帧是BGR格式
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # 创建显示帧副本
        display_frame = frame.copy()
        
        # 获取窗口配置
        config = self.windows.get(window_name, {})
        target_size = config.get('size', (800, 600))
        
        # 缩放帧以适应窗口
        if (display_frame.shape[1] > target_size[0] or 
            display_frame.shape[0] > target_size[1]):
            scale = min(target_size[0] / display_frame.shape[1], 
                       target_size[1] / display_frame.shape[0])
            new_width = int(display_frame.shape[1] * scale)
            new_height = int(display_frame.shape[0] * scale)
            display_frame = cv2.resize(display_frame, (new_width, new_height))
        
        # 添加窗口标题
        cv2.putText(display_frame, window_name, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 添加模块信息
        module_key = config.get('module_key', 'unknown')
        cv2.putText(display_frame, f"module: {module_key}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 添加性能信息
        if performance_stats and module_key in performance_stats:
            stats = performance_stats[module_key]
            fps = stats.get('fps', 0)
            proc_time = stats.get('avg_processing_time', 0) * 1000
            
            perf_text = f"FPS: {fps:.1f}, process_time: {proc_time:.1f}ms"
            cv2.putText(display_frame, perf_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 添加边框
        cv2.rectangle(display_frame, (0, 0), 
                     (display_frame.shape[1]-1, display_frame.shape[0]-1), 
                     (0, 255, 0), 2)
        
        return display_frame
    
    def _create_default_frame(self, window_name, module_info):
        """创建默认显示帧"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 添加渐变背景
        for i in range(3):
            frame[:, :, i] = np.linspace(50, 200, 640, dtype=np.uint8)
        
        # 添加窗口标题
        cv2.putText(frame, window_name, (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        if isinstance(module_info, dict) and 'thread_name' in module_info:
            module_text = module_info['thread_name']
        elif isinstance(module_info, str):
            module_text = module_info
        else:
            module_text = str(module_info)
        
        cv2.putText(frame, f"module: {module_text}", 
                   (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 1)
        
        cv2.putText(frame, "waiting detection...", 
                   (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 1)
        
        # 添加测试图形
        cv2.rectangle(frame, (100, 250), (540, 400), (0, 255, 0), 3)
        cv2.putText(frame, "test visualization", 
                   (200, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
        
        return frame
    
    def _toggle_fullscreen(self):
        """切换全屏模式"""
        # 可以指定一个窗口进行全屏切换
        pass

# 移除原来的DisplayThread类，我们不再需要它
# ResultManager保持不变
class ResultManager:
    """结果管理器 - 线程安全的结果存储"""
    
    def __init__(self):
        self.results = {}
        self.lock = threading.Lock()
        self.performance_stats = {}
        self.update_times = {}  # 记录每个模块的最后更新时间
        
    def put_result(self, module_name, result):
        """放入模块结果"""
        with self.lock:
            self.results[module_name] = result
            self.update_times[module_name] = time.time()
            
    def get_result(self, module_name):
        """获取模块结果"""
        with self.lock:
            return self.results.get(module_name)
            
    def get_all_results(self):
        """获取所有结果"""
        with self.lock:
            # 返回所有未过时的结果
            current_time = time.time()
            valid_results = {}
            for key, result in self.results.items():
                # 如果结果在5秒内更新过，则认为有效
                if key in self.update_times and (current_time - self.update_times[key]) < 5.0:
                    valid_results[key] = result
            return valid_results
            
    def update_performance(self, module_name, stats):
        """更新性能统计"""
        with self.lock:
            self.performance_stats[module_name] = stats
            
    def get_performance_stats(self):
        """获取性能统计"""
        with self.lock:
            return self.performance_stats.copy()