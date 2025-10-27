# test_person_vehicle_detection_fixed.py
import cv2
import torch
import numpy as np
import time
import os
from collections import defaultdict
from pathlib import Path

print("🔍 初始化系统...")

# ✅ 修复SORT导入和调用问题
try:
    from sort import SortTracker
    
    # 创建适配器来修复参数和格式问题
    class SortAdapter:
        def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
            self.tracker = SortTracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        
        def update(self, detections):
            # 🔧 修复：确保检测结果有6个元素（x1, y1, x2, y2, confidence, class_id）
            if len(detections) == 0:
                return self.tracker.update(np.empty((0, 6)), None)
            else:
                # 确保每个检测有6个元素
                formatted_detections = []
                for det in detections:
                    if len(det) == 5:
                        # 添加默认的class_id (0)
                        formatted_detections.append([det[0], det[1], det[2], det[3], det[4], 0])
                    elif len(det) >= 6:
                        formatted_detections.append(det[:6])
                    else:
                        print(f"⚠️ 忽略无效检测: {det}")
                
                if len(formatted_detections) > 0:
                    tracked_results = self.tracker.update(np.array(formatted_detections), None)
                    # 🔧 修复：只返回前5个元素给StayDetector
                    if len(tracked_results) > 0 and tracked_results.shape[1] > 5:
                        return tracked_results[:, :5]  # 只返回 x1, y1, x2, y2, track_id
                    else:
                        return tracked_results
                else:
                    return self.tracker.update(np.empty((0, 6)), None)
    
    Sort = SortAdapter
    print("✅ SORT跟踪器加载成功 (适配器模式)")
    
except ImportError as e:
    print(f"❌ SORT导入失败: {e}")

class StayDetector:
    """停留检测器"""
    def __init__(self, stay_threshold=10, roi=None):
        self.stay_threshold = stay_threshold
        self.roi = roi
        self.track_history = defaultdict(list)
        self.staying_objects = set()
        self.alerted_objects = set()
        self.alert_dir = Path("alerts")
        self.alert_dir.mkdir(exist_ok=True)
        print(f"✅ 停留检测器初始化完成 - 阈值: {stay_threshold}秒")
        
    def update(self, tracked_objects, timestamp, frame=None):
        current_ids = set()
        
        for obj in tracked_objects:
            # 🔧 修复：安全地解包跟踪对象
            if len(obj) >= 5:
                # 只取前5个元素：x1, y1, x2, y2, track_id
                x1, y1, x2, y2, track_id = obj[:5]
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                # ROI检查
                if self.roi and not self._in_roi(center_x, center_y):
                    continue
                    
                current_ids.add(track_id)
                
                # 更新轨迹历史
                self.track_history[track_id].append((center_x, center_y, timestamp))
                
                # 清理旧数据（保持最近30秒）
                self.track_history[track_id] = [
                    pt for pt in self.track_history[track_id] 
                    if timestamp - pt[2] <= 30
                ]
                
                # 计算停留时间
                stay_duration = self._calculate_stay_duration(track_id, timestamp)
                
                # 触发报警
                if (stay_duration >= self.stay_threshold and 
                    track_id not in self.alerted_objects and 
                    frame is not None):
                    self.staying_objects.add(track_id)
                    self._trigger_alert(track_id, stay_duration, (x1, y1, x2, y2), frame)
            else:
                print(f"⚠️ 忽略无效跟踪对象: {obj}")
        
        # 清理不再存在的轨迹
        self.staying_objects = self.staying_objects.intersection(current_ids)
    
    def _calculate_stay_duration(self, track_id, current_time):
        if not self.track_history[track_id]:
            return 0
        return current_time - self.track_history[track_id][0][2]
    
    def _in_roi(self, x, y):
        if self.roi is None:
            return True
        x1, y1, x2, y2 = self.roi
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def _trigger_alert(self, track_id, duration, bbox, frame):
        print(f"🚨 违规停留报警 - ID: {track_id}, 时长: {duration:.1f}秒")
        self.alerted_objects.add(track_id)
        
        # 保存证据
        x1, y1, x2, y2 = map(int, bbox)
        alert_img = frame.copy()
        cv2.rectangle(alert_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(alert_img, f'Stay: {duration:.1f}s', (x1, y1-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(alert_img, f'ID: {track_id}', (x1, y1-40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        filename = f"stay_alert_id{track_id}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = self.alert_dir / filename
        cv2.imwrite(str(filepath), alert_img)
        print(f"💾 报警截图已保存: {filepath}")

class PersonVehicleTracker:
    def __init__(self, model_size='s', conf_threshold=0.5, use_gpu=True):
        # 设备配置
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"🎯 使用设备: {self.device}")
        
        # 🔧 使用ultralytics YOLO
        print("🔄 加载YOLO模型...")
        self.model = self._load_ultralytics_yolo(model_size, conf_threshold)
        
        # 跟踪器
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        
        # 停留检测
        self.stay_detector = StayDetector(stay_threshold=8)
        
        # 目标类别
        self.target_classes = ['person', 'car', 'truck', 'bus']
        self.processing_times = []
        self.frame_counter = 0
        
        print("✅ 人员车辆跟踪器初始化完成")
    
    def _load_ultralytics_yolo(self, model_size, conf_threshold):
        """使用ultralytics加载YOLO"""
        try:
            from ultralytics import YOLO
            
            # 尝试加载yolov5su.pt
            model_file = 'yolov5su.pt'
            
            if os.path.exists(model_file):
                print(f"📁 从本地加载模型: {model_file}")
                model = YOLO(model_file)
            else:
                print("❌ yolov5su.pt不存在，尝试其他模型...")
                model = YOLO('yolov8n.pt')  # 使用内置模型
            
            # 设置参数
            model.conf = conf_threshold
            model.iou = 0.45
            
            print("✅ YOLO模型加载成功")
            return model
            
        except Exception as e:
            print(f"❌ YOLO加载失败: {e}")
            print("🔄 使用模拟检测器进行测试...")
            return self._create_mock_detector()
    
    def _create_mock_detector(self):
        """创建模拟检测器用于测试"""
        class MockDetector:
            def __init__(self):
                self.conf = 0.5
                self.names = {
                    0: 'person', 2: 'car', 5: 'bus', 7: 'truck'
                }
                print("✅ 模拟检测器创建成功 (用于测试)")
            
            def predict(self, frame, verbose=False, imgsz=640):
                # 返回空的检测结果
                class Results:
                    def __init__(self):
                        self.boxes = type('Boxes', (), {
                            'data': torch.zeros((0, 6)),
                            'cpu': lambda self: self,
                            'numpy': lambda self: np.zeros((0, 6))
                        })()
                
                return [Results()]
        
        return MockDetector()
    
    def process_frame(self, frame):
        """处理单帧图像"""
        start_time = time.time()
        self.frame_counter += 1
        
        try:
            # YOLO检测
            if hasattr(self.model, 'predict'):
                # ultralytics格式
                results = self.model.predict(frame, verbose=False, imgsz=640)
                
                # 🔧 修复：安全解析检测结果
                filtered_detections = self._parse_detections_safely(results)
                
                print(f"🔍 检测到 {len(filtered_detections)} 个目标")
                
                # 目标跟踪
                if len(filtered_detections) > 0:
                    tracked_objects = self.tracker.update(np.array(filtered_detections))
                else:
                    tracked_objects = self.tracker.update(np.empty((0, 5)))
                
                print(f"🎯 跟踪到 {len(tracked_objects)} 个对象")
                
                # 停留检测
                current_time = self.frame_counter / 30.0
                self.stay_detector.update(tracked_objects, current_time, frame)
                
                # 可视化
                result_frame = self._visualize_results(frame, tracked_objects)
                
                # 性能统计
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 60:
                    self.processing_times.pop(0)
                
                return result_frame, tracked_objects
            else:
                # 模拟检测器
                return frame, np.empty((0, 5))
            
        except Exception as e:
            print(f"❌ 帧处理错误: {e}")
            import traceback
            traceback.print_exc()
            return frame, np.empty((0, 5))
    
    def _parse_detections_safely(self, results):
        """安全解析检测结果，处理不同版本的输出格式"""
        filtered_detections = []
        
        try:
            if len(results) == 0:
                return filtered_detections
            
            result = results[0]
            
            # 检查不同的输出格式
            if hasattr(result, 'boxes') and result.boxes is not None:
                # ultralytics v8格式
                boxes_data = result.boxes.data.cpu().numpy()
                
                for det in boxes_data:
                    # 安全地访问数组元素
                    if len(det) >= 6:  # 确保有足够的元素
                        x1, y1, x2, y2, confidence, class_id = det[:6]
                        class_id = int(class_id)
                        
                        # 获取类别名称
                        class_name = self._get_class_name(class_id)
                        
                        if class_name in self.target_classes and confidence > self.model.conf:
                            # 🔧 修复：提供6个元素给SORT
                            filtered_detections.append([x1, y1, x2, y2, confidence, class_id])
                    else:
                        print(f"⚠️ 检测结果维度不足: {len(det)}")
            
            elif hasattr(result, 'xyxy') and len(result.xyxy) > 0:
                # 旧版本格式
                boxes_data = result.xyxy[0].cpu().numpy()
                
                for det in boxes_data:
                    if len(det) >= 6:
                        x1, y1, x2, y2, confidence, class_id = det[:6]
                        class_id = int(class_id)
                        
                        class_name = self._get_class_name(class_id)
                        
                        if class_name in self.target_classes and confidence > self.model.conf:
                            filtered_detections.append([x1, y1, x2, y2, confidence, class_id])
            
            else:
                print("⚠️ 未知的检测结果格式")
                
        except Exception as e:
            print(f"❌ 解析检测结果时出错: {e}")
        
        return filtered_detections
    
    def _get_class_name(self, class_id):
        """获取类别名称"""
        try:
            if hasattr(self.model, 'names'):
                return self.model.names.get(class_id, f'class_{class_id}')
            else:
                coco_classes = {
                    0: 'person', 2: 'car', 5: 'bus', 7: 'truck'
                }
                return coco_classes.get(class_id, f'class_{class_id}')
        except:
            return f'class_{class_id}'
    
    def _visualize_results(self, frame, tracked_objects):
        """可视化跟踪和停留状态"""
        display_frame = frame.copy()
        
        # 绘制跟踪结果
        for obj in tracked_objects:
            # 🔧 修复：安全地解包跟踪对象
            if len(obj) >= 5:
                x1, y1, x2, y2, track_id = map(int, obj[:5])
                
                color = (0, 255, 0)  # 绿色-正常
                label = f"ID:{track_id}"
                
                if track_id in self.stay_detector.staying_objects:
                    color = (0, 0, 255)  # 红色-停留
                    label = f"ID:{track_id} (STAY)"
                
                # 绘制边界框和标签
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 显示性能信息
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # FPS显示
            cv2.putText(display_frame, f'FPS: {avg_fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 跟踪数量显示
            cv2.putText(display_frame, f'Tracks: {len(tracked_objects)}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 停留数量显示
            stay_count = len(self.stay_detector.staying_objects)
            color = (0, 0, 255) if stay_count > 0 else (0, 255, 0)
            cv2.putText(display_frame, f'Staying: {stay_count}', (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return display_frame
    
    def get_performance_stats(self):
        """获取性能统计"""
        if not self.processing_times:
            return "暂无性能数据"
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        return f"平均处理时间: {avg_time:.3f}s, 平均FPS: {avg_fps:.1f}"

def main():
    print("🚀 启动人员车辆停留检测系统")
    print("=" * 50)
    
    tracker = None
    
    try:
        # 初始化跟踪器
        tracker = PersonVehicleTracker(model_size='s', conf_threshold=0.5, use_gpu=True)
        
        # 视频源选择
        video_source = "data/test_videos/trash_in_area/1.mp4"  # 视频文件路径
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_source}")
            return
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📹 视频信息: {fps:.1f} FPS, 总帧数: {total_frames}")
        
        print("✅ 视频源打开成功")
        print("🎮 控制说明:")
        print("  - 按 'q' 键退出程序")
        print("  - 按 's' 键保存当前帧")
        print("  - 按 'r' 键重置停留检测")
        print("  - 人员/车辆停留超过8秒会触发报警")
        print("=" * 50)
        
        last_performance_log = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("📹 视频流结束")
                break
            
            # 处理帧
            result_frame, tracked_objects = tracker.process_frame(frame)
            
            # 显示结果
            cv2.imshow('人车停留检测系统', result_frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前状态
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"system_state_{timestamp}.jpg", result_frame)
                print(f"💾 系统状态已保存: system_state_{timestamp}.jpg")
            elif key == ord('r'):
                # 重置停留检测
                tracker.stay_detector = StayDetector(stay_threshold=8)
                print("🔄 停留检测已重置")
            
            # 定期显示性能信息（每5秒）
            current_time = time.time()
            if current_time - last_performance_log >= 5:
                stats = tracker.get_performance_stats()
                print(f"📊 {stats} | 跟踪对象: {len(tracked_objects)}")
                last_performance_log = current_time
                
    except KeyboardInterrupt:
        print("\n⏹ 用户中断程序")
    except Exception as e:
        print(f"❌ 系统运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        if tracker is not None:
            print(f"📊 最终统计: {tracker.get_performance_stats()}")
        print("🛑 系统已关闭")

if __name__ == "__main__":
    main()