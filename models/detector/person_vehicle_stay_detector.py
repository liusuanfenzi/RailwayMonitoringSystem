#!/usr/bin/env python3
# jetson_person_vehicle_detection.py  ← 零 PyTorch 版
import time
import cv2
import numpy as np
from collections import defaultdict, deque
from pathlib import Path
import gc   # 代替 torch.cuda.empty_cache

# 零 PyTorch 的检测器 & 跟踪器
from models.detector.yolo_detector import YOLODetector
from models.tracker.multi_object_tracker import MultiObjectTracker

# 工具类（无 torch）
try:
    from utils.utils import JetsonROIManager, JetsonPerformanceMonitor
except ImportError:
    # 如果工具类不存在，创建简单的替代
    class JetsonROIManager:
        def __init__(self):
            self.rois = {}
        def add_roi(self, name, points):
            self.rois[name] = points
        def point_in_roi(self, x, y, name):
            if name not in self.rois:
                return False
            (x1, y1), (x2, y2) = self.rois[name]
            return x1 <= x <= x2 and y1 <= y <= y2
        def get_roi_names(self):
            return list(self.rois.keys())
    
    class JetsonPerformanceMonitor:
        def start_timing(self):
            return time.time()
        def end_timing(self, start_time, operation_name):
            return time.time() - start_time


class JetsonPersonVehicleDetection:
    """Jetson 人车停留检测系统（零 PyTorch）"""

    def __init__(self, engine_path='yolov8n.engine', conf_threshold=0.5,
                 use_gpu=True, stay_threshold=8, movement_threshold=15,
                 skip_frame_mode=True, detection_interval=3):
        print("🚀 初始化Jetson人车检测系统（零PyTorch）...")

        # 1. TensorRT 检测器（零 torch）
        self.detector = YOLODetector(engine_path, conf_threshold)
        # 2. IOU-Only 跟踪器（零 torch）
        self.tracker = MultiObjectTracker(
            max_age=50, min_hits=2, iou_threshold=0.3,
            max_cosine_distance=0.3, use_gpu=use_gpu)

        self.roi_manager = JetsonROIManager()
        self.performance_monitor = JetsonPerformanceMonitor()

        # 停留参数
        self.stay_threshold = stay_threshold
        self.movement_threshold = movement_threshold

        # 跟踪状态
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.stationary_start_time = {}
        self.stationary_frames = {}
        self.staying_objects = set()
        self.alerted_objects = set()

        # 跳帧
        self.skip_frame_mode = skip_frame_mode
        self.detection_interval = detection_interval
        self.last_detection_frame = 0
        self.last_detections = np.empty((0, 6), dtype=np.float32)

        # 性能
        self.frame_counter = 0
        self.last_performance_log = 0
        self.memory_cleanup_interval = 100

        print("✅ Jetson人车检测系统（零PyTorch）初始化完成")

    # ---------------- 对外接口 ----------------
    def set_detection_roi(self, name, points):
        self.detector.set_roi(points)
        self.tracker.set_roi(points)
        self.roi_manager.add_roi(name, points)
        print(f"🎯 设置检测ROI {name}: {points}")

    def disable_roi_detection(self):
        self.detector.disable_roi()
        self.tracker.disable_roi()
        print("🔓 禁用ROI检测")

    def add_stay_detection_roi(self, name, points):
        self.roi_manager.add_roi(name, points)
        print(f"🎯 添加停留检测ROI {name}: {points}")

    # ---------------- 主流程 ----------------
    def process_frame(self, frame):
        start_time = self.performance_monitor.start_timing()
        self.frame_counter += 1

        # 定期内存清理（纯 Python）
        if self.frame_counter % self.memory_cleanup_interval == 0:
            self._cleanup_memory()

        try:
            # 1. 检测（跳帧）
            detections = self._optimized_detection(frame)
            # 2. 跟踪
            tracked_objects = self.tracker.update(detections, frame)
            # 3. 停留判定
            self._update_stay_detection(tracked_objects)
            # 4. 可视化
            result_frame = self._create_visualization(frame, tracked_objects)
            # 5. 性能
            self.performance_monitor.end_timing(start_time, "帧处理")
            self._log_performance()
            return result_frame, tracked_objects
        except Exception as e:
            print(f"❌ 帧处理错误: {e}")
            self._cleanup_memory()
            return frame, []

    # ---------------- 子模块 ----------------
    def _optimized_detection(self, frame):
        if self.skip_frame_mode:
            if (self.frame_counter - self.last_detection_frame) >= self.detection_interval:
                detections = self.detector.detect(frame)
                self.last_detections = detections if len(detections) > 0 else np.empty((0, 6), dtype=np.float32)
                self.last_detection_frame = self.frame_counter
                if len(detections) > 0:
                    print(f"🔍 检测到 {len(detections)} 个目标")
            else:
                detections = self.last_detections
        else:
            detections = self.detector.detect(frame)
            if len(detections) > 0:
                print(f"🔍 检测到 {len(detections)} 个目标")
        return detections

    def _update_stay_detection(self, tracked_objects):
        current_time = time.time()
        current_ids = set()
        
        for obj in tracked_objects:
            if len(obj) < 5:
                continue
            x1, y1, x2, y2, track_id = obj[:5]
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            current_ids.add(track_id)
            
            # 历史轨迹
            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=30)
            self.track_history[track_id].append((cx, cy, current_time))
            
            # 静止判定
            if self._is_stationary(track_id):
                if track_id not in self.stationary_start_time:
                    self.stationary_start_time[track_id] = current_time
                    self.stationary_frames[track_id] = 1
                else:
                    self.stationary_frames[track_id] += 1
                    
                stationary_duration = current_time - self.stationary_start_time[track_id]
                if (self.stationary_frames[track_id] >= 5 and
                    stationary_duration >= self.stay_threshold):
                    self.staying_objects.add(track_id)
                    if track_id not in self.alerted_objects:
                        self.alerted_objects.add(track_id)
                        print(f"🚨 停留报警 - ID: {track_id}, 时长: {stationary_duration:.1f}秒")
            else:
                self._reset_track_state(track_id)
                
        # 清理过期
        expired_ids = set(self.track_history.keys()) - current_ids
        for tid in expired_ids:
            self._reset_track_state(tid)
            if tid in self.track_history:
                del self.track_history[tid]

    def _is_stationary(self, track_id):
        history = self.track_history[track_id]
        if len(history) < 2:
            return False
            
        recent = list(history)[-min(5, len(history)):]
        positions = np.array([(x, y) for x, y, _ in recent], dtype=np.float32)
        
        if len(positions) > 1:
            movement = np.sqrt(np.sum((positions[-1] - positions[0]) ** 2))
            return movement < self.movement_threshold
        return False

    def _reset_track_state(self, track_id):
        self.stationary_start_time.pop(track_id, None)
        self.stationary_frames.pop(track_id, None)
        self.staying_objects.discard(track_id)
        self.alerted_objects.discard(track_id)

    def _cleanup_memory(self):
        # 纯 Python 垃圾回收
        gc.collect()
        print("🧹 内存清理完成")

    def _create_visualization(self, frame, tracked_objects):
        vis = frame.copy()
        
        # ROI 框
        for name, points in self.roi_manager.rois.items():
            cv2.rectangle(vis, points[0], points[1], (0, 255, 0), 2)
            
        # 跟踪框
        for obj in tracked_objects:
            if len(obj) < 5:
                continue
            x1, y1, x2, y2, tid = obj[:5]
            color = (0, 0, 255) if tid in self.staying_objects else (0, 255, 0)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(vis, f'ID:{int(tid)}', (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
        # 信息
        current_time = time.time()
        fps = 1.0 / (current_time - getattr(self, '_last_t', current_time) + 1e-3)
        self._last_t = current_time
        
        info = f'FPS:{fps:.1f}  Stay:{len(self.staying_objects)}'
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return vis

    def _log_performance(self):
        current_time = time.time()
        if current_time - getattr(self, '_last_log', 0) > 10:
            fps = 1.0 / (current_time - getattr(self, '_last_t', current_time) + 1e-3)
            print(f"📊 FPS:{fps:.1f}  跟踪:{len(self.track_history)}  停留:{len(self.staying_objects)}")
            self._last_log = current_time

    def reset_detection(self):
        self.track_history.clear()
        self.stationary_start_time.clear()
        self.stationary_frames.clear()
        self.staying_objects.clear()
        self.alerted_objects.clear()
        gc.collect()
        print("🔄 检测状态已重置（零PyTorch）")

    def toggle_skip_frame_mode(self, interval=None):
        self.skip_frame_mode = not self.skip_frame_mode
        if interval:
            self.detection_interval = interval
        print(f"⏩ 跳帧模式:{self.skip_frame_mode}  interval:{self.detection_interval}")