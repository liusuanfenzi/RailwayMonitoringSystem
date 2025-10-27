import time
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path

class StayDetector:
    """人员车辆停留检测器（基于位置变化的停留检测）"""
    
    def __init__(self, stay_threshold=10, movement_threshold=20, min_frames=10, 
                 roi_manager=None, alert_dir="alerts"):
        """
        初始化停留检测器
        
        Args:
            stay_threshold: 停留时间阈值（秒）
            movement_threshold: 移动阈值（像素），小于此值认为静止
            min_frames: 最小连续静止帧数
            roi_manager: ROI管理器实例
            alert_dir: 报警截图保存目录
        """
        self.stay_threshold = stay_threshold
        self.movement_threshold = movement_threshold
        self.min_frames = min_frames
        self.roi_manager = roi_manager
        self.alert_dir = Path(alert_dir)
        self.alert_dir.mkdir(exist_ok=True)

        # 跟踪状态
        self.track_history = defaultdict(list)
        self.stationary_start_time = defaultdict(float)  # 静止开始时间
        self.stationary_frames = defaultdict(int)        # 连续静止帧数
        self.staying_objects = set()
        self.alerted_objects = set()
        
        print(f"✅ 停留检测器初始化完成 - 阈值: {stay_threshold}秒, 移动阈值: {movement_threshold}像素")
    
    def update(self, tracked_objects, timestamp, frame=None):
        """
        更新停留检测状态（基于位置变化）
        
        Args:
            tracked_objects: 跟踪结果 [[x1, y1, x2, y2, track_id], ...]
            timestamp: 当前时间戳
            frame: 当前帧图像
        """
        current_ids = set()
        
        # 确保timestamp是数字类型
        try:
            timestamp = float(timestamp)
        except (ValueError, TypeError):
            timestamp = 0.0
        
        for obj in tracked_objects:
            if len(obj) >= 5:
                try:
                    x1, y1, x2, y2, track_id = obj[:5]
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    track_id = int(track_id)
                    
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # 只处理ROI区域内的目标
                    if not self._check_in_any_roi(center_x, center_y):
                        # 目标离开ROI，重置状态
                        self._reset_track_state(track_id)
                        continue
                        
                    current_ids.add(track_id)
                    
                    # 更新轨迹历史
                    self.track_history[track_id].append((center_x, center_y, timestamp))
                    
                    # 清理旧数据（保持最近30秒）
                    self.track_history[track_id] = [
                        pt for pt in self.track_history[track_id] 
                        if timestamp - float(pt[2]) <= 30
                    ]
                    
                    # 检查是否静止
                    is_stationary = self._check_stationary(track_id)
                    
                    if is_stationary:
                        # 目标静止
                        if self.stationary_start_time[track_id] == 0:
                            # 第一次检测到静止，记录开始时间
                            self.stationary_start_time[track_id] = timestamp
                            self.stationary_frames[track_id] = 1
                        else:
                            # 继续静止，增加帧数
                            self.stationary_frames[track_id] += 1
                        
                        # 计算静止时间
                        stationary_duration = timestamp - self.stationary_start_time[track_id]
                        
                        # 只有连续静止足够多帧才认为是真正的停留
                        if (self.stationary_frames[track_id] >= self.min_frames and 
                            stationary_duration >= self.stay_threshold and 
                            track_id not in self.alerted_objects and 
                            frame is not None):
                            
                            self.staying_objects.add(track_id)
                            self._trigger_alert(track_id, stationary_duration, (x1, y1, x2, y2), frame)
                        
                        print(f"🕒 轨迹ID {track_id} 静止时间: {stationary_duration:.1f}秒, 连续帧: {self.stationary_frames[track_id]}")
                        
                    else:
                        # 目标在移动，重置静止状态
                        self._reset_track_state(track_id)
                        print(f"🚶 轨迹ID {track_id} 在移动")
                        
                except (ValueError, TypeError) as e:
                    print(f"⚠️ 坐标解析错误: {e}，跳过该目标")
                    continue
        
        # 清理不再存在的轨迹
        expired_ids = set(self.track_history.keys()) - current_ids
        for track_id in expired_ids:
            self._reset_track_state(track_id)
        
        self.staying_objects = self.staying_objects.intersection(current_ids)
    
    def _check_stationary(self, track_id):
        """
        检查目标是否静止
        
        Args:
            track_id: 轨迹ID
            
        Returns:
            bool: 是否静止
        """
        history = self.track_history[track_id]
        if len(history) < 2:
            return False
        
        # 获取最近几个位置点（例如最近5个点）
        recent_points = history[-5:] if len(history) >= 5 else history
        
        # 计算位置变化
        positions = np.array([(x, y) for x, y, _ in recent_points])
        
        # 计算所有点之间的最大距离
        if len(positions) > 1:
            # 方法1：计算位置的标准差
            position_std = np.std(positions, axis=0)
            movement_magnitude = np.sqrt(np.sum(position_std ** 2))
            
            # 方法2：计算 bounding box 的大小
            min_pos = np.min(positions, axis=0)
            max_pos = np.max(positions, axis=0)
            bbox_size = np.sqrt(np.sum((max_pos - min_pos) ** 2))
            
            # 使用两种方法的较小值
            movement = min(movement_magnitude, bbox_size)
            
            return movement < self.movement_threshold
        
        return False
    
    def _reset_track_state(self, track_id):
        """重置目标的停留状态"""
        self.stationary_start_time[track_id] = 0
        self.stationary_frames[track_id] = 0
        if track_id in self.staying_objects:
            self.staying_objects.remove(track_id)
    
    def _check_in_any_roi(self, x, y):
        """检查点是否在任意ROI区域内"""
        if self.roi_manager is None:
            return True
            
        for roi_name in self.roi_manager.get_roi_names():
            if self.roi_manager.point_in_roi(int(x), int(y), roi_name):
                return True
        return False
    
    def _trigger_alert(self, track_id, duration, bbox, frame):
        """
        触发停留报警
        """
        print(f"🚨 违规停留报警 - ID: {track_id}, 时长: {duration:.1f}秒")
        self.alerted_objects.add(track_id)
        
        x1, y1, x2, y2 = map(int, bbox)
        alert_img = frame.copy()
        
        # 绘制报警信息
        cv2.rectangle(alert_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(alert_img, f'Stay: {duration:.1f}s', (x1, y1-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(alert_img, f'ID: {track_id}', (x1, y1-40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        filename = f"stay_alert_id{track_id}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = self.alert_dir / filename
        cv2.imwrite(str(filepath), alert_img)
        print(f"💾 报警截图已保存: {filepath}")
    
    def reset(self):
        """重置停留检测状态"""
        self.track_history.clear()
        self.stationary_start_time.clear()
        self.stationary_frames.clear()
        self.staying_objects.clear()
        self.alerted_objects.clear()
        print("🔄 停留检测已重置")
    
    def get_staying_count(self):
        """获取当前停留对象数量"""
        return len(self.staying_objects)
    
    def get_staying_objects(self):
        """获取当前停留对象集合"""
        return self.staying_objects.copy()