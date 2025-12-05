# foreign_object_detector.py
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import os
import time


class MotionDetector:
    """
    基于固定背景模型的运动检测器 - 简化版本
    """

    def __init__(self, roi_coords,
                 motion_threshold=1000,
                 background_frames=10,
                 difference_threshold=25):
        """
        初始化运动检测器

        Args:
            roi_coords: ROI区域列表 [(x, y, w, h), ...]
            motion_threshold: 运动像素阈值
            background_frames: 背景模型帧数
            difference_threshold: 差分阈值
        """
        self.roi_coords = roi_coords
        self.motion_threshold = motion_threshold
        self.background_frames = background_frames
        self.difference_threshold = difference_threshold

        # 内部变量
        self.cap = None
        self.background_model = None
        self.frame_count = 0
        self.background_initialized = False

    def build_background_from_buffer(self, frame_buffer, stop_event):
        """
        从帧缓冲区构建背景模型
        
        Args:
            frame_buffer: 帧缓冲区实例
            stop_event: 停止事件
        """
        print(f"📊 正在从视频前 {self.background_frames} 帧构建背景模型...")
        
        frames_for_bg = []
        bg_frame_count = 0
        
        # 等待足够多的帧
        max_wait_time = 30  # 最大等待时间（秒）
        start_time = time.time()
        
        while bg_frame_count < self.background_frames:
            if time.time() - start_time > max_wait_time:
                print(f"⚠️ 等待超时，只读取了 {bg_frame_count} 帧用于背景建模")
                break
            
            if stop_event and stop_event.is_set():
                print("⏹️ 构建背景模型被中断")
                return False
            
            # 从缓冲区获取帧 - 直接处理 numpy 数组
            frame = frame_buffer.get_latest_frame()
            
            # 检查帧是否有效
            if frame is not None:
                # 确保是 numpy 数组
                if isinstance(frame, np.ndarray) and frame.size > 0:
                    try:
                        # 转换为灰度图
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frames_for_bg.append(gray_frame)
                        bg_frame_count += 1
                        
                        if bg_frame_count % 5 == 0:
                            print(f"  读取第 {bg_frame_count}/{self.background_frames} 帧...")
                    except Exception as e:
                        print(f"⚠️ 转换灰度图失败: {e}")
                        # 继续尝试，不中断
                else:
                    print("⚠️ 获取到无效的帧")
            
            time.sleep(0.05)  # 短暂等待
        
        if not frames_for_bg:
            print("❌ 无法读取任何帧用于背景建模")
            return False
        
        # 计算平均背景
        print("📈 计算平均背景模型...")
        self.background_model = np.mean(
            np.array(frames_for_bg, dtype=np.float32), axis=0).astype(np.uint8)
        
        # 可选：对背景进行模糊处理，减少噪声
        self.background_model = cv2.GaussianBlur(
            self.background_model, (5, 5), 0)
        
        self.background_initialized = True
        print(f"✅ 背景模型构建完成，使用 {len(frames_for_bg)} 帧")
        return True
    
    def _create_roi_mask(self, width, height):
        """创建ROI掩码"""
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        for (x, y, w, h) in self.roi_coords:
            cv2.rectangle(roi_mask, (x, y), (x + w, y + h), 255, -1)
        return roi_mask

    def process_frame(self, frame):
        """处理单帧并返回多个结果"""
        if frame is None or not self.background_initialized:
            return None, None, None, None

        # 转换为灰度
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. 背景差分
        diff = cv2.absdiff(gray_frame, self.background_model)

        # 2. 二值化
        _, fgmask = cv2.threshold(
            diff, self.difference_threshold, 255, cv2.THRESH_BINARY)

        # 3. 创建ROI掩码
        roi_mask = self._create_roi_mask(frame.shape[1], frame.shape[0])

        # 4. 应用ROI
        roi_fgmask = cv2.bitwise_and(fgmask, roi_mask)

        # 5. 形态学处理
        kernel = np.ones((5, 5), np.uint8)
        roi_fgmask = cv2.morphologyEx(roi_fgmask, cv2.MORPH_CLOSE, kernel)
        roi_fgmask = cv2.morphologyEx(roi_fgmask, cv2.MORPH_OPEN, kernel)

        # 6. 统计运动像素
        motion_pixels = np.sum(roi_fgmask > 0)
        has_motion = motion_pixels > self.motion_threshold

        # 7. 创建彩色掩码帧
        colored_mask = cv2.cvtColor(roi_fgmask, cv2.COLOR_GRAY2BGR)

        # 8. 创建带有ROI框的原始帧
        frame_with_roi = frame.copy()
        for (x, y, w, h) in self.roi_coords:
            # 用绿色矩形框出ROI区域
            cv2.rectangle(frame_with_roi, (x, y),
                          (x + w, y + h), (0, 255, 0), 2)

        # 9. 裁剪ROI区域的前景掩码
        cropped_masks = []
        for (x, y, w, h) in self.roi_coords:
            cropped_mask = roi_fgmask[y:y+h, x:x+w]
            cropped_masks.append(cropped_mask)

        self.frame_count += 1

        return frame_with_roi, colored_mask, cropped_masks, has_motion


class ForeignObjectDetector:
    def __init__(self, roi_coords, min_static_duration=2.0, threshold=200, min_area=100,
                 alert_dir="alerts/foreign_object_detection"):
        """
        初始化检测器

        Args:
            roi_coords: ROI坐标列表 [(x, y, w, h), ...]
            min_static_duration: 最小静止时间(秒)
            threshold: 白色阈值(0-255，越高越严格)
            min_area: 最小区域面积(像素)
            alert_dir: 警报截图保存目录
        """
        self.min_static_duration = min_static_duration
        self.threshold = threshold
        self.min_area = min_area
        self.fps = 30
        self.roi_coords = roi_coords
        self.alert_dir = alert_dir
        
        # 状态管理
        self.static_candidates = {}  # 候选静止区域
        self.alerted_regions = set()  # 已报警区域ID
        self.frame_count = 0
        self.last_alert_time = {}  # 每个区域上次报警时间
        
        # 运动检测器
        self.motion_detector = None
        
        # 创建警报目录
        os.makedirs(self.alert_dir, exist_ok=True)
        print(f"📁 异物检测警报目录: {os.path.abspath(self.alert_dir)}")

    def initialize(self, motion_detector: MotionDetector):
        """初始化运动检测器"""
        self.motion_detector = motion_detector
        return True

    def extract_white_regions(self, frame: np.ndarray) -> List[np.ndarray]:
        """从帧中提取白色区域的二值掩码"""
        if frame is None:
            return []
            
        # 转换为灰度图（如果不是的话）
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # 创建白色区域掩码
        _, white_mask = cv2.threshold(
            gray, self.threshold, 255, cv2.THRESH_BINARY)

        # 形态学操作去除噪声
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 过滤小区域
        large_contours = [
            cnt for cnt in contours if cv2.contourArea(cnt) > self.min_area]

        return large_contours

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        处理单帧图像
        
        Args:
            frame: 输入帧
            
        Returns:
            处理结果字典
        """
        if frame is None or self.motion_detector is None:
            return {}
        
        # 处理当前帧
        frame_with_roi, colored_mask, cropped_masks, has_motion = self.motion_detector.process_frame(frame)
        
        if frame_with_roi is None:
            return {}
        
        self.frame_count += 1
        contours = self.extract_white_regions(colored_mask)
        
        # 更新候选区域
        self._update_static_candidates(contours, self.frame_count)
        
        # 检查静止区域并触发警报
        alert_info = self._check_and_trigger_alerts(frame_with_roi, self.frame_count)
        
        # 在帧上绘制结果
        result_frame = self._visualize_results(frame_with_roi, contours, cropped_masks)
        
        return {
            'frame': result_frame,
            'alert_info': alert_info,
            'contours': len(contours),
            'static_count': len([r for r in self.static_candidates.values() 
                               if r['duration'] >= self.min_static_duration * 25]),
            'alert_count': len(self.alerted_regions),
            'frame_count': self.frame_count,
            'has_motion': has_motion
        }

    def _update_static_candidates(self, contours, frame_count):
        """更新候选静止区域"""
        for i, contour in enumerate(contours):
            # 计算轮廓的边界框和面积
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # 查找匹配的现有区域
            matched_region_id = None
            for region_id, data in self.static_candidates.items():
                if self._is_region_stable(data['contour'], contour, x, y, w, h):
                    matched_region_id = region_id
                    break

            if matched_region_id is not None:
                # 更新现有区域
                self.static_candidates[matched_region_id]['last_frame'] = frame_count
                self.static_candidates[matched_region_id]['duration'] += 1
                self.static_candidates[matched_region_id]['bbox'] = (x, y, w, h)
                self.static_candidates[matched_region_id]['contour'] = contour
            else:
                # 创建新区域
                region_id = len(self.static_candidates)
                self.static_candidates[region_id] = {
                    'first_frame': frame_count,
                    'last_frame': frame_count,
                    'duration': 0,
                    'bbox': (x, y, w, h),
                    'contour': contour
                }

    def _check_and_trigger_alerts(self, frame_with_roi, frame_count):
        """检查并触发警报"""
        alert_info = None
        current_time = time.time()
        
        for region_id, data in self.static_candidates.items():
            if data['duration'] >= self.min_static_duration * 25:
                if data['last_frame'] == frame_count:
                    # 计算持续时间（秒）
                    duration_seconds = (frame_count - data['first_frame']) / self.fps
                    
                    # 检查是否需要报警（防重复）
                    should_alert = False
                    if region_id not in self.alerted_regions:
                        should_alert = True
                    else:
                        # 如果已经报警过，检查是否超过一定时间
                        last_time = self.last_alert_time.get(region_id, 0)
                        if current_time - last_time > 300:  # 5分钟后再报警
                            should_alert = True
                    
                    if should_alert:
                        alert_info = self._trigger_alert(region_id, duration_seconds, data['bbox'], frame_with_roi)
                        self.alerted_regions.add(region_id)
                        self.last_alert_time[region_id] = current_time
        
        return alert_info

    def _trigger_alert(self, region_id: int, duration: float, bbox: Tuple, frame_with_roi: np.ndarray):
        """触发警报并保存截图"""
        x, y, w, h = bbox
        
        # 保存原始窗口当前帧的截图
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(
            self.alert_dir, 
            f"foreign_object_region_{region_id}_{timestamp}.jpg"
        )
        cv2.imwrite(save_path, frame_with_roi)
        
        alert_info = {
            'region_id': region_id,
            'duration': duration,
            'save_path': save_path,
            'timestamp': timestamp,
            'type': 'foreign_object'
        }
        
        print(f"🚨 异物警报！区域 {region_id} 静止 {duration:.2f} 秒")
        print(f"💾 警报截图已保存: {save_path}")
        
        return alert_info

    def _visualize_results(self, frame_with_roi, contours, cropped_masks):
        """可视化处理结果"""
        # 绘制满足静止条件的区域
        for region_id, data in self.static_candidates.items():
            if data['duration'] >= self.min_static_duration * 25:
                x, y, w, h = data['bbox']
                
                # 绘制橙色框表示满足条件的区域
                cv2.rectangle(frame_with_roi, (x, y),
                              (x + w, y + h), (0, 165, 255), 3)
                
                # 显示停留时间
                duration_seconds = (self.frame_count - data['first_frame']) / self.fps
                text = f"{duration_seconds:.2f}s"
                cv2.putText(
                    img=frame_with_roi,
                    text=text,
                    org=(x, y - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 165, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
        
        # 添加帧计数信息
        cv2.putText(frame_with_roi, f'Frame: {self.frame_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_with_roi, f'Detected: {len(contours)}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        static_count = len([r for r in self.static_candidates.values() 
                          if r['duration'] >= self.min_static_duration * 25])
        cv2.putText(frame_with_roi, f'Static: {static_count}',
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        cv2.putText(frame_with_roi, f'Alerts: {len(self.alerted_regions)}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame_with_roi

    def _is_region_stable(self, old_contour, new_contour, x, y, w, h, tolerance=0.1):
        """检查区域是否稳定在同一位置"""
        if old_contour is None or new_contour is None:
            return False
            
        prev_x, prev_y, prev_w, prev_h = cv2.boundingRect(old_contour)

        # 检查位置变化
        center_x = x + w // 2
        center_y = y + h // 2
        prev_center_x = prev_x + prev_w // 2
        prev_center_y = prev_y + prev_h // 2

        dx = abs(center_x - prev_center_x)
        dy = abs(center_y - prev_center_y)
        if dx > max(w, prev_w) * tolerance or dy > max(h, prev_h) * tolerance:
            return False

        # 检查尺寸变化
        current_area = cv2.contourArea(old_contour)
        prev_area = cv2.contourArea(new_contour)
        if prev_area > 0:
            area_change = abs(current_area - prev_area) / prev_area
            if area_change > 0.2:  # 允许20%的面积变化
                return False

        return True

    def reset(self):
        """重置检测器状态"""
        self.static_candidates = {}
        self.alerted_regions.clear()
        self.frame_count = 0
        self.last_alert_time.clear()

    def cleanup(self):
        """清理资源"""
        self.reset()