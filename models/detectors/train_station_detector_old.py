# """
# 列车进出站检测模块 - 基于状态机的事件检测
# """
from typing import Dict, List
import cv2
import numpy as np
from pathlib import Path
from utils.state_manager_old import TrainState, TrainStateManager
from models.background_subtractors.gmm_model_old1 import GMMBackgroundSubtractor


class TrainStationDetector:
    """列车进出站检测器 - 基于状态机的事件检测"""

    def __init__(self, **kwargs):
        """
        初始化列车检测器

        Args:
            **kwargs: 状态机参数
        """
        # 初始化状态管理器
        self.state_manager = TrainStateManager(
            min_stay_duration=kwargs.get('min_stay_duration', 5.0),
            cooldown_duration=kwargs.get('cooldown_duration', 3.0),
            entering_timeout=kwargs.get('entering_timeout', 10.0),
            exiting_timeout=kwargs.get('exiting_timeout', 10.0)
        )

        self.event_history = []  # 事件历史记录
        self.entry_threshold = kwargs.get('entry_threshold', 0.05)
        self.exit_threshold = kwargs.get('exit_threshold', 0.05)

        print("✅ 列车进出站检测器初始化成功")

    def detect_events(self, bg_subtractor_results: dict, frame_timestamp: float = None) -> dict:
        """
        基于背景减除结果检测列车事件

        Args:
            bg_subtractor_results: GMMBackgroundSubtractor.apply_with_roi_analysis返回的结果
            frame_timestamp: 当前帧时间戳

        Returns:
            事件检测结果和状态信息
        """
        if frame_timestamp is None:
            import time
            frame_timestamp = time.time()

        # 检测各ROI区域的活动
        entry_detected = False
        exit_detected = False
        entry_confidence = 0.0
        exit_confidence = 0.0

        # 计算各区域前景比例
        if 'entry_region' in bg_subtractor_results:
            entry_ratio = bg_subtractor_results['entry_region']['foreground_ratio']
            entry_confidence = entry_ratio
            entry_detected = entry_ratio > self.entry_threshold

        if 'exit_region' in bg_subtractor_results:
            exit_ratio = bg_subtractor_results['exit_region']['foreground_ratio']
            exit_confidence = exit_ratio
            exit_detected = exit_ratio > self.exit_threshold

        # 更新状态
        state_result = self.state_manager.update_state(
            entry_detected, exit_detected, frame_timestamp
        )

        # 构建完整结果
        events = {
            'entry_detected': entry_detected,
            'exit_detected': exit_detected,
            'entry_confidence': entry_confidence,
            'exit_confidence': exit_confidence,
            'current_state': state_result['state'],
            'state_changed': state_result['state_changed'],
            'event_triggered': state_result['event_triggered'],
            'event_type': state_result['event_type'],
            'in_cooldown': state_result['in_cooldown'],
            'state_duration': state_result['state_duration']
        }

        # 记录重要事件
        if state_result['event_triggered'] and state_result['event_type'] in ['train_entered', 'train_exited']:
            event_record = {
                'timestamp': frame_timestamp,
                'event_type': state_result['event_type'],
                'state': state_result['new_state'].name if state_result['new_state'] else events['current_state'].name,
                'entry_confidence': entry_confidence,
                'exit_confidence': exit_confidence
            }
            self.event_history.append(event_record)
            print(f"📝 记录事件: {event_record}")

        return events
    
    def visualize_detection(self, frame: np.ndarray, result: dict, events: dict, bg_subtractor: GMMBackgroundSubtractor = None):
        """
        可视化显示检测结果 - 基于原有结构修改
        
        Args:
            frame: 原始帧
            result: apply_with_roi_analysis返回的结果
            events: 事件检测结果
            bg_subtractor: 背景减除器实例（用于获取ROI信息）
        """
        # 获取前景掩码
        fg_mask = result['full_frame']['mask']

        # 在原图上绘制所有ROI区域
        frame_with_rois = frame.copy()
        
        # 使用背景减除器的ROI管理器绘制ROI区域
        if bg_subtractor and hasattr(bg_subtractor, 'roi_manager'):
            # 直接使用ROI管理器中的ROI坐标绘制绿色矩形
            roi_manager = bg_subtractor.roi_manager
            for roi_name, points in roi_manager.rois.items():
                # 绘制绿色矩形边框
                cv2.rectangle(frame_with_rois, points[0], points[1], (0, 255, 0), 2)
                
                # 添加ROI标签 - 修复文字位置问题
                label = roi_name.replace('_', ' ').title()
                
                # 计算文字位置，确保在图像内
                text_x = points[0][0]
                text_y = points[0][1] - 10
                
                # 如果ROI在图像顶部，将文字放在矩形内部
                if text_y < 20:
                    text_y = points[0][1] + 25
                
                cv2.putText(frame_with_rois, label, 
                          (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # 如果没有提供bg_subtractor，使用默认绘制方法
            frame_with_rois = self._draw_rois_default(frame_with_rois, result)

        # 添加状态信息
        y_offset = 30
        status_info = [
            f"State: {events['current_state'].name}",
            f"Duration: {events['state_duration']:.1f}s",
            f"Entry: {events['entry_confidence']:.3f}",
            f"Exit: {events['exit_confidence']:.3f}"
        ]
        
        for text in status_info:
            cv2.putText(frame_with_rois, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # 添加事件状态
        if events['event_triggered']:
            event_text = f"Event: {events['event_type']}"
            cv2.putText(frame_with_rois, event_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30
        
        if events['in_cooldown']:
            cool_text = "COOLDOWN"
            cv2.putText(frame_with_rois, cool_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 根据状态改变边框颜色
        border_color = {
            TrainState.NO_TRAIN: (0, 0, 255),      # 红色
            TrainState.ENTERING: (0, 255, 255),    # 黄色
            TrainState.IN_STATION: (0, 255, 0),    # 绿色
            TrainState.EXITING: (0, 165, 255)      # 橙色
        }[events['current_state']]
        
        # 添加边框
        cv2.rectangle(frame_with_rois, (0, 0), 
                     (frame_with_rois.shape[1]-1, frame_with_rois.shape[0]-1),
                     border_color, 3)
        
        # 显示原图窗口
        cv2.imshow('Train Detection - State Machine', frame_with_rois)
        
        # 显示各ROI区域的前景掩码
        for roi_name, roi_data in result.items():
            if roi_name not in ['full_frame']:
                roi_mask = roi_data['mask']
                roi_ratio = roi_data['foreground_ratio']
                
                roi_display = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
                text = f"{roi_name}: {roi_ratio:.3f}"
                cv2.putText(roi_display, text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow(f'ROI: {roi_name}', roi_display)
    
    def _draw_rois_default(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        默认ROI绘制方法（当没有提供bg_subtractor时使用）
        
        Args:
            frame: 输入帧
            result: 分析结果
            
        Returns:
            绘制了ROI的帧
        """
        frame_with_rois = frame.copy()
        
        # 简单的ROI绘制逻辑
        for roi_name in result.keys():
            if roi_name not in ['full_frame']:
                # 这里可以根据需要添加默认的ROI绘制逻辑
                # 示例：在左上角显示ROI名称
                cv2.putText(frame_with_rois, roi_name, (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame_with_rois

    def process_video_with_detection(self, video_path: str, bg_subtractor: GMMBackgroundSubtractor,
                                     max_frames: int = 100, show_visualization: bool = False) -> Dict:
        """
        使用检测器处理视频

        Args:
            video_path: 视频文件路径
            bg_subtractor: 背景减除器实例
            max_frames: 最大处理帧数
            show_visualization: 是否显示可视化

        Returns:
            处理结果统计
        """
        import time

        if not Path(video_path).exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        frame_count = 0
        start_time = time.time()

        print(f"🎬 开始列车进出站检测: {video_path}")
        print(f"📊 视频信息: {width}x{height}, {fps}FPS")

        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break

            current_time = time.time()
            frame_timestamp = start_time + \
                (frame_count / fps) if fps > 0 else current_time

            # 应用背景减除并分析
            bg_results = bg_subtractor.apply_with_roi_analysis(frame)

            # 检测列车事件
            events = self.detect_events(bg_results, frame_timestamp)

            # 显示可视化 - 修复：传递bg_subtractor参数
            if show_visualization:
                self.visualize_detection(frame, bg_results, events, bg_subtractor)  # 添加bg_subtractor参数
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            if frame_count % 50 == 0:
                status = self.state_manager.get_status()
                print(f"📈 帧: {frame_count}, 状态: {status['current_state']}, "
                      f"进站: {status['entry_count']}, 出站: {status['exit_count']}")

        cap.release()
        if show_visualization:
            cv2.destroyAllWindows()

        # 获取最终状态
        final_status = self.state_manager.get_status()

        stats = {
            'total_frames': frame_count,
            'entry_events': final_status['entry_count'],
            'exit_events': final_status['exit_count'],
            'final_state': final_status['current_state'],
            'event_history': self.event_history,
            'processing_time': time.time() - start_time
        }

        print(f"✅ 列车进出站检测完成")
        print(f"📊 共处理 {frame_count} 帧")
        print(f"🚂 进站事件: {final_status['entry_count']} 次")
        print(f"🚂 出站事件: {final_status['exit_count']} 次")
        print(f"🏁 最终状态: {final_status['current_state']}")

        return stats

    def get_detection_status(self) -> dict:
        """获取当前检测状态"""
        return self.state_manager.get_status()

    def reset_detector(self):
        """重置检测器"""
        self.state_manager.reset()
        self.event_history.clear()
        print("🔄 列车检测器已重置")

    def set_detection_thresholds(self, entry_threshold: float, exit_threshold: float):
        """设置检测阈值"""
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        print(f"⚙️ 设置检测阈值 - 进站: {entry_threshold}, 出站: {exit_threshold}")
