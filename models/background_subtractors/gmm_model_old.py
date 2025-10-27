# """
# 简化的GMM背景减除模型
# """
from pathlib import Path
from typing import Optional, Dict
import cv2
import numpy as np
from typing import Optional
from utils.video.video_utils import ROIManager
from utils.state_manager_old import TrainState, TrainStateManager


class GMMBackgroundSubtractor:
    """简化的GMM背景减除器 - 支持多个ROI区域"""

    def __init__(self, algorithm: str = 'MOG2', **kwargs):
        """
        初始化背景减除器

        Args:
            algorithm: 'MOG2' 或 'KNN'
            **kwargs: 算法参数
        """
        self.algorithm = algorithm.upper()

        # 初始化ROI管理器
        self.roi_manager = ROIManager()

        if self.algorithm == 'MOG2':
            # MOG2参数
            self.history = kwargs.get('history', 500)
            self.var_threshold = kwargs.get('var_threshold', 16)
            self.detect_shadows = kwargs.get('detect_shadows', False)

            self.back_sub = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows
            )
        elif self.algorithm == 'KNN':
            # KNN参数
            self.history = kwargs.get('history', 500)
            self.dist2_threshold = kwargs.get('dist2_threshold', 400)
            self.detect_shadows = kwargs.get('detect_shadows', False)

            self.back_sub = cv2.createBackgroundSubtractorKNN(
                history=self.history,
                dist2Threshold=self.dist2_threshold,
                detectShadows=self.detect_shadows
            )
        else:
            raise ValueError("算法必须是 'MOG2' 或 'KNN'")
        print(f"✅ {self.algorithm}背景减除器初始化成功")

        # 初始化状态管理器
        from utils.state_manager_old import TrainStateManager
        self.state_manager = TrainStateManager(
            min_stay_duration=kwargs.get('min_stay_duration', 5.0),
            cooldown_duration=kwargs.get('cooldown_duration', 3.0),
            entering_timeout=kwargs.get('entering_timeout', 10.0),
            exiting_timeout=kwargs.get('exiting_timeout', 10.0)
        )
        self.event_history = []  # 事件历史记录

    def setup_track_rois(self, entry_roi: list, exit_roi: list):
        """
        设置进出站ROI区域

        Args:
            entry_roi: 进站检测ROI [(x1,y1), (x2,y2)]
            exit_roi: 出站检测ROI [(x1,y1), (x2,y2)]
        """
        if len(entry_roi) != 2 or len(exit_roi) != 2:
            raise ValueError("ROI点必须是两个点 [(x1,y1), (x2,y2)]")

        # 设置进站ROI
        self.roi_manager.add_roi('entry_region', entry_roi)
        # 设置出站ROI
        self.roi_manager.add_roi('exit_region', exit_roi)

        print(f"🎯 设置进站ROI: {entry_roi}")
        print(f"🎯 设置出站ROI: {exit_roi}")

    def setup_single_roi(self, points: list, roi_name: str = 'track_region'):
        """
        设置单个ROI区域（保持向后兼容）

        Args:
            points: 矩形区域点列表 [(x1,y1), (x2,y2)]
            roi_name: ROI区域名称
        """
        if len(points) != 2:
            raise ValueError("ROI点必须是两个点 [(x1,y1), (x2,y2)]")

        self.roi_manager.add_roi(roi_name, points)
        print(f"🎯 设置ROI区域 {roi_name}: {points}")

    # 增强暗色前景检测
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        优化预处理 - 增强暗色前景检测
        """
        if frame is None:
            raise ValueError("输入帧不能为None")

        # Step 1: 转换为灰度图
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame.copy()

        # Step 2: 直方图均衡化 - 增强对比度，特别是暗色区域
        gray_frame = cv2.equalizeHist(gray_frame)

        # Step 3: 应用CLAHE（对比度受限的自适应直方图均衡化）- 更好的暗色区域增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_frame = clahe.apply(gray_frame)

        # Step 4: 应用多个ROI掩膜
        if self.roi_manager.rois:
            mask = np.zeros(gray_frame.shape[:2], dtype=np.uint8)
            for points in self.roi_manager.rois.values():
                cv2.rectangle(mask, points[0], points[1], 255, -1)
            gray_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

        # Step 5: 轻度高斯模糊降噪（避免过度模糊）
        blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)

        return blurred_frame

    def _postprocess_mask(self, fg_mask: np.ndarray) -> np.ndarray:
        """
        优化后处理 - 提高暗色前景检测灵敏度
        """
        # Step 1: 降低二值化阈值，让更多暗色区域被检测为前景
        # 对于MOG2，降低阈值可以检测更多暗色物体
        _, binary_mask = cv2.threshold(
            fg_mask, 100, 255, cv2.THRESH_BINARY)  # 从200降到100

        # Step 2: 形态学闭运算先填充孔洞（暗色区域可能不连续）
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed_mask = cv2.morphologyEx(
            binary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # Step 3: 形态学开运算去除小噪声
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened_mask = cv2.morphologyEx(
            closed_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # Step 4: 中值滤波降噪
        filtered_mask = cv2.medianBlur(opened_mask, 3)  # 使用较小的核

        return filtered_mask

    def apply(self, frame: np.ndarray, learning_rate: float = 0.005) -> np.ndarray:
        """
        应用背景减除

        Args:
            frame: 输入帧
            learning_rate: 学习率

        Returns:
            处理后的前景掩码
        """
        # 预处理
        preprocessed_frame = self._preprocess_frame(frame)

        # 应用背景减除
        fg_mask = self.back_sub.apply(
            preprocessed_frame, learningRate=learning_rate)

        # 后处理
        processed_mask = self._postprocess_mask(fg_mask)

        return processed_mask

    def apply_with_roi_analysis(self, frame: np.ndarray, learning_rate: float = 0.005) -> dict:
        """
        应用背景减除并分析所有ROI区域

        Args:
            frame: 输入帧
            learning_rate: 学习率

        Returns:
            包含所有ROI区域结果的字典
        """
        # 应用背景减除
        fg_mask = self.apply(frame, learning_rate)

        # 计算完整帧统计
        full_foreground_pixels = np.sum(fg_mask > 0)
        full_foreground_ratio = full_foreground_pixels / fg_mask.size

        results = {
            'full_frame': {
                'mask': fg_mask,
                'foreground_pixels': full_foreground_pixels,
                'foreground_ratio': full_foreground_ratio
            }
        }

        # 计算每个ROI区域的统计
        for roi_name in self.roi_manager.rois.keys():
            try:
                roi_mask = self.roi_manager.crop_roi(fg_mask, roi_name)

                roi_size = roi_mask.shape[0] * roi_mask.shape[1]
                roi_foreground_pixels = np.sum(roi_mask > 0)
                roi_foreground_ratio = roi_foreground_pixels / roi_size if roi_size > 0 else 0

                results[roi_name] = {
                    'mask': roi_mask,
                    'foreground_pixels': roi_foreground_pixels,
                    'foreground_ratio': roi_foreground_ratio,
                    'roi_size': roi_size
                }
            except Exception as e:
                print(f"⚠️ ROI分析失败 {roi_name}: {e}")

        return results

    def detect_train_events_with_state(self, result: dict,
                                       entry_threshold: float = 0.05,
                                       exit_threshold: float = 0.05,
                                       frame_timestamp: float = None) -> dict:
        """
        基于状态机的列车事件检测

        Args:
            result: apply_with_roi_analysis返回的结果
            entry_threshold: 进站检测阈值
            exit_threshold: 出站检测阈值
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
        if 'entry_region' in result:
            entry_ratio = result['entry_region']['foreground_ratio']
            entry_confidence = entry_ratio
            entry_detected = entry_ratio > entry_threshold

        if 'exit_region' in result:
            exit_ratio = result['exit_region']['foreground_ratio']
            exit_confidence = exit_ratio
            exit_detected = exit_ratio > exit_threshold

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

    def visualize_comparison_with_state(self, frame: np.ndarray, result: dict, events: dict):
        """
        可视化显示，包含状态信息

        Args:
            frame: 原始帧
            result: apply_with_roi_analysis返回的结果
            events: 事件检测结果
        """
        # 获取前景掩码
        fg_mask = result['full_frame']['mask']

        # 在原图上绘制所有ROI区域
        frame_with_rois = frame.copy()
        frame_with_rois = self.roi_manager.draw_rois(frame_with_rois)

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

    def process_video_with_state_machine(self, video_path: str, max_frames: int = 100,
                                         show_visualization: bool = False,
                                         entry_threshold: float = 0.05,
                                         exit_threshold: float = 0.05) -> Dict:
        """
        使用状态机处理视频

        Args:
            video_path: 视频文件路径
            max_frames: 最大处理帧数
            show_visualization: 是否显示可视化
            entry_threshold: 进站检测阈值
            exit_threshold: 出站检测阈值

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

        print(f"🎬 开始状态机视频处理: {video_path}")
        print(f"📊 视频信息: {width}x{height}, {fps}FPS")

        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break

            current_time = time.time()
            frame_timestamp = start_time + \
                (frame_count / fps) if fps > 0 else current_time

            # 应用背景减除并分析
            result = self.apply_with_roi_analysis(frame)

            # 基于状态机检测事件
            events = self.detect_train_events_with_state(
                result, entry_threshold, exit_threshold, frame_timestamp
            )

            # 显示可视化
            if show_visualization:
                self.visualize_comparison_with_state(frame, result, events)
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

        print(f"✅ 状态机处理完成")
        print(f"📊 共处理 {frame_count} 帧")
        print(f"🚂 进站事件: {final_status['entry_count']} 次")
        print(f"🚂 出站事件: {final_status['exit_count']} 次")
        print(f"🏁 最终状态: {final_status['current_state']}")

        return stats

    def get_state_info(self) -> dict:
        """获取当前状态信息"""
        return self.state_manager.get_status()

    def reset_state(self):
        """重置状态机"""
        self.state_manager.reset()
        self.event_history.clear()
