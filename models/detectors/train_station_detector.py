# """
# 简化的列车进出站检测器 - 基于单个ROI区域
# """
from typing import Dict
import cv2
import numpy as np
from pathlib import Path
from models.background_subtractors.gmm_model import GMMBackgroundSubtractor
from utils.state_manager import TrainStateManager

class TrainStationDetector:
    """简化的列车进出站检测器 - 基于单个ROI区域"""

    def __init__(self, **kwargs):
        """
        初始化列车检测器

        Args:
            **kwargs: 状态机参数
        """
        # 初始化状态管理器
        self.state_manager = TrainStateManager(
            spatial_threshold=kwargs.get('spatial_threshold', 0.05),
            temporal_frames=kwargs.get('temporal_frames', 100),
            temporal_threshold=kwargs.get('temporal_threshold', 90)
        )

        self.event_history = []  # 事件历史记录

        print("✅ 简化列车进出站检测器初始化成功")

    def detect_events(self, bg_subtractor_results: dict, frame_index: int) -> dict:
        """
        基于背景减除结果检测列车事件

        Args:
            bg_subtractor_results: GMMBackgroundSubtractor.apply_with_roi_analysis返回的结果
            frame_index: 当前帧索引

        Returns:
            事件检测结果和状态信息
        """
        # 获取ROI区域的置信度
        confidence = 0.0
        roi_name = None

        # 查找ROI区域结果
        for key in bg_subtractor_results.keys():
            if key != 'full_frame':
                roi_name = key
                confidence = bg_subtractor_results[key]['foreground_ratio']
                break

        # 更新状态
        state_result = self.state_manager.update_state(confidence, frame_index)

        # 构建完整结果
        events = {
            'confidence': confidence,
            'spatial_detected': state_result['spatial_detected'],
            'current_state': state_result['state'],
            'event_triggered': state_result['event_triggered'],
            'event_type': state_result.get('event_type', None),
            'buffer_size': state_result['buffer_size'],
            'roi_name': roi_name
        }

        # 记录重要事件
        if state_result['event_triggered']:
            event_record = {
                'frame_index': frame_index,
                'event_type': state_result['event_type'],
                'confidence': confidence,
                'true_count': state_result.get('true_count', 0)
            }
            self.event_history.append(event_record)
            print(f"📝 记录事件: {event_record}")

        return events

    def visualize_detection(self, frame: np.ndarray, result: dict, events: dict, bg_subtractor: GMMBackgroundSubtractor = None):
        """
        可视化显示检测结果

        Args:
            frame: 原始帧
            result: apply_with_roi_analysis返回的结果
            events: 事件检测结果
            bg_subtractor: 背景减除器实例
        """
        # 获取前景掩码
        fg_mask = result['full_frame']['mask']

        # 在原图上绘制ROI区域
        frame_with_rois = frame.copy()

        # 使用背景减除器的ROI管理器绘制ROI区域
        if bg_subtractor and hasattr(bg_subtractor, 'roi_manager'):
            roi_manager = bg_subtractor.roi_manager
            for roi_name, points in roi_manager.rois.items():
                # 绘制绿色矩形边框
                cv2.rectangle(frame_with_rois,
                            points[0], points[1], (0, 255, 0), 2)

                # 添加ROI标签
                label = roi_name.replace('_', ' ').title()
                text_x = points[0][0]
                text_y = points[0][1] - 10

                if text_y < 20:
                    text_y = points[0][1] + 25

                cv2.putText(frame_with_rois, label,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 添加状态信息
        y_offset = 30
        status_info = [
            f"State: {events['current_state'].name}",
            f"Confidence: {events['confidence']:.3f}",
            f"Spatial Detected: {events['spatial_detected']}",
            f"Buffer Size: {events['buffer_size']}"
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

        # 根据状态改变边框颜色
        border_color = (
            0, 255, 0) if events['event_triggered'] else (0, 0, 255)

        # 添加边框
        cv2.rectangle(frame_with_rois, (0, 0),
                    (frame_with_rois.shape[1]-1, frame_with_rois.shape[0]-1),
                    border_color, 3)

        # 显示原图窗口
        cv2.imshow('Simple Train Detection', frame_with_rois)

        # 显示ROI区域的后处理前景掩码
        if events['roi_name'] and events['roi_name'] in result:
            roi_data = result[events['roi_name']]
            roi_mask = roi_data['mask']
            roi_ratio = roi_data['foreground_ratio']

            roi_display = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
            text = f"{events['roi_name']} (Post-processed): {roi_ratio:.3f}"
            cv2.putText(roi_display, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(f'ROI: {events["roi_name"]} (Post-processed)', roi_display)
        
        # 新增：显示未做后处理的ROI区域前景掩码
        if bg_subtractor and events['roi_name']:
            try:
                # 获取未后处理的掩码
                # 先进行预处理
                preprocessed_frame = bg_subtractor._preprocess_frame(frame)
                # 应用背景减除但不进行后处理
                raw_fg_mask = bg_subtractor.back_sub.apply(preprocessed_frame, learningRate=0.005)
                
                # 裁剪ROI区域
                roi_manager = bg_subtractor.roi_manager
                raw_roi_mask = roi_manager.crop_roi(raw_fg_mask, events['roi_name'])
                
                # 计算未后处理的ROI前景比例
                raw_roi_size = raw_roi_mask.shape[0] * raw_roi_mask.shape[1]
                raw_roi_foreground_pixels = np.sum(raw_roi_mask > 0)
                raw_roi_foreground_ratio = raw_roi_foreground_pixels / raw_roi_size if raw_roi_size > 0 else 0
                
                # 显示未后处理的ROI掩码
                raw_roi_display = cv2.cvtColor(raw_roi_mask, cv2.COLOR_GRAY2BGR)
                raw_text = f"{events['roi_name']} (Raw): {raw_roi_foreground_ratio:.3f}"
                cv2.putText(raw_roi_display, raw_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow(f'ROI: {events["roi_name"]} (Raw - No Post-processing)', raw_roi_display)
                
                # 在控制台输出两种掩码的对比信息
                if events['event_triggered']:
                    print(f"🔍 掩码对比 - 帧 {events.get('frame_index', 'N/A')}:")
                    print(f"   预处理+后处理掩码前景比例: {roi_ratio:.4f}")
                    print(f"   预处理掩码前景比例: {raw_roi_foreground_ratio:.4f}")
                    print(f"   差异: {abs(roi_ratio - raw_roi_foreground_ratio):.4f}")
                    
            except Exception as e:
                print(f"⚠️ 显示原始ROI掩码失败: {e}")
    
    def process_video_with_detection(self, video_path: str, bg_subtractor: GMMBackgroundSubtractor,
                                     max_frames: int = 1000, show_visualization: bool = True) -> Dict:
        """
        使用检测器处理视频
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        frame_count = 0
        print(f"🎬 开始简化列车进出站检测: {video_path}")
        
        # 添加背景模型预热期
        print("🔥 预热背景模型...")
        warmup_frames = 30
        for i in range(warmup_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # 使用较高的学习率快速建立背景模型
            bg_subtractor.apply(frame, learning_rate=0.1)
            frame_count += 1
        
        print(f"✅ 背景模型预热完成，已处理 {warmup_frames} 帧")
        
        # 重置帧计数
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break

            # 应用背景减除并分析
            bg_results = bg_subtractor.apply_with_roi_analysis(frame)

            # 检测列车事件
            events = self.detect_events(bg_results, frame_count)

            # 显示可视化
            if show_visualization:
                self.visualize_detection(frame, bg_results, events, bg_subtractor)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            if frame_count % 50 == 0:
                status = self.state_manager.get_status()
                print(f"📈 帧: {frame_count}, 状态: {status['current_state']}, "
                    f"进站: {status['entry_count']}, 缓冲区: {status['buffer_size']}")


        cap.release()
        if show_visualization:
            cv2.destroyAllWindows()

        # 获取最终状态
        final_status = self.state_manager.get_status()

        stats = {
            'total_frames': frame_count,
            'entry_events': final_status['entry_count'],
            'final_state': final_status['current_state'],
            'event_history': self.event_history
        }

        print(f"✅ 简化列车进出站检测完成")
        print(f"📊 共处理 {frame_count} 帧")
        print(f"🚂 进站事件: {final_status['entry_count']} 次")

        return stats

    def get_detection_status(self) -> dict:
        """获取当前检测状态"""
        return self.state_manager.get_status()

    def reset_detector(self):
        """重置检测器"""
        self.state_manager.reset()
        self.event_history.clear()
        print("🔄 列车检测器已重置")
