# models/jetson_train_detector_visual.py
import cv2
import numpy as np
from pathlib import Path
from typing import Dict
from datetime import datetime
from models.background_subtractor.gmm_model import GMMBackgroundSubtractor
from utils.state_manager import TrainStateManager
from utils.utils import PerformanceMonitor
from utils.output_manager import OutputManager


class TrainDetector:
    """Jetson优化的列车进出站检测器 - 可视化版本"""

    def __init__(self, **kwargs):
        # 初始化状态管理器
        self.state_manager = TrainStateManager(
            spatial_threshold=kwargs.get('spatial_threshold', 0.05),
            temporal_frames=kwargs.get('temporal_frames', 50),
            temporal_threshold=kwargs.get('temporal_threshold', 45)
        )

        self.performance_monitor = PerformanceMonitor()
        self.output_manager = OutputManager()
        self.event_history = []
        self.frame_count = 0

        # 可视化配置
        self.original_window = "Original Frame with ROI"
        self.mask_window = "ROI Foreground Mask"
        self.show_visualization = True
        self.print_interval = kwargs.get('print_interval', 10)
        self.last_print_frame = 0

        print("✅ Jetson列车检测器(可视化版本)初始化成功")

    def detect_events(self, bg_subtractor_results: dict, frame: np.ndarray, frame_index: int) -> dict:
        """优化事件检测，支持截图保存"""
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

        events = {
            'confidence': confidence,
            'spatial_detected': state_result['spatial_detected'],
            'current_state': state_result['state'],
            'event_triggered': state_result['event_triggered'],
            'event_type': state_result.get('event_type', None),
            'roi_name': roi_name,
            'frame_index': frame_index
        }

        # 记录重要事件
        if state_result['event_triggered']:
            event_record = {
                'frame_index': frame_index,
                'event_type': state_result['event_type'],
                'confidence': confidence,
            }
            self.event_history.append(event_record)

            # 保存事件截图 - 使用ROI前景掩码
            self._save_event_snapshot(frame, events, bg_subtractor_results)

            print(f"🚂 检测到列车进站! 帧: {frame_index}, 置信度: {confidence:.3f}")

        return events

    def _save_event_snapshot(self, frame: np.ndarray, events: dict, bg_results: dict):
        """保存事件截图 - 使用ROI前景掩码"""
        try:
            # 获取ROI前景掩码
            if 'roi_name' not in events or events['roi_name'] not in bg_results:
                return None

            roi_data = bg_results[events['roi_name']]
            roi_mask = roi_data['mask']

            # 直接保存原始掩码，不添加标注
            success = self.output_manager.save_event_frame(
                frame=roi_mask,
                event_type=events['event_type'],
                confidence=events['confidence'],
                frame_index=events['frame_index'],
                subfolder="train_detection"
            )

            if not success:
                print(f"⚠️ 保存截图失败，请检查目录权限")

        except Exception as e:
            print(f"⚠️ 保存事件截图失败: {e}")

    def _create_original_display(self, frame: np.ndarray, events: dict, bg_subtractor: GMMBackgroundSubtractor) -> np.ndarray:
        """创建原始帧显示，只保留ROI框，移除所有文字标注"""
        display_frame = frame.copy()

        # 绘制ROI区域
        if hasattr(bg_subtractor, 'roi_manager') and bg_subtractor.roi_manager.rois:
            for roi_name, points in bg_subtractor.roi_manager.rois.items():
                # 根据事件触发状态选择颜色
                color = (0, 255, 0) if events['event_triggered'] else (0, 0, 255)
                cv2.rectangle(display_frame, points[0], points[1], color, 2)

        return display_frame

    def _create_mask_display(self, events: dict, bg_results: dict) -> np.ndarray:
        """创建ROI前景掩码显示，移除所有文字标注"""
        if 'roi_name' not in events or events['roi_name'] not in bg_results:
            # 如果没有掩码，返回黑色图像
            return np.zeros((480, 640, 3), dtype=np.uint8)

        roi_data = bg_results[events['roi_name']]
        roi_mask = roi_data['mask']

        # 将二值掩码转换为彩色图像
        if len(roi_mask.shape) == 2:
            mask_display = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_display = roi_mask.copy()

        return mask_display

    def _print_confidence_info(self, confidence: float, frame_index: int, events: dict):
        """打印置信度信息到控制台"""
        if (frame_index - self.last_print_frame >= self.print_interval or
            events['event_triggered'] or
                confidence > 0.1):

            status_info = f"frame {frame_index}: confidence={confidence:.3f}"

            if events['spatial_detected']:
                status_info += " [detecting]"
            if events['event_triggered']:
                status_info += " [event triggered!]"

            print(status_info)
            self.last_print_frame = frame_index

    def process_video_with_visualization(self, video_path: str,
                                         bg_subtractor: GMMBackgroundSubtractor,
                                         max_frames: int = 1000) -> Dict:
        """
        Jetson优化的视频处理 - 可视化版本（无文字标注）
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        self.frame_count = 0
        print(f"🎬 开始Jetson列车检测(可视化版本): {video_path}")

        # 创建两个独立窗口
        if self.show_visualization:
            try:
                cv2.namedWindow(self.original_window, cv2.WINDOW_NORMAL)
                cv2.namedWindow(self.mask_window, cv2.WINDOW_NORMAL)
                
                # 设置窗口位置（避免重叠）
                cv2.moveWindow(self.original_window, 100, 100)
                cv2.moveWindow(self.mask_window, 800, 100)
                
                print("✅ 双窗口可视化已创建（无文字标注）")
            except Exception as e:
                print(f"⚠️ 创建可视化窗口失败: {e}")
                self.show_visualization = False

        # 简化的背景模型预热
        print("🔥 预热背景模型...")
        warmup_frames = 15
        for i in range(warmup_frames):
            ret, frame = cap.read()
            if not ret:
                break
            bg_subtractor.apply(frame, learning_rate=0.1)

        print(f"✅ 背景模型预热完成")

        # 重置帧计数
        self.frame_count = 0
        self.last_print_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret or self.frame_count >= max_frames:
                break

            # 应用背景减除并分析
            start_time = self.performance_monitor.start_timing()
            bg_results = bg_subtractor.apply_with_roi_analysis(frame)
            self.performance_monitor.end_timing(start_time, "背景分析")

            # 检测列车事件
            events = self.detect_events(bg_results, frame, self.frame_count)

            # 打印置信度信息到控制台
            self._print_confidence_info(
                events['confidence'], self.frame_count, events)

            # 显示双窗口可视化（无文字标注）
            if self.show_visualization:
                try:
                    # 创建两个独立的显示图像（无文字标注）
                    original_display = self._create_original_display(
                        frame, events, bg_subtractor)
                    mask_display = self._create_mask_display(events, bg_results)

                    # 分别显示在两个窗口中
                    cv2.imshow(self.original_window, original_display)
                    cv2.imshow(self.mask_window, mask_display)

                    # 处理键盘输入
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("⏹️ 用户请求退出")
                        break
                    elif key == ord('s'):
                        # 手动保存截图
                        self._save_event_snapshot(frame, events, bg_results)
                        print("💾 手动保存截图")

                except Exception as e:
                    print(f"⚠️ 可视化显示失败: {e}")
                    self.show_visualization = False

            self.frame_count += 1

            # 减少状态打印频率
            if self.frame_count % 100 == 0 or events['event_triggered']:
                status = self.state_manager.get_status()
                perf_stats = self.performance_monitor.get_performance_stats()
                print(f"📈 帧: {self.frame_count}, 状态: {status['current_state']}, "
                      f"进站: {status['entry_count']}, FPS: {perf_stats['fps']:.1f}")

        cap.release()

        # 关闭显示窗口
        if self.show_visualization:
            try:
                cv2.destroyAllWindows()
            except:
                pass

        # 获取最终状态
        final_status = self.state_manager.get_status()
        perf_stats = self.performance_monitor.get_performance_stats()

        stats = {
            'total_frames': self.frame_count,
            'entry_events': final_status['entry_count'],
            'final_state': final_status['current_state'],
            'avg_fps': perf_stats['fps'],
            'event_history': self.event_history,
            'saved_snapshots': len(self.event_history)
        }

        print(f"✅ Jetson列车检测完成")
        print(f"📊 共处理 {self.frame_count} 帧, 平均FPS: {perf_stats['fps']:.1f}")
        print(f"🚂 进站事件: {final_status['entry_count']} 次")
        print(f"💾 保存ROI掩码截图: {len(self.event_history)} 张")

        # 显示截图保存位置
        output_path = self.output_manager.base_output_dir / "train_detection"
        print(f"📁 ROI掩码截图位置: {output_path.absolute()}")

        return stats

    def get_detection_status(self) -> dict:
        """获取检测状态"""
        status = self.state_manager.get_status()
        perf_stats = self.performance_monitor.get_performance_stats()
        status.update(perf_stats)
        status['total_frames'] = self.frame_count
        status['saved_snapshots'] = len(self.event_history)
        return status

    def reset_detector(self):
        """重置检测器"""
        self.state_manager.reset()
        self.event_history.clear()
        self.frame_count = 0
        self.last_print_frame = 0
        print("🔄 列车检测器已重置")
