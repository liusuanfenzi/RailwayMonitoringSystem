import cv2
import time
import numpy as np
from pathlib import Path

# 导入自定义模块
from models.detectors.yolo_detector import YOLODetector
from models.detectors.stay_detector import StayDetector
from models.trackers.multi_object_tracker import MultiObjectTracker
from utils.video.video_utils import ROIManager


class PersonVehicleTrackingSystem:
    """人员车辆跟踪系统（仅跟踪ROI区域内目标）"""

    def __init__(self, model_path='yolov5su.pt', conf_threshold=0.5,
                 use_gpu=True, stay_threshold=8):
        """
        初始化跟踪系统

        Args:
            model_path: YOLO模型路径
            conf_threshold: 检测置信度阈值
            use_gpu: 是否使用GPU
            stay_threshold: 停留时间阈值
        """
        # 初始化各个模块
        self.detector = YOLODetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            use_gpu=use_gpu
        )

        self.tracker = MultiObjectTracker(
            max_age=20,
            min_hits=3,
            iou_threshold=0.3
        )

        self.roi_manager = ROIManager()

        # 停留检测器使用ROI管理器
        self.stay_detector = StayDetector(
            stay_threshold=stay_threshold,
            roi_manager=self.roi_manager,  # 传入ROI管理器
            alert_dir="alerts"
        )

        # 性能监控
        self.processing_times = []
        self.frame_counter = 0

        print("✅ 人员车辆跟踪系统初始化完成（ROI区域跟踪模式）")

    def process_frame(self, frame):
        """
        处理单帧图像（仅处理ROI区域内目标）

        Args:
            frame: 输入图像

        Returns:
            result_frame: 处理后的图像
            tracked_objects: 跟踪结果（仅ROI区域内）
        """
        start_time = time.time()
        self.frame_counter += 1

        try:
            # 1. 目标检测（全图检测）
            detections = self.detector.detect(frame)
            # 2. 过滤ROI区域外的检测结果
            roi_detections = self.tracker.filter_detections_by_roi(
                detections, self.roi_manager)
            # 3. 目标跟踪（仅跟踪ROI区域内目标）
            tracked_objects = self.tracker.update(roi_detections)
            # 4. 停留检测（仅检测ROI区域内目标）
            current_time = self.frame_counter / 30.0  # 视频FPS为30
            self.stay_detector.update(tracked_objects, current_time, frame)
            # 5. 可视化跟踪结果
            result_frame = self.tracker.visualize_tracking(
                frame,
                tracked_objects,
                self.stay_detector.get_staying_objects()
            )
            # 6. 绘制ROI区域
            result_frame = self.roi_manager.draw_rois(result_frame)
            # 7. 添加性能信息
            result_frame = self._add_performance_info(
                result_frame, tracked_objects)
            # 性能统计
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 60:
                self.processing_times.pop(0)
            return result_frame, tracked_objects
        except Exception as e:
            print(f"❌ 帧处理错误: {e}")
            return frame, np.empty((0, 5))

    def _add_performance_info(self, frame, tracked_objects):
        """添加性能信息到图像"""
        if not self.processing_times:
            return frame

        # 平均处理时间=总处理时间/处理帧数（处理1帧需要多少秒）
        avg_time = sum(self.processing_times) / len(self.processing_times)
        # 平均FPS=1/平均处理时间（每秒处理多少帧）
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0

        # FPS显示
        cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 跟踪数量显示（ROI区域内）
        cv2.putText(frame, f'ROI Tracks: {len(tracked_objects)}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 停留数量显示
        stay_count = self.stay_detector.get_staying_count()
        color = (0, 0, 255) if stay_count > 0 else (0, 255, 0)
        cv2.putText(frame, f'Staying: {stay_count}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ROI区域数量显示
        roi_count = len(self.roi_manager.get_roi_names())
        cv2.putText(frame, f'ROI Areas: {roi_count}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame

    def get_performance_stats(self):
        """获取性能统计"""
        if not self.processing_times:
            return "暂无性能数据"

        avg_time = sum(self.processing_times) / len(self.processing_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        return f"平均处理时间: {avg_time:.3f}s, 平均FPS: {avg_fps:.1f}"

    def reset_stay_detection(self):
        """重置停留检测"""
        self.stay_detector.reset()

    def add_roi(self, name, points):
        """添加ROI区域"""
        self.roi_manager.add_roi(name, points)
        print(f"✅ 添加ROI区域: {name}, 坐标: {points}")

def main():
    """主函数"""
    print("🚀 启动人员车辆停留检测系统（ROI区域模式）")
    print("=" * 50)

    tracker = None

    try:
        # 初始化跟踪系统
        tracker = PersonVehicleTrackingSystem(
            model_path='yolov5su.pt',
            conf_threshold=0.8,
            use_gpu=True,
            stay_threshold=5
        )

        # 添加ROI区域（只跟踪这个区域内的目标）
        tracker.add_roi("monitor_area", [(400, 300), (700, 600)])

        # 视频源选择
        video_source = "data/test_videos/callpose_test/callpose_test (online-video-cutter.com).mp4"
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_source}")
            return

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📹 视频信息: {fps:.1f} FPS, 总帧数: {total_frames}")

        print("✅ 系统启动成功")
        print("🎯 工作模式: 仅跟踪ROI区域内目标")
        print("🎮 控制说明:")
        print("  - 按 'q' 键退出程序")
        print("  - 按 's' 键保存当前帧")
        print("  - 按 'r' 键重置停留检测")
        print("=" * 50)

        last_performance_log = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("📹 视频流结束")
                break

            # 处理帧（只处理ROI区域内目标）
            result_frame, tracked_objects = tracker.process_frame(frame)

            # 显示结果
            cv2.imshow('人车停留检测系统 - ROI模式', result_frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"system_state_{timestamp}.jpg", result_frame)
                print(f"💾 系统状态已保存: system_state_{timestamp}.jpg")
            elif key == ord('r'):
                tracker.reset_stay_detection()

            # 定期显示性能信息
            current_time = time.time()
            if current_time - last_performance_log >= 5:
                stats = tracker.get_performance_stats()
                print(f"📊 {stats} | ROI跟踪对象: {len(tracked_objects)}")
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