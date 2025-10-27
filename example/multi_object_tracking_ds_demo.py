import time
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
# 导入自定义模块
from models.detectors.yolo_detector import YOLODetector
from models.detectors.stay_detector import StayDetector
from models.trackers.multi_object_tracker import MultiObjectTracker
from utils.video.video_utils import ROIManager


class PersonVehicleStayDetection:
    """人员车辆跟踪系统（基于DeepSORT）"""

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

        # 使用DeepSORT跟踪器
        self.tracker = MultiObjectTracker(
            max_age=70,  # DeepSORT建议值
            min_hits=3,
            iou_threshold=0.3,
            max_cosine_distance=0.3,  # 外观特征匹配阈值
            # max_age=70,  # DeepSORT建议值
            # min_hits=2,
            # iou_threshold=0.4,
            # max_cosine_distance=0.35,  # 外观特征匹配阈值
            use_gpu=use_gpu
        )

        self.roi_manager = ROIManager()

        # 停留检测器使用ROI管理器
        self.stay_detector = StayDetector(
            roi_manager=self.roi_manager,
            alert_dir="alerts",
            stay_threshold=stay_threshold,
            movement_threshold=5,
            min_frames=10
        )

        # 性能监控
        self.processing_times = []
        self.frame_counter = 0

        print("✅ 人员车辆跟踪系统初始化完成（DeepSORT版本）")

    def process_frame(self, frame):
        """
        处理单帧图像

        Args:
            frame: 输入图像

        Returns:
            result_frame: 处理后的图像
            tracked_objects: 跟踪结果
        """
        start_time = time.time()
        self.frame_counter += 1

        try:
            # 1. 目标检测
            detections = self.detector.detect(frame)
            print(f"🔍 检测到 {len(detections)} 个目标")

            # 2. 过滤ROI区域外的检测结果
            roi_detections = self.tracker.filter_detections_by_roi(
                detections, self.roi_manager)
            print(f"🎯 ROI区域内 {len(roi_detections)} 个目标")

            # 3. 目标跟踪（DeepSORT需要传入frame进行特征提取）
            tracked_objects = self.tracker.update(roi_detections, frame)
            print(f"📈 跟踪 {len(tracked_objects)} 个对象")

            # 调试：检查跟踪对象的类型
            # if len(tracked_objects) > 0:
            #     sample_obj = tracked_objects[0]
            #     print(f"🔧 跟踪对象样本: {sample_obj}, 类型: {[type(x) for x in sample_obj]}")

            # 4. 停留检测
            current_time = float(self.frame_counter) / 30.0
            print(f"⏰ 当前时间戳: {current_time} ")
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

            # 确保返回的是列表，不是numpy数组
            return result_frame, tracked_objects

        except Exception as e:
            print(f"❌ 帧处理错误: {e}")
            import traceback
            traceback.print_exc()
            # 确保在异常情况下也返回两个值
            return frame, []

    def _add_performance_info(self, frame, tracked_objects):
        """添加性能信息到图像"""
        if not self.processing_times:
            return frame

        avg_time = sum(self.processing_times) / len(self.processing_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0

        # 性能信息显示
        info_lines = [
            f'FPS: {avg_fps:.1f}',
            f'ROI Tracks: {len(tracked_objects)}',
            f'Staying: {self.stay_detector.get_staying_count()}',
            f'Tracker: DeepSORT'
        ]

        for i, line in enumerate(info_lines):
            color = (0, 255, 0)
            if "Staying" in line and self.stay_detector.get_staying_count() > 0:
                color = (0, 0, 255)

            cv2.putText(frame, line, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
    print("🚀 启动人员车辆停留检测系统（DeepSORT版本）")
    print("=" * 50)

    tracker = None

    try:
        # 初始化跟踪系统
        tracker = PersonVehicleStayDetection(
            model_path='yolov5su.pt',
            conf_threshold=0.6,
            use_gpu=True,
            stay_threshold=8
        )

        # 添加ROI区域

        # 视频源选择
        # tracker.add_roi("monitor_area", [(300, 300), (800, 700)])
        # video_source = "data/test_videos/safe_gesture/gf1 (online-video-cutter.com).mp4"
        # tracker.add_roi("monitor_area", [(200, 200), (700, 600)])
        # video_source = "data/test_videos/safe_gesture/1 (online-video-cutter.com).mp4"
        tracker.add_roi("monitor_area", [(350, 370), (750, 580)])
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
        print("🎯 跟踪算法: DeepSORT")
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

            # 处理帧 - DeepSORT会自动在内部使用frame进行特征提取
            result_frame, tracked_objects = tracker.process_frame(frame)

            # 显示结果
            cv2.imshow('人车停留检测系统 - DeepSORT', result_frame)

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
