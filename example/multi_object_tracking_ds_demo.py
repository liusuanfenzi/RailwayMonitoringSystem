import time
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
from models.detectors.yolo_detector import YOLODetector
from models.detectors.stay_detector import StayDetector
from models.trackers.multi_object_tracker import MultiObjectTracker
from utils.video.video_utils import ROIManager

class PersonVehicleStayDetection:
    """人员车辆跟踪系统（基于DeepSORT）- ROI区域检测模式"""

    def __init__(self, model_path='yolov5su.pt', conf_threshold=0.5,
                 use_gpu=True, stay_threshold=8,skip_frame_mode=False,detection_interval=3):
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
            max_age=70,
            min_hits=3,
            iou_threshold=0.3,
            max_cosine_distance=0.3,
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
        
        # 跳帧检测模式
        self.skip_frame_mode = skip_frame_mode
        self.detection_interval = detection_interval  # 每3帧检测一次
        self.last_detection_frame = 0
        
        # 缓存上一帧的检测结果
        self.last_detections = np.empty((0, 6))

        print("✅ 人员车辆跟踪系统初始化完成（ROI区域检测模式）")

    def set_detection_roi(self, points):
        """
        设置检测和跟踪的ROI区域
        
        Args:
            points: ROI区域坐标 [(x1,y1), (x2,y2)]
        """
        # 设置检测器ROI
        self.detector.set_roi(points)
        
        # 设置跟踪器ROI
        self.tracker.set_roi(points)
        
        # 同时添加到ROI管理器用于停留检测
        self.roi_manager.add_roi("detection_roi", points)
        
        print(f"🎯 已设置检测和跟踪ROI区域: {points}")

    def disable_roi_detection(self):
        """禁用ROI检测，切换回全图检测"""
        self.detector.disable_roi()
        self.tracker.disable_roi()
        print("🔓 已禁用ROI检测，切换为全图检测模式")
    
    def toggle_skip_frame_mode(self, interval=3):
        """
        切换跳帧检测模式
        
        Args:
            interval: 检测间隔帧数
        """
        self.skip_frame_mode = not self.skip_frame_mode
        self.detection_interval = interval
        self.last_detection_frame = 0
        
        if self.skip_frame_mode:
            print(f"⏩ 启用跳帧检测模式，每 {interval} 帧检测一次")
        else:
            print("🔍 禁用跳帧检测模式，每帧都检测")

    def process_frame(self, frame):
        """
        处理单帧图像（在ROI区域内进行检测和跟踪）
        
        Args:
            frame: 输入图像
            
        Returns:
            result_frame: 处理后的图像
            tracked_objects: 跟踪结果
        """
        start_time = time.time()
        self.frame_counter += 1

        try:
            # 1. 目标检测（在ROI区域内）
            if self.skip_frame_mode:
                # 跳帧检测模式
                if (self.frame_counter - self.last_detection_frame) >= self.detection_interval:
                    # 进行检测
                    detections = self.detector.detect(frame)
                    self.last_detections = detections
                    self.last_detection_frame = self.frame_counter
                    print(f"🔍 检测到 {len(detections)} 个目标 (跳帧模式)")
                else:
                    # 使用上一帧的检测结果
                    detections = self.last_detections
                    print(f"⏩ 跳帧，使用缓存检测结果: {len(detections)} 个目标")
            else:
                # 正常模式，每帧都检测
                detections = self.detector.detect(frame)
                print(f"🔍 检测到 {len(detections)} 个目标")

            # 2. 目标跟踪（在ROI区域内）
            tracked_objects = self.tracker.update(detections, frame)
            print(f"📈 跟踪 {len(tracked_objects)} 个对象")

            # 3. 停留检测
            current_time = float(self.frame_counter) / 30.0
            self.stay_detector.update(tracked_objects, current_time, frame)

            # 4. 可视化跟踪结果 - 使用当前停留和曾经报警的对象
            result_frame = self.tracker.visualize_tracking(
                frame,
                tracked_objects,
                self.stay_detector.get_staying_objects(),
                self.stay_detector.get_alerted_objects()
            )

            # 5. 添加性能信息
            result_frame = self._add_performance_info(result_frame, tracked_objects)

            # 性能统计
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 60:
                self.processing_times.pop(0)

            return result_frame, tracked_objects

        except Exception as e:
            print(f"❌ 帧处理错误: {e}")
            import traceback
            traceback.print_exc()
            return frame, []

    def _add_performance_info(self, frame, tracked_objects):
        """添加性能信息到图像"""
        if not self.processing_times:
            return frame

        avg_time = sum(self.processing_times) / len(self.processing_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0

        # 检测模式显示
        mode = "ROI_mode" if self.detector.roi_active else "full_frame_mode"
        skip_mode = f"skipx{self.detection_interval}" if self.skip_frame_mode else "normal_detection"

        # 性能信息显示
        info_lines = [
            f'FPS: {avg_fps:.1f}',
            f'mode: {mode} | {skip_mode}',
            f'tracked_numbers: {len(tracked_objects)}',
            f'staying_numbers: {self.stay_detector.get_staying_count()}',
            f'alerted_numbers: {len(self.stay_detector.get_alerted_objects())}',
            f'algorithm: DeepSORT'
        ]

        for i, line in enumerate(info_lines):
            color = (0, 255, 0)
            if "staying" in line and self.stay_detector.get_staying_count() > 0:
                color = (0, 0, 255)
            elif "alerted" in line and len(self.stay_detector.get_alerted_objects()) > 0:
                color = (0, 0, 255)

            cv2.putText(frame, line, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def get_performance_stats(self):
        """获取性能统计"""
        if not self.processing_times:
            return "暂无性能数据"

        avg_time = sum(self.processing_times) / len(self.processing_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        mode = "ROI模式" if self.detector.roi_active else "全图模式"
        skip_mode = f"跳帧x{self.detection_interval}" if self.skip_frame_mode else "正常检测"
        
        return f"模式: {mode} | {skip_mode}, 平均处理时间: {avg_time:.3f}s, 平均FPS: {avg_fps:.1f}"

    def reset_stay_detection(self):
        """重置停留检测"""
        self.stay_detector.reset()

    def add_roi(self, name, points):
        """添加ROI区域（仅用于停留检测）"""
        self.roi_manager.add_roi(name, points)
        print(f"✅ 添加停留检测ROI区域: {name}, 坐标: {points}")


def main():
    """主函数 - ROI区域检测模式"""
    print("🚀 启动人员车辆停留检测系统（ROI区域检测模式）")
    print("=" * 50)

    tracker = None

    try:
        # 初始化跟踪系统
        tracker = PersonVehicleStayDetection(
            model_path='yolov5su.pt',
            conf_threshold=0.8,
            use_gpu=True,
            stay_threshold=8,
            skip_frame_mode=True,
            detection_interval=3
        )

        # 设置检测和跟踪的ROI区域
        # detection_roi = [(350, 340), (750, 580)]
        # tracker.set_detection_roi(detection_roi)
        # video_source = "data/test_videos/callpose_test/callpose_test (online-video-cutter.com).mp4"

        # detection_roi = [(200, 200), (700, 600)]
        # tracker.set_detection_roi(detection_roi)
        # video_source = "data/test_videos/safe_gesture/1 (online-video-cutter.com).mp4"

        detection_roi = [(300, 300), (800, 700)]
        tracker.set_detection_roi(detection_roi)
        video_source = "data/test_videos/safe_gesture/gf1 (online-video-cutter.com).mp4"
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_source}")
            return

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📹 视频信息: {fps:.1f} FPS, 总帧数: {total_frames}")

        print("✅ 系统启动成功")
        print("🎯 工作模式: ROI区域检测和跟踪")
        print("🎮 控制说明:")
        print("  - 按 'q' 键退出程序")
        print("  - 按 's' 键保存当前帧")
        print("  - 按 'r' 键重置停留检测")
        print("  - 按 'd' 键切换检测模式（ROI/全图）")
        # print("  - 按 'f' 键切换跳帧检测模式")
        # print("  - 按 '1' 键设置跳帧间隔为2")
        # print("  - 按 '2' 键设置跳帧间隔为3")
        # print("  - 按 '3' 键设置跳帧间隔为5")
        print("=" * 50)

        last_performance_log = 0
        current_mode = "ROI"
        skip_mode = "正常检测"

        while True:
            ret, frame = cap.read()
            if not ret:
                print("📹 视频流结束")
                break

            # 处理帧 - 在ROI区域内进行检测和跟踪
            result_frame, tracked_objects = tracker.process_frame(frame)

            # 显示结果
            window_title = f'PersonVehicleDetectSystem - {current_mode}mode | {skip_mode}'
            cv2.imshow(window_title, result_frame)

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
            elif key == ord('d'):
                # 切换检测模式
                if current_mode == "ROI":
                    tracker.disable_roi_detection()
                    current_mode = "全图"
                else:
                    detection_roi = [(350, 370), (750, 580)]
                    tracker.set_detection_roi(detection_roi)
                    current_mode = "ROI"
                print(f"🔄 切换到{current_mode}检测模式")
            # elif key == ord('f'):
            #     # 切换跳帧检测模式
            #     tracker.toggle_skip_frame_mode()
            #     skip_mode = f"跳帧x{tracker.detection_interval}" if tracker.skip_frame_mode else "正常检测"
            #     print(f"🔄 切换跳帧检测模式: {skip_mode}")
            # elif key == ord('1'):
            #     # 设置跳帧间隔为2
            #     tracker.detection_interval = 2
            #     if tracker.skip_frame_mode:
            #         skip_mode = f"跳帧x{tracker.detection_interval}"
            #         print(f"🔄 设置跳帧间隔为2帧")
            # elif key == ord('2'):
            #     # 设置跳帧间隔为3
            #     tracker.detection_interval = 3
            #     if tracker.skip_frame_mode:
            #         skip_mode = f"跳帧x{tracker.detection_interval}"
            #         print(f"🔄 设置跳帧间隔为3帧")
            # elif key == ord('3'):
            #     # 设置跳帧间隔为5
            #     tracker.detection_interval = 5
            #     if tracker.skip_frame_mode:
            #         skip_mode = f"跳帧x{tracker.detection_interval}"
            #         print(f"🔄 设置跳帧间隔为5帧")

            # 定期显示性能信息
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

# def main():
#     """主函数 - ROI区域检测模式"""
#     print("🚀 启动人员车辆停留检测系统（ROI区域检测模式）")
#     print("=" * 50)

#     tracker = None

#     try:
#         # 初始化跟踪系统
#         tracker = PersonVehicleStayDetection(
#             model_path='yolov5su.pt',
#             conf_threshold=0.8,
#             use_gpu=True,
#             stay_threshold=8
#         )

#         # 设置检测和跟踪的ROI区域

#         detection_roi = [(350, 340), (750, 580)]
#         tracker.set_detection_roi(detection_roi)
#         video_source = "data/test_videos/callpose_test/callpose_test (online-video-cutter.com).mp4"

#         # detection_roi = [(300, 300), (800, 700)]
#         # tracker.set_detection_roi(detection_roi)
#         # video_source = "data/test_videos/safe_gesture/gf1 (online-video-cutter.com).mp4"

#         # detection_roi = [(200, 200), (700, 600)]
#         # tracker.set_detection_roi(detection_roi)
#         # video_source = "data/test_videos/safe_gesture/1 (online-video-cutter.com).mp4"

#         cap = cv2.VideoCapture(video_source)
#         if not cap.isOpened():
#             print(f"❌ 无法打开视频文件: {video_source}")
#             return

#         # 获取视频信息
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         print(f"📹 视频信息: {fps:.1f} FPS, 总帧数: {total_frames}")

#         print("✅ 系统启动成功")
#         print("🎯 工作模式: ROI区域检测和跟踪")
#         print("🎮 控制说明:")
#         print("  - 按 'q' 键退出程序")
#         print("  - 按 's' 键保存当前帧")
#         print("  - 按 'r' 键重置停留检测")
#         print("  - 按 'd' 键切换检测模式（ROI/全图）")
#         print("=" * 50)

#         last_performance_log = 0
#         current_mode = "ROI"

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("📹 视频流结束")
#                 break

#             # 处理帧 - 在ROI区域内进行检测和跟踪
#             result_frame, tracked_objects = tracker.process_frame(frame)

#             # 显示结果
#             cv2.imshow(f'人车停留检测系统 - {current_mode}模式', result_frame)

#             # 键盘控制
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord('s'):
#                 timestamp = time.strftime("%Y%m%d_%H%M%S")
#                 cv2.imwrite(f"system_state_{timestamp}.jpg", result_frame)
#                 print(f"💾 系统状态已保存: system_state_{timestamp}.jpg")
#             elif key == ord('r'):
#                 tracker.reset_stay_detection()
#             elif key == ord('d'):
#                 # 切换检测模式
#                 if current_mode == "ROI":
#                     tracker.disable_roi_detection()
#                     current_mode = "全图"
#                 else:
#                     detection_roi = [(350, 370), (750, 580)]
#                     tracker.set_detection_roi(detection_roi)
#                     current_mode = "ROI"
#                 print(f"🔄 切换到{current_mode}检测模式")

#             # 定期显示性能信息
#             current_time = time.time()
#             if current_time - last_performance_log >= 5:
#                 stats = tracker.get_performance_stats()
#                 print(f"📊 {stats} | 跟踪对象: {len(tracked_objects)}")
#                 last_performance_log = current_time

#     except KeyboardInterrupt:
#         print("\n⏹ 用户中断程序")
#     except Exception as e:
#         print(f"❌ 系统运行错误: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         if 'cap' in locals():
#             cap.release()
#         cv2.destroyAllWindows()
#         if tracker is not None:
#             print(f"📊 最终统计: {tracker.get_performance_stats()}")
#         print("🛑 系统已关闭")


# if __name__ == "__main__":
#     main()