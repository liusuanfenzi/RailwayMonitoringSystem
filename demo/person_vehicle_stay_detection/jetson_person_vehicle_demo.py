#!/usr/bin/env python3
"""
Jetson人车停留检测演示程序 - 零PyTorch版本
使用TensorRT YOLO + IoU-Only DeepSORT + StayDetector
"""

import argparse
import cv2
import os
import time
import numpy as np
import gc
from pathlib import Path

# 零PyTorch的检测器 & 跟踪器
from models.detector.yolo_detector import JetsonYOLODetectorTensorRT as JetsonYOLODetector
from models.tracker.multi_object_tracker import MultiObjectTrackerTensorRT as MultiObjectTracker
from models.detector.stay_detector import StayDetector  

# 简单的工具类替代
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


def setup_jetson_environment():
    """设置运行环境（无桌面也可跑）"""
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':0'
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    print("🔧 Jetson环境设置完成（零PyTorch）")


def cleanup_resources(cap=None, detector=None, tracker=None, stay_detector=None):
    """清理资源"""
    print("🧹 清理资源...")
    if cap: 
        cap.release()
    cv2.destroyAllWindows()
    if detector:
        try:
            stats = detector.get_performance_stats()
            print(f"📊 最终性能: {stats['avg_fps']:.1f}FPS")
        except: 
            pass
        detector.cleanup()
    if stay_detector:
        stay_detector.reset()
    gc.collect()
    print("✅ 资源清理完成")

# ---------------- 主函数 ----------------
def main():
    parser = argparse.ArgumentParser(description='Jetson人车停留检测演示（零PyTorch）')
    parser.add_argument('--engine', type=str, default='yolov8n.engine',
                        help='TensorRT引擎路径')
    parser.add_argument('--video', type=str,
                        default="data/test_videos/callpose_test/callpose_test.mp4",
                        help='视频文件路径')
    parser.add_argument('--detection_roi', type=str, default='350,340,750,580',
                        help='检测ROI坐标 x1,y1,x2,y2')
    parser.add_argument('--stay_roi', type=str,
                        help='停留ROI坐标（可选）')
    parser.add_argument('--conf', type=float, default=0.6,
                        help='检测置信度阈值')
    parser.add_argument('--stay_threshold', type=float, default=10.0,
                        help='停留阈值（秒）')  
    parser.add_argument('--movement_threshold', type=float, default=15.0,
                        help='移动阈值（像素）')
    parser.add_argument('--min_frames', type=int, default=5,
                        help='最小连续静止帧数')
    parser.add_argument('--detection_interval', type=int, default=3,
                        help='检测间隔帧数')
    parser.add_argument('--max_frames', type=int, default=1000,
                        help='最大处理帧数')
    parser.add_argument('--frame_skip', type=int, default=0,
                        help='额外跳帧数量')
    parser.add_argument('--save', type=str, default='video/output.mp4',
                        help='保存路径（空则不保存）')
    args = parser.parse_args()

    setup_jetson_environment()

    # 解析ROI
    try:
        detection_coords = [int(x) for x in args.detection_roi.split(',')]
        detection_points = [(detection_coords[0], detection_coords[1]),
                            (detection_coords[2], detection_coords[3])]
        stay_points = None
        if args.stay_roi:
            stay_coords = [int(x) for x in args.stay_roi.split(',')]
            stay_points = [(stay_coords[0], stay_coords[1]),
                           (stay_coords[2], stay_coords[3])]
    except Exception as e:
        print(f"❌ ROI解析错误: {e}")
        return

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ 视频不存在: {args.video}")
        return

    print("🚀 启动Jetson人车检测系统（零PyTorch）")
    print(f"📹 视频: {args.video}")
    print(f"🎯 检测ROI: {detection_points}")
    if stay_points: 
        print(f"🎯 停留ROI: {stay_points}")
    print(f"🔧 引擎: {args.engine}")
    print(f"⚙️ 置信度: {args.conf}")
    print(f"⏱️ 停留阈值: {args.stay_threshold}秒")  # 显示秒数
    print(f"🏃 移动阈值: {args.movement_threshold}像素")
    print(f"⏩ 检测间隔: 每{args.detection_interval}帧")
    print("-" * 60)

    cap = None
    detector = None
    tracker = None
    stay_detector = None
    
    try:
        # 1. 初始化系统（零PyTorch）
        detector = JetsonYOLODetector(args.engine, conf_threshold=args.conf)
        tracker = MultiObjectTracker(max_age=50, min_hits=2, iou_threshold=0.3, use_gpu=True)
        roi_manager = JetsonROIManager()
        perf_mon = JetsonPerformanceMonitor()

        # 2. 设置ROI
        detector.set_roi(detection_points)
        tracker.set_roi(detection_points)
        roi_manager.add_roi("detection_roi", detection_points)
        if stay_points:
            roi_manager.add_roi("stay_roi", stay_points)

        # 3. 初始化StayDetector
        stay_detector = StayDetector(
            stay_threshold=args.stay_threshold,
            movement_threshold=args.movement_threshold,
            min_frames=args.min_frames,
            roi_manager=roi_manager,
            alert_dir="alerts"
        )

        # 4. 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): 
            print("❌ 无法打开视频")
            return
            
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        # 视频写入器
        writer = None
        if args.save:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

        print("✅ 系统启动成功")
        print("🎮 按键: q=退出  r=重置  s=切换跳帧  c=清理内存")
        print("-" * 60)

        frame_count = 0
        last_log_time = time.time()
        last_frame_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= args.max_frames: 
                break
                
            # 额外跳帧
            if args.frame_skip > 0 and frame_count % (args.frame_skip + 1) != 0:
                frame_count += 1
                continue

            # 定期gc
            if frame_count % 200 == 0: 
                gc.collect()

            try:
                current_time = time.time()
                
                # 推理 - 直接使用detect方法，不需要调用_postprocess
                detections = detector.detect(frame)
                
                # 过滤检测结果，只保留目标类别
                if len(detections) > 0:
                    # detections已经是后处理后的结果，格式为[[x1,y1,x2,y2,conf,class_id], ...]
                    valid_detections = []
                    for det in detections:
                        if len(det) >= 6:  # 确保有足够的元素
                            class_id = int(det[5])
                            # 只保留person(0), car(2), bus(5), truck(7)
                            if class_id in [0, 2, 5, 7]:
                                valid_detections.append(det)
                    detections = np.array(valid_detections, dtype=np.float32)
                else:
                    detections = np.empty((0, 6), dtype=np.float32)
                
                # 跟踪
                tracked_objects = tracker.update(detections, frame) if len(detections) > 0 else []
                
                # 使用StayDetector进行停留检测
                stay_detector.update(tracked_objects, current_time, frame)
                staying_objects = stay_detector.get_staying_objects()

                # 可视化
                vis = frame.copy()
                
                # 绘制ROI框
                for name, points in roi_manager.rois.items():
                    cv2.rectangle(vis, points[0], points[1], (0, 255, 0), 2)
                    cv2.putText(vis, name, (points[0][0], points[0][1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 绘制检测框
                for obj in tracked_objects:
                    if len(obj) < 5:
                        continue
                    x1, y1, x2, y2, tid = obj[:5]
                    
                    # 根据是否停留设置颜色
                    if tid in staying_objects:
                        color = (0, 0, 255)  # 红色 - 停留
                        status = "STAYING"
                    else:
                        color = (0, 255, 0)   # 绿色 - 正常
                        status = "MOVING"
                    
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # 显示ID和状态
                    label = f'ID:{int(tid)} {status}'
                    cv2.putText(vis, label, (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 显示性能信息
                current_time = time.time()
                fps = 1.0 / (current_time - last_frame_time + 1e-7)
                last_frame_time = current_time
                
                info = f'FPS:{fps:.1f}  Track:{len(tracked_objects)}  Stay:{len(staying_objects)}'
                cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # 显示停留阈值信息
                threshold_info = f'Stay Threshold: {args.stay_threshold}s  Move Threshold: {args.movement_threshold}px'
                cv2.putText(vis, threshold_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # 保存/显示
                if writer:
                    writer.write(vis)
                cv2.imshow('Jetson Zero-PyTorch Demo', vis)
                
                # 性能日志
                if current_time - last_log_time > 5:
                    print(f"📊 帧: {frame_count}  FPS: {fps:.1f}  跟踪: {len(tracked_objects)}  停留: {len(staying_objects)}")
                    last_log_time = current_time

                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # 重置状态
                    stay_detector.reset()
                    print("🔄 停留检测状态已重置")
                elif key == ord('c'):
                    gc.collect()
                    print("🧹 强制垃圾回收")

            except Exception as e:
                print(f"❌ 处理帧 {frame_count} 时出错: {e}")
                continue

            frame_count += 1

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_resources(cap, detector, tracker, stay_detector)
        print("🛑 程序结束（零PyTorch）")


if __name__ == "__main__":
    main()
