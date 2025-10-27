import cv2
import numpy as np
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.trackers.multi_object_tracking import MultiObjectTracker

def test_motion_detector():
    """测试运动检测器"""
    print("测试运动检测器...")
    
    # 直接读取视频文件
    video_path = 'data/test_videos/trash_in_area/1.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {width}x{height}, FPS: {fps:.1f}")
    
    # 初始化运动检测器
    from models.trackers.multi_object_tracking import MotionDetector
    detector = MotionDetector(method='mog2')
    
    # 创建输出视频
    output_path = "motion_detection_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 运动检测
        detections = detector.detect(frame)
        
        # 绘制检测结果
        for det in detections:
            x, y, w, h = det
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 显示统计信息
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {len(detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(frame)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"已处理 {frame_count} 帧, 检测到 {len(detections)} 个目标")
    
    cap.release()
    out.release()
    print(f"运动检测测试完成: {output_path}")

def test_sort_tracker():
    """测试SORT跟踪器"""
    print("测试SORT跟踪器...")
    
    # 直接读取视频文件
    video_path = 'data/test_videos/trash_in_area/1.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {width}x{height}, FPS: {fps:.1f}")
    
    # 初始化跟踪器
    tracker = MultiObjectTracker(tracker_type='sort', use_motion_detector=True)
    
    # 创建输出视频
    output_path = "sort_tracking_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧
        result = tracker.process_frame(frame)
        
        # 显示统计信息
        cv2.putText(result['frame'], f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result['frame'], f"Detections: {result['num_detections']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result['frame'], f"Tracks: {result['num_tracks']}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(result['frame'])
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"已处理 {frame_count} 帧 - 检测: {result['num_detections']}, 跟踪: {result['num_tracks']}")
    
    cap.release()
    out.release()
    print(f"SORT跟踪测试完成: {output_path}")

def test_deepsort_tracker():
    """测试DeepSORT跟踪器"""
    print("测试DeepSORT跟踪器...")
    
    # 直接读取视频文件
    video_path = 'data/test_videos/trash_in_area/1.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {width}x{height}, FPS: {fps:.1f}")
    
    # 初始化跟踪器
    tracker = MultiObjectTracker(tracker_type='deepsort', use_motion_detector=True)
    
    # 创建输出视频
    output_path = "deepsort_tracking_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧
        result = tracker.process_frame(frame)
        
        # 显示统计信息
        cv2.putText(result['frame'], f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result['frame'], f"Detections: {result['num_detections']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result['frame'], f"Tracks: {result['num_tracks']}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(result['frame'])
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"已处理 {frame_count} 帧 - 检测: {result['num_detections']}, 跟踪: {result['num_tracks']}")
    
    cap.release()
    out.release()
    print(f"DeepSORT跟踪测试完成: {output_path}")

def main():
    """主测试函数"""
    print("开始多目标跟踪测试...")
    
    # 测试运动检测器
    test_motion_detector()
    
    # 测试SORT跟踪器
    test_sort_tracker()
    
    # 测试DeepSORT跟踪器
    test_deepsort_tracker()
    
    print("\n所有测试完成!")
    print("生成的输出文件:")
    print("  motion_detection_result.mp4 - 运动检测结果")
    print("  sort_tracking_result.mp4 - SORT跟踪结果") 
    print("  deepsort_tracking_result.mp4 - DeepSORT跟踪结果")

if __name__ == "__main__":
    main()