#!/usr/bin/env python3
"""
简化版列车进出站检测演示程序
基于单个ROI区域和时域判断的进站检测
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='简化版列车进出站检测演示')
    parser.add_argument('--video', type=str, default='data/test_videos/train_enter_station.mp4', 
                       help='输入视频文件路径')
    parser.add_argument('--roi', type=str, default='200,200,600,700',
                       help='ROI区域坐标 x1,y1,x2,y2')
    parser.add_argument('--spatial_threshold', type=float, default=0.3,
                       help='空域检测阈值')
    parser.add_argument('--temporal_frames', type=int, default=100,
                       help='时域判断帧数')
    parser.add_argument('--temporal_threshold', type=int, default=65,
                       help='时域判断阈值')
    parser.add_argument('--max_frames', type=int, default=1500,
                       help='最大处理帧数')
    
    args = parser.parse_args()
    
    # 解析ROI坐标
    try:
        roi_coords = [int(x) for x in args.roi.split(',')]
        if len(roi_coords) != 4:
            raise ValueError("ROI坐标必须是4个数字")
        roi_points = [(roi_coords[0], roi_coords[1]), (roi_coords[2], roi_coords[3])]
    except Exception as e:
        print(f"❌ ROI坐标解析错误: {e}")
        return
    
    # 检查视频文件
    if not Path(args.video).exists():
        print(f"❌ 视频文件不存在: {args.video}")
        return
    
    print("🚀 启动简化版列车进出站检测系统")
    print(f"📹 视频文件: {args.video}")
    print(f"🎯 ROI区域: {roi_points}")
    print(f"⚙️ 空域阈值: {args.spatial_threshold}")
    print(f"⏱️ 时域帧数: {args.temporal_frames}")
    print(f"📊 时域阈值: {args.temporal_threshold}")
    
    try:
        # 初始化GMM背景减除器
        bg_subtractor = GMMBackgroundSubtractor(algorithm='MOG2', 
                                                history=200, 
                                                varThreshold=16,
                                                preprocess_mode='enhance_dark',
                                                noise_reduction='light',
                                                detect_shadows=False)
        
        # 设置单个ROI区域
        bg_subtractor.setup_single_roi(roi_points, 'train_detection_roi')
        
        # 初始化列车检测器
        detector = TrainStationDetector(
            spatial_threshold=args.spatial_threshold,
            temporal_frames=args.temporal_frames,
            temporal_threshold=args.temporal_threshold
        )
        
        # 处理视频
        stats = detector.process_video_with_detection(
            video_path=args.video,
            bg_subtractor=bg_subtractor,
            max_frames=args.max_frames,
            show_visualization=True
        )
        
        # 输出统计结果
        print("\n📈 检测统计结果:")
        print(f"   总处理帧数: {stats['total_frames']}")
        print(f"   进站事件数: {stats['entry_events']}")
        print(f"   最终状态: {stats['final_state']}")
        
        # 显示事件历史
        if stats['event_history']:
            print(f"\n📋 事件历史:")
            for i, event in enumerate(stats['event_history']):
                print(f"   事件{i+1}: 帧{event['frame_index']} - {event['event_type']} "
                      f"(置信度: {event['confidence']:.3f})")
        
    except Exception as e:
        print(f"❌ 程序执行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 导入所需的类（确保这些类在同一个文件中或已正确导入）
    from models.background_subtractors.gmm_model import GMMBackgroundSubtractor
    from models.detectors.train_station_detector import TrainStationDetector
    
    main()