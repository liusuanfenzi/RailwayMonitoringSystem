# jetson_train_detection_visual_demo.py
#!/usr/bin/env python3
"""
Jetson优化的列车进出站检测演示程序 - 可视化版本
实时显示检测结果，适合在Jetson本地测试效果
"""

import argparse
from pathlib import Path
import os
from models.detector.train_detector import JetsonTrainStationDetectorVisual
from models.background_subtractor.gmm_model import JetsonGMMBackgroundSubtractor

def setup_display():
    """设置显示环境"""
    # 设置显示（适用于Jetson）
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':0'
    
    # 设置GTK后端（如果需要）
    os.environ['GDK_BACKEND'] = 'x11'

def main():
    """主函数 - 可视化版本"""
    parser = argparse.ArgumentParser(description='Jetson列车进出站检测演示 - 可视化版本')
    parser.add_argument('--video', type=str, default='data/test_videos/train_enter_station.mp4',
                       help='输入视频文件路径 ')
    parser.add_argument('--roi', type=str, default='180,200,600,700',
                       help='ROI区域坐标 x1,y1,x2,y2 ')
    parser.add_argument('--preprocess_mode', type=str, default='enhance_dark', 
                       choices=['basic', 'enhance_dark'],
                       help='预处理模式: basic(性能优先) 或 enhance_dark(效果优先) ')
    parser.add_argument('--spatial_threshold', type=float, default=0.3,
                       help='空域检测阈值 (默认: 0.3)')
    parser.add_argument('--temporal_frames', type=int, default=50,
                       help='时域判断帧数 (默认: 50)')
    parser.add_argument('--temporal_threshold', type=int, default=25,
                       help='时域判断阈值 ')
    parser.add_argument('--max_frames', type=int, default=1500,
                       help='最大处理帧数 ')
    parser.add_argument('--print_interval', type=int, default=10,
                       help='置信度打印间隔帧数 (默认: 10)')
    parser.add_argument('--no_visualization', action='store_true',
                       help='禁用可视化显示')
    
    args = parser.parse_args()
    
    # 设置显示环境
    if not args.no_visualization:
        setup_display()
    
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
        print("💡 请确保视频文件已复制到Jetson，或使用 --video 参数指定正确路径")
        return
    
    print("🚀 启动Jetson列车进出站检测系统 - 可视化版本")
    print(f"📹 视频文件: {args.video}")
    print(f"🎯 ROI区域: {roi_points}")
    print(f"🔧 预处理模式: {args.preprocess_mode}")
    print(f"⚙️ 空域阈值: {args.spatial_threshold}")
    print(f"⏱️ 时域帧数: {args.temporal_frames}")
    print(f"📊 时域阈值: {args.temporal_threshold}")
    print(f"📝 打印间隔: 每 {args.print_interval} 帧")
    print(f"👁️ 可视化: {'启用' if not args.no_visualization else '禁用'}")
    print(f"💾 截图类型: ROI前景掩码 (后处理)")
    print(f"📁 保存位置: outputs/train_detection/")
    print("=" * 60)
    
    try:
        # 初始化Jetson优化的GMM背景减除器
        bg_subtractor = JetsonGMMBackgroundSubtractor(
            algorithm='MOG2', 
            preprocess_mode=args.preprocess_mode,
            history=150,
            varThreshold=16,
            detect_shadows=False
        )
        
        # 设置ROI区域
        bg_subtractor.setup_single_roi(roi_points, 'train_detection_roi')
        
        # 初始化Jetson优化的列车检测器
        detector = JetsonTrainStationDetectorVisual(
            spatial_threshold=args.spatial_threshold,
            temporal_frames=args.temporal_frames,
            temporal_threshold=args.temporal_threshold,
            print_interval=args.print_interval
        )
        
        # 设置可视化模式
        detector.show_visualization = not args.no_visualization
        
        print("🎯 开始检测...")
        print("控制说明:")
        print("  - 按 'q' 键退出程序")
        print("  - 按 's' 键手动保存当前帧截图")
        print("-" * 60)
        
        # 处理视频（可视化版本）
        stats = detector.process_video_with_visualization(
            video_path=args.video,
            bg_subtractor=bg_subtractor,
            max_frames=args.max_frames
        )
        
        # 输出统计结果
        print("\n" + "=" * 60)
        print("📈 检测统计结果:")
        print("=" * 60)
        print(f"   总处理帧数: {stats['total_frames']}")
        print(f"   进站事件数: {stats['entry_events']}")
        print(f"   平均FPS: {stats['avg_fps']:.1f}")
        print(f"   保存ROI掩码截图: {stats['saved_snapshots']} 张")
        print(f"   最终状态: {stats['final_state']}")
        
        # 显示事件历史
        if stats['event_history']:
            print(f"\n📋 事件历史:")
            for i, event in enumerate(stats['event_history']):
                print(f"   事件{i+1}: 帧{event['frame_index']} - {event['event_type']} "
                      f"(置信度: {event['confidence']:.3f})")
        
        # 显示性能统计
        bg_stats = bg_subtractor.get_performance_stats()
        detector_stats = detector.get_detection_status()
        print(f"\n🎯 性能统计:")
        print(f"   预处理模式: {args.preprocess_mode}")
        print(f"   背景减除平均耗时: {bg_stats['avg_time']:.1f}ms")
        print(f"   检测器平均FPS: {detector_stats['fps']:.1f}")
        
        # 输出文件位置
        print(f"\n💾 输出文件位置:")
        print(f"   ROI前景掩码截图保存至: outputs/train_detection/")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断程序")
        # 即使中断也显示统计信息
        try:
            detector_stats = detector.get_detection_status()
            print(f"\n📊 中断时统计:")
            print(f"   已处理帧数: {detector_stats.get('total_frames', 0)}")
            print(f"   进站事件: {detector_stats.get('entry_count', 0)}")
            print(f"   保存ROI掩码截图: {detector_stats.get('saved_snapshots', 0)}")
        except:
            pass
    except Exception as e:
        print(f"❌ 程序执行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
