#!/usr/bin/env python3
"""
多模块检测系统主入口
"""

import argparse
import os
import sys
import atexit
import signal

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_module_system.system_controller import MultiModuleSystemController

def signal_handler(sig, frame):
    """处理退出信号的统一入口"""
    print(f"\n🛑 收到退出信号 {sig}，正在退出...")
    # 注意：对于autoinit方式，不需要手动清理CUDA上下文
    sys.exit(0)

def run_system_controller(args):
    """运行系统控制器的核心函数"""
    # 创建系统控制器
    controller = MultiModuleSystemController(config_path=args.config)

    # 命令行参数覆盖配置
    if args.rtsp1:
        # 设置RTSP源
        controller.config['rtsp_sources'] = [args.rtsp1]
        if args.rtsp2:
            controller.config['rtsp_sources'].append(args.rtsp2)
        elif args.video2:
            controller.config['rtsp_sources'].append(args.video2)
    elif args.video1:
        # 使用文件/摄像头源
        if args.video2:
            controller.config['video_sources'] = [args.video1, args.video2]
        else:
            controller.config['video_sources'] = [args.video1, controller.config.get('video_source')]
    if args.no_display:
        controller.config['fullscreen'] = False
    
    # 运行系统
    controller.run()

def main():
    parser = argparse.ArgumentParser(description='多模块检测系统 - 支持RTSP流')
    parser.add_argument('--config', type=str, default='configs/system_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--video1', type=str, 
                       default="data/test_videos/safe_gesture/gf1_new.mp4",
                       help='第一个视频文件路径或摄像头ID')
    parser.add_argument('--video2', type=str, 
                       default="data/test_videos/trash_in_area/1.mp4",
                       help='第二个视频文件路径或摄像头ID')
    parser.add_argument('--rtsp1', type=str,
                       help='第一个RTSP流URL（海康摄像头等）')
    parser.add_argument('--rtsp2', type=str,
                       help='第二个RTSP流URL（海康摄像头等）')
    parser.add_argument('--no-display', action='store_true',
                       help='无头模式运行（不显示窗口）')
    parser.add_argument('--test-rtsp', type=str,
                       help='测试RTSP连接（不运行完整系统）')
    
    args = parser.parse_args()
    
    # 测试RTSP连接模式
    if args.test_rtsp:
        print(f"🔧 测试RTSP连接: {args.test_rtsp}")
        from tests.rtsp_test import test_rtsp_connection
        test_rtsp_connection(args.test_rtsp)
        return

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        run_system_controller(args)
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    except Exception as e:
        print(f"\n❌ 系统运行异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 主程序逻辑结束")

if __name__ == "__main__":
    main()