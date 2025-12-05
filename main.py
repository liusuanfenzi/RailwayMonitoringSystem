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

# 注意：我们移除了对cuda_context_manager的导入
# from multi_module_system.cuda_context_manager import cuda_context_aware, context_manager
from multi_module_system.system_controller import MultiModuleSystemController

def signal_handler(sig, frame):
    """处理退出信号的统一入口"""
    print(f"\n🛑 收到退出信号 {sig}，正在退出...")
    # 注意：对于autoinit方式，不需要手动清理CUDA上下文
    sys.exit(0)

# 移除cuda_context_aware装饰器
def run_system_controller(args):
    """运行系统控制器的核心函数"""
    # 创建系统控制器
    controller = MultiModuleSystemController(config_path=args.config)

    # 覆盖配置中的视频源
    if args.video:
        controller.config['video_source'] = args.video

    # 如果提供了两个视频源，使用 video_sources 配置
    if args.video1 and args.video2:
        controller.config['video_sources'] = [args.video1, args.video2]
    elif args.video1:
        controller.config['video_sources'] = [args.video1, controller.config.get('video_source')]
    elif args.video2:
        controller.config['video_sources'] = [controller.config.get('video_source'), args.video2]
    
    if args.no_display:
        controller.config['fullscreen'] = False
        # 这里可以修改不启动显示线程

    # 运行系统
    controller.run()

def main():
    parser = argparse.ArgumentParser(description='多模块检测系统')
    parser.add_argument('--config', type=str, default='configs/system_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--video', type=str, 
                       default="data/test_videos/safe_gesture/gf1_new.mp4",
                       help='视频文件路径或摄像头ID')
    parser.add_argument('--video1', type=str, default="data/test_videos/safe_gesture/gf1_new.mp4",
                       help='第一个视频文件路径或摄像头ID（可为RTSP URL）')
    parser.add_argument('--video2', type=str, default="data/test_videos/trash_in_area/1.mp4",
                       help='第二个视频文件路径或摄像头ID（可为RTSP URL）')
    parser.add_argument('--no-display', action='store_true',
                       help='无头模式运行（不显示窗口）')
    
    args = parser.parse_args()

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill命令

    try:
        # 直接调用函数，不再使用装饰器
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