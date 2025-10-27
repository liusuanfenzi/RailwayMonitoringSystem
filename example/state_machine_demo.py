#!/usr/bin/env python3
"""
状态机列车检测演示 - 修改版
"""

from models.background_subtractors.gmm_model_old1 import GMMBackgroundSubtractor
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():

    print("🚂 状态机列车检测演示 - 基于检测信号版")

    video_path = "data/test_videos/train_enter_station.mp4"

    if not Path(video_path).exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return

    # 初始化检测器 - 使用新的参数
    detector = GMMBackgroundSubtractor(
        'MOG2',
        history=100,
        varThreshold=8,
        min_stay_duration=5.0,        # 最小停留时间
        min_entering_duration=3.0,    # 新增：最小进站时间
        cooldown_duration=1.5,        # 事件冷却
        entering_timeout=15.0,        # 进站超时
        exiting_timeout=10.0         # 出站超时
    )

    # 设置进出站ROI区域
    entry_roi = [(400, 0), (700, 300)]   # 进站检测区域
    exit_roi = [(100, 450), (520, 900)]  # 出站检测区域

    detector.setup_track_rois(entry_roi, exit_roi)

    print("🎯 状态机参数:")
    print(f"   - 最小停留时间: {detector.state_manager.min_stay_duration}秒")
    print(f"   - 最小进站时间: {detector.state_manager.min_entering_duration}秒")
    print(f"   - 事件冷却时间: {detector.state_manager.cooldown_duration}秒")
    print(f"   - 进站超时: {detector.state_manager.entering_timeout}秒")
    print(f"   - 出站超时: {detector.state_manager.exiting_timeout}秒")

    # 使用状态机处理视频
    stats = detector.process_video_with_state_machine(
        video_path=video_path,
        max_frames=1500,
        show_visualization=True,
        entry_threshold=0.3,
        exit_threshold=0.2
    )

    # 输出事件历史
    print("\n📋 事件历史记录:")
    for i, event in enumerate(stats['event_history']):
        print(f"   {i+1}. {event}")

    # 输出统计信息
    print(f"\n📊 处理统计:")
    print(f"   - 总帧数: {stats['total_frames']}")
    print(f"   - 进站事件: {stats['entry_events']}次")
    print(f"   - 出站事件: {stats['exit_events']}次")
    print(f"   - 最终状态: {stats['final_state']}")
    print(f"   - 处理时间: {stats['processing_time']:.2f}秒")


if __name__ == "__main__":
    main()
