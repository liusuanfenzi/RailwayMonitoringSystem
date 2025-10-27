#!/usr/bin/env python3
"""
列车检测器测试脚本
"""

from utils.video.video_utils import VideoReader
from models.detectors.train_detector import TrainEntryExitDetector
import cv2
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_train_detector_initialization():
    """测试列车检测器初始化"""
    print("🔧 测试列车检测器初始化...")

    try:
        detector = TrainEntryExitDetector()
        print("✅ 列车检测器初始化成功")
        return True
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False


def test_roi_setup():
    """测试ROI设置"""
    print("🎯 测试ROI设置...")

    try:
        detector = TrainEntryExitDetector()

        # 设置轨道ROI
        track_roi = [(100, 100), (900, 800)]  # 假设的轨道区域
        detector.setup_track_roi(track_roi)

        # 设置多个ROI
        roi_config = {
            'track_region': [(100, 100), (900, 800)],
            'platform_area': [(50, 800), (950, 1000)],
            'entry_zone': [(800, 200), (1000, 600)]
        }
        detector.setup_roi_regions(roi_config)

        print("✅ ROI设置成功")
        return True
    except Exception as e:
        print(f"❌ ROI设置失败: {e}")
        return False


def test_train_detection_on_video():
    """在视频上测试列车检测"""
    print("🎬 在视频上测试列车检测...")

    video_path = "data/test_videos/train_enter_station.mp4"

    if not Path(video_path).exists():
        print(f"❌ 测试视频不存在: {video_path}")
        return False

    try:
        # 初始化检测器
        detector = TrainEntryExitDetector()

        # 设置轨道ROI（根据你的视频调整这些坐标）
        # 这些坐标需要根据实际视频中的轨道位置调整
        track_roi = [(300, 200), (800, 900)]  # 示例坐标
        detector.setup_track_roi(track_roi)

        # 初始化视频读取器
        reader = VideoReader(video_path)

        # 处理视频
        enter_events = 0
        exit_events = 0
        frame_count = 0

        print("🔄 开始处理视频...")

        while True:
            frame = reader.read_frame()
            if frame is None:
                break

            # 处理帧
            result = detector.process_frame(frame)

            # 可视化结果
            vis_frame = detector.visualize_detection(frame, result)

            # 显示结果
            cv2.imshow('列车进出站检测', vis_frame)

            # 记录事件
            if result.train_entering:
                enter_events += 1
                print(f"🚂 检测到列车进站! 帧: {frame_count}")

            if result.train_exiting:
                exit_events += 1
                print(f"🚂 检测到列车离站! 帧: {frame_count}")

            frame_count += 1

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        reader.release()
        cv2.destroyAllWindows()

        # 输出统计
        stats = detector.get_statistics()
        print(f"\n📊 检测统计:")
        print(f"   总帧数: {stats['total_frames']}")
        print(f"   进站事件: {enter_events}")
        print(f"   离站事件: {exit_events}")
        print(f"   检测比例: {stats['detection_ratio']:.3f}")

        return True

    except Exception as e:
        print(f"❌ 视频检测测试失败: {e}")
        return False


def test_synthetic_train_detection():
    """测试合成数据上的列车检测"""
    print("🎨 测试合成数据上的列车检测...")

    try:
        detector = TrainEntryExitDetector()

        # 设置轨道ROI
        detector.setup_track_roi([(100, 100), (500, 400)])

        # 创建合成帧序列模拟列车进站
        background = np.full((600, 800, 3), 100, dtype=np.uint8)

        # 模拟列车进站过程
        train_detected = False
        enter_detected = False

        for i in range(50):
            frame = background.copy()

            # 模拟列车移动（从右侧进入）
            if i >= 10 and i < 40:
                # 列车在场
                train_x = 700 - i * 15
                cv2.rectangle(frame, (train_x, 150),
                              (train_x + 100, 350), (200, 200, 200), -1)

                if not train_detected:
                    print(f"🚂 合成数据: 列车出现在帧 {i}")
                    train_detected = True

            # 处理帧
            result = detector.process_frame(frame)

            if result.train_entering and not enter_detected:
                print(f"✅ 成功检测到列车进站! 帧 {i}")
                enter_detected = True

        if enter_detected:
            print("✅ 合成数据测试通过")
            return True
        else:
            print("❌ 合成数据测试失败")
            return False

    except Exception as e:
        print(f"❌ 合成数据测试失败: {e}")
        return False


def run_all_train_detector_tests():
    """运行所有列车检测器测试"""
    print("=" * 60)
    print("🧪 列车进出站检测器全面测试")
    print("=" * 60)

    tests = [
        ("检测器初始化", test_train_detector_initialization),
        ("ROI设置测试", test_roi_setup),
        ("合成数据测试", test_synthetic_train_detection),
        ("真实视频测试", test_train_detection_on_video),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
                print("✅ 测试通过")
            else:
                print("❌ 测试失败")
        except Exception as e:
            print(f"❌ 测试异常: {e}")

    print("=" * 60)
    print(f"📊 列车检测器测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有列车检测器测试通过！")
    else:
        print("⚠️  部分测试失败，请检查相关问题")

    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    run_all_train_detector_tests()
