#!/usr/bin/env python3
"""
视频基础功能测试 - MP4文件读取测试
"""

from utils.video.video_utils import VideoReader, ROIManager
import cv2
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_mp4_video_stream():
    """测试MP4视频文件流读取"""
    print("📹 测试MP4视频文件流读取...")

    # 使用现有的测试视频（确保文件存在）
    video_path = "data/test_videos/rubbish_detection.mp4"

    # 检查文件是否存在
    if not Path(video_path).exists():
        print(f"❌ 测试视频不存在: {video_path}")
        print("💡 请先确保测试视频文件存在")
        return False

    try:
        # 初始化视频读取器
        reader = VideoReader(video_path)
        _, _, _, frame_count = reader.get_properties()

        # 读取前5帧进行测试
        print("🔄 读取视频帧...")
        frames_read = 0
        for i in range(min(5, frame_count)):
            frame = reader.read_frame()
            if frame is not None:
                frames_read += 1
                cv2.imshow(f"Test Frame{frames_read}", frame)
                print(f"✅ 第{i+1}帧读取成功: {frame.shape}")
            else:
                print(f"❌ 第{i+1}帧读取失败")
                break
        cv2.waitKey(5000)  # 显示1000ms
        cv2.destroyAllWindows()
        reader.release()
        print(f"🎉 MP4视频测试完成，成功读取 {frames_read} 帧")
        return True

    except Exception as e:
        print(f"❌ MP4视频测试失败: {e}")
        return False


def test_video_properties():
    """测试视频属性获取功能"""
    print("📋 测试视频属性获取...")

    video_path = "data/test_videos/rubbish_detection.mp4"

    if not Path(video_path).exists():
        print(f"❌ 测试视频不存在: {video_path}")
        return False

    try:
        reader = VideoReader(video_path)

        # 获取详细属性
        width, height, fps, frame_count = reader.get_properties()

        print("📊 视频详细属性:")
        print(f"   - 宽度: {width} 像素")
        print(f"   - 高度: {height} 像素")
        print(f"   - 宽高比: {width/height:.2f}")
        print(f"   - 帧率: {fps} FPS")
        print(f"   - 总帧数: {frame_count}")

        if fps > 0 and frame_count > 0:
            duration = frame_count / fps
            print(f"   - 视频时长: {duration:.2f} 秒")

        reader.release()
        print("✅ 视频属性测试完成")
        return True

    except Exception as e:
        print(f"❌ 视频属性测试失败: {e}")
        return False


def test_roi_functionality():
    """测试ROI功能"""
    print("🎯 测试ROI功能...")

    try:
        frames_read = 0
        # 初始化ROI管理器
        roi_manager = ROIManager()

        # 添加不同类型的ROI
        roi_manager.add_roi("矩形ROI", [(100, 100), (300, 300)])  # 矩形
        roi_manager.add_roi("小区域", [(50, 50), (150, 150)])     # 小矩形

        # 使用现有的测试视频帧的前五帧作为图像
        video_path = "data/test_videos/rubbish_detection.mp4"
        if not Path(video_path).exists():
            print(f"❌ 测试视频不存在: {video_path}")
            return False
        reader = VideoReader(video_path)
        _, _, _, frame_count = reader.get_properties()
        for i in range(min(5, frame_count)):
            test_image = reader.read_frame()
            if test_image is not None:
                frames_read += 1
                # 测试ROI绘制
                image_with_roi = roi_manager.draw_rois(test_image)
                print("✅ ROI绘制成功")
                # 测试ROI裁剪
                cropped_roi = roi_manager.crop_roi(test_image, "矩形ROI")
                print(f"✅ ROI裁剪成功: {test_image.shape} -> {cropped_roi.shape}")
                # 保存测试结果
                output_dir = Path("outputs/tests")
                output_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_dir / "roi_test.jpg"), image_with_roi)
                cv2.imwrite(str(output_dir / "roi_cropped.jpg"), cropped_roi)
                print(f"💾 测试结果已保存到: {output_dir}")
                print("✅ ROI功能测试完成")
            else:
                print(f"❌ 第{i+1}帧读取失败")
                break

        reader.release()
        print(f"🎉 ROI功能测试完成，成功读取 {frames_read} 帧")
        return True

    except Exception as e:
        print(f"❌ ROI功能测试失败: {e}")
        return False


def test_nonexistent_video():
    """测试不存在的视频文件处理"""
    print("❓ 测试不存在的视频文件处理...")

    try:
        reader = VideoReader("data/test_videos/nonexistent_video.mp4")
        print("❌ 应该抛出异常但未抛出")
        reader.release()
        return False
    except ValueError as e:
        print(f"✅ 正确处理不存在的文件: {e}")
        return True
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("🧪 开始视频基础功能测试")
    print("=" * 50)

    tests = [
        ("MP4视频流读取", test_mp4_video_stream),
        ("视频属性获取", test_video_properties),
        ("ROI功能测试", test_roi_functionality),
        ("错误文件处理", test_nonexistent_video),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 测试: {test_name}")
        print("-" * 30)
        try:
            if test_func():
                passed += 1
                print("✅ 测试通过")
            else:
                print("❌ 测试失败")
        except Exception as e:
            print(f"❌ 测试异常: {e}")

    print("=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！视频基础功能正常")
    else:
        print("⚠️  部分测试失败，请检查相关问题")

    print("=" * 50)
    return passed == total


if __name__ == "__main__":
    # 运行所有测试
    run_all_tests()
