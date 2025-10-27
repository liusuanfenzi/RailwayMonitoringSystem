# #!/usr/bin/env python3
# """
# 简化版GMM算法测试
# """

# import cv2
# import numpy as np
# import sys
# import os
# from pathlib import Path

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from models.background_subtractors.gmm_model import GMMBackgroundSubtractor

# def test_algorithm_initialization():
#     """测试算法初始化"""
#     print("🔧 测试算法初始化...")
    
#     try:
#         mog2 = GMMBackgroundSubtractor('MOG2')
#         knn = GMMBackgroundSubtractor('KNN')
#         print("✅ MOG2和KNN初始化成功")
#         return True
#     except Exception as e:
#         print(f"❌ 初始化失败: {e}")
#         return False

# def test_roi_setup():
#     """测试ROI设置"""
#     print("🎯 测试ROI设置...")
    
#     try:
#         gmm = GMMBackgroundSubtractor('MOG2')
#         gmm.setup_track_roi([(100, 100), (500, 400)])
#         print("✅ ROI设置成功")
#         return True
#     except Exception as e:
#         print(f"❌ ROI设置失败: {e}")
#         return False

# def test_background_subtraction():
#     """测试背景减除功能"""
#     print("🎨 测试背景减除...")
    
#     try:
#         gmm = GMMBackgroundSubtractor('MOG2', history=50, var_threshold=10)
        
#         # 创建测试帧
#         background = np.full((200, 300, 3), 100, dtype=np.uint8)
#         foreground = background.copy()
#         cv2.rectangle(foreground, (50, 50), (150, 150), (255, 255, 255), -1)
        
#         # 训练背景
#         for _ in range(5):
#             gmm.apply(background, learning_rate=0.1)
        
#         # 检测前景
#         fg_mask = gmm.apply(foreground)
#         foreground_pixels = np.sum(fg_mask > 0)
        
#         print(f"检测到前景像素: {foreground_pixels}")
        
#         if foreground_pixels > 100:
#             print("✅ 背景减除功能正常")
#             return True
#         else:
#             print("❌ 背景减除功能异常")
#             return False
#     except Exception as e:
#         print(f"❌ 背景减除测试失败: {e}")
#         return False

# def test_roi_analysis():
#     """测试ROI分析"""
#     print("📊 测试ROI分析...")
    
#     try:
#         gmm = GMMBackgroundSubtractor('MOG2')
#         gmm.setup_track_roi([(100, 100), (500, 400)])
        
#         frame = np.full((600, 800, 3), 100, dtype=np.uint8)
#         result = gmm.apply_with_roi_analysis(frame)
        
#         if 'track_region' in result:
#             print("✅ ROI分析功能正常")
#             return True
#         else:
#             print("❌ ROI分析功能异常")
#             return False
#     except Exception as e:
#         print(f"❌ ROI分析测试失败: {e}")
#         return False

# def test_video_processing():
#     """测试视频处理"""
#     print("🎬 测试视频处理...")
    
#     video_path = "data/test_videos/train_enter_station.mp4"
    
#     if not Path(video_path).exists():
#         print(f"❌ 测试视频不存在: {video_path}")
#         return False
    
#     try:
#         gmm = GMMBackgroundSubtractor('MOG2', history=200, var_threshold=10)
        
#         # 设置ROI（根据实际视频调整）
#         gmm.setup_track_roi([(300, 200), (800, 900)])
        
#         # 处理视频
#         stats = gmm.process_video(video_path, max_frames=50)
        
#         print(f"📊 处理帧数: {stats['total_frames']}")
#         print(f"📊 平均前景比例: {stats['avg_foreground_ratio']:.4f}")
        
#         if stats['total_frames'] > 0:
#             print("✅ 视频处理功能正常")
#             return True
#         else:
#             print("❌ 视频处理功能异常")
#             return False
#     except Exception as e:
#         print(f"❌ 视频处理测试失败: {e}")
#         return False

# def test_algorithm_comparison():
#     """测试算法对比"""
#     print("⚖️ 测试算法对比...")
    
#     video_path = "data/test_videos/train_enter_station.mp4"
    
#     if not Path(video_path).exists():
#         print(f"❌ 测试视频不存在: {video_path}")
#         return False
    
#     try:
#         algorithms = [
#             ('MOG2', {'history': 200, 'var_threshold': 10}),
#             ('KNN', {'history': 200, 'dist2_threshold': 400}),
#         ]
        
#         for algo_name, params in algorithms:
#             print(f"\n🔍 测试{algo_name}算法...")
#             gmm = GMMBackgroundSubtractor(algo_name, **params)
#             stats = gmm.process_video(video_path, max_frames=30)
#             print(f"   平均前景比例: {stats['avg_foreground_ratio']:.4f}")
        
#         print("✅ 算法对比测试完成")
#         return True
#     except Exception as e:
#         print(f"❌ 算法对比测试失败: {e}")
#         return False

# def run_all_tests():
#     """运行所有测试"""
#     print("=" * 50)
#     print("🧪 GMM算法简化测试")
#     print("=" * 50)
    
#     tests = [
#         ("算法初始化", test_algorithm_initialization),
#         ("ROI设置", test_roi_setup),
#         ("背景减除", test_background_subtraction),
#         ("ROI分析", test_roi_analysis),
#         ("可视化功能", test_visualization),
#         # ("视频处理", test_video_processing),
#         ("算法对比", test_algorithm_comparison),
#     ]
    
#     passed = 0
#     total = len(tests)
    
#     for test_name, test_func in tests:
#         print(f"\n🔍 {test_name}")
#         print("-" * 30)
#         try:
#             if test_func():
#                 passed += 1
#                 print("✅ 测试通过")
#             else:
#                 print("❌ 测试失败")
#         except Exception as e:
#             print(f"❌ 测试异常: {e}")
    
#     print("=" * 50)
#     print(f"📊 测试结果: {passed}/{total} 通过")
    
#     if passed == total:
#         print("🎉 所有测试通过！")
#     else:
#         print("⚠️  部分测试失败")
    
#     print("=" * 50)
#     return passed == total

# def test_visualization():
#     """测试可视化功能"""
#     print("🎨 测试可视化功能...")
    
#     try:
#         gmm = GMMBackgroundSubtractor('MOG2')
#         gmm.setup_track_roi([(100, 100), (500, 400)])
        
#         # 创建测试帧
#         frame = np.full((600, 800, 3), 100, dtype=np.uint8)
#         cv2.rectangle(frame, (200, 200), (300, 300), (255, 255, 255), -1)
        
#         # 应用背景减除
#         result = gmm.apply_with_roi_analysis(frame)
        
#         # 测试双图可视化
#         comparison_2 = gmm.visualize_comparison(frame, result)
        
#         # 测试三图可视化
#         comparison_3 = gmm.visualize_roi_comparison(frame, result)
        
#         print(f"双图模式形状: {comparison_2.shape}")
#         print(f"三图模式形状: {comparison_3.shape}")
        
#         # 保存测试图像
#         output_dir = Path("outputs/tests")
#         output_dir.mkdir(parents=True, exist_ok=True)
        
#         cv2.imwrite(str(output_dir / "visualization_2panel.jpg"), comparison_2)
#         cv2.imwrite(str(output_dir / "visualization_3panel.jpg"), comparison_3)
        
#         print("✅ 可视化功能正常")
#         return True
#     except Exception as e:
#         print(f"❌ 可视化测试失败: {e}")
#         return False

# if __name__ == "__main__":
#     run_all_tests()


#!/usr/bin/env python3
"""
简化版GMM算法测试
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.background_subtractors.gmm_model_old1 import GMMBackgroundSubtractor

def test_algorithm_initialization():
    """测试算法初始化"""
    print("🔧 测试算法初始化...")
    
    try:
        mog2 = GMMBackgroundSubtractor('MOG2')
        knn = GMMBackgroundSubtractor('KNN')
        print("✅ MOG2和KNN初始化成功")
        return True
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False

def test_roi_setup():
    """测试ROI设置"""
    print("🎯 测试ROI设置...")
    
    try:
        gmm = GMMBackgroundSubtractor('MOG2')
        gmm.setup_track_roi([(100, 100), (500, 400)])
        print("✅ ROI设置成功")
        return True
    except Exception as e:
        print(f"❌ ROI设置失败: {e}")
        return False

def test_background_subtraction():
    """测试背景减除功能"""
    print("🎨 测试背景减除...")
    
    try:
        gmm = GMMBackgroundSubtractor('MOG2', history=50, var_threshold=10)
        
        # 创建测试帧
        background = np.full((200, 300, 3), 100, dtype=np.uint8)
        foreground = background.copy()
        cv2.rectangle(foreground, (50, 50), (150, 150), (255, 255, 255), -1)
        
        # 训练背景
        for _ in range(5):
            gmm.apply(background, learning_rate=0.1)
        
        # 检测前景
        fg_mask = gmm.apply(foreground)
        foreground_pixels = np.sum(fg_mask > 0)
        
        print(f"检测到前景像素: {foreground_pixels}")
        
        if foreground_pixels > 100:
            print("✅ 背景减除功能正常")
            return True
        else:
            print("❌ 背景减除功能异常")
            return False
    except Exception as e:
        print(f"❌ 背景减除测试失败: {e}")
        return False

def test_roi_analysis():
    """测试ROI分析"""
    print("📊 测试ROI分析...")
    
    try:
        gmm = GMMBackgroundSubtractor('MOG2')
        gmm.setup_track_roi([(100, 100), (500, 400)])
        
        frame = np.full((600, 800, 3), 100, dtype=np.uint8)
        result = gmm.apply_with_roi_analysis(frame)
        
        if 'track_region' in result:
            print("✅ ROI分析功能正常")
            return True
        else:
            print("❌ ROI分析功能异常")
            return False
    except Exception as e:
        print(f"❌ ROI分析测试失败: {e}")
        return False

def test_visualization():
    """测试可视化功能"""
    print("🎨 测试可视化功能...")
    
    try:
        gmm = GMMBackgroundSubtractor('MOG2')
        gmm.setup_track_roi([(100, 100), (500, 400)])
        
        # 创建测试帧
        frame = np.full((600, 800, 3), 100, dtype=np.uint8)
        cv2.rectangle(frame, (200, 200), (300, 300), (255, 255, 255), -1)
        
        # 应用背景减除
        result = gmm.apply_with_roi_analysis(frame)
        
        # 测试可视化（短暂显示）
        gmm.visualize_comparison(frame, result)
        cv2.waitKey(100)  # 短暂显示100ms
        cv2.destroyAllWindows()
        
        print("✅ 可视化功能正常，三个窗口创建成功")
        return True
    except Exception as e:
        print(f"❌ 可视化测试失败: {e}")
        return False

def test_video_processing():
    """测试视频处理"""
    print("🎬 测试视频处理...")
    
    video_path = "data/test_videos/train_enter_station.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ 测试视频不存在: {video_path}")
        return False
    
    try:
        gmm = GMMBackgroundSubtractor('MOG2', history=200, var_threshold=10)
        
        # 设置ROI（根据实际视频调整）
        gmm.setup_track_roi([(300, 200), (800, 900)])
        
        # 处理视频（不显示可视化）
        stats = gmm.process_video(video_path, max_frames=50, show_visualization=False)
        
        print(f"📊 处理帧数: {stats['total_frames']}")
        print(f"📊 平均前景比例: {stats['avg_foreground_ratio']:.4f}")
        
        if stats['total_frames'] > 0:
            print("✅ 视频处理功能正常")
            return True
        else:
            print("❌ 视频处理功能异常")
            return False
    except Exception as e:
        print(f"❌ 视频处理测试失败: {e}")
        return False

def test_algorithm_comparison():
    """测试算法对比"""
    print("⚖️ 测试算法对比...")
    
    video_path = "data/test_videos/train_enter_station.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ 测试视频不存在: {video_path}")
        return False
    
    try:
        algorithms = [
            ('MOG2', {'history': 200, 'var_threshold': 10}),
            ('KNN', {'history': 200, 'dist2_threshold': 400}),
        ]
        
        for algo_name, params in algorithms:
            print(f"\n🔍 测试{algo_name}算法...")
            gmm = GMMBackgroundSubtractor(algo_name, **params)
            stats = gmm.process_video(video_path, max_frames=30, show_visualization=False)
            print(f"   平均前景比例: {stats['avg_foreground_ratio']:.4f}")
        
        print("✅ 算法对比测试完成")
        return True
    except Exception as e:
        print(f"❌ 算法对比测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("🧪 GMM算法简化测试")
    print("=" * 50)
    
    tests = [
        ("算法初始化", test_algorithm_initialization),
        ("ROI设置", test_roi_setup),
        ("背景减除", test_background_subtraction),
        ("ROI分析", test_roi_analysis),
        ("可视化功能", test_visualization),
        ("视频处理", test_video_processing),
        ("算法对比", test_algorithm_comparison),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
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
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败")
    
    print("=" * 50)
    return passed == total

if __name__ == "__main__":
    run_all_tests()