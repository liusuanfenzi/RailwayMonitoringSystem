#!/usr/bin/env python3
"""
铁轨光影变化处理策略测试
"""

import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.background_subtractors.gmm_model_old1 import GMMBackgroundSubtractor

def simulate_light_change_scenario():
    """模拟铁轨光影变化场景"""
    print("🎬 模拟铁轨光影变化场景...")
    
    # 创建基础铁轨场景
    width, height = 800, 600
    frames = []
    
    # 阶段1: 正常光照
    for i in range(30):
        frame = np.full((height, width, 3), 120, dtype=np.uint8)  # 中等亮度
        # 添加铁轨纹理
        cv2.line(frame, (100, 300), (700, 300), (80, 80, 80), 10)  # 铁轨
        cv2.line(frame, (100, 320), (700, 320), (80, 80, 80), 10)  # 铁轨
        frames.append(frame)
    
    # 阶段2: 光影变化（模拟列车进入）
    for i in range(20):
        # 逐渐变亮再变暗，模拟光影变化
        brightness = 120 + 50 * np.sin(i * 0.3)
        frame = np.full((height, width, 3), max(50, min(200, brightness)), dtype=np.uint8)
        cv2.line(frame, (100, 300), (700, 300), (80, 80, 80), 10)
        cv2.line(frame, (100, 320), (700, 320), (80, 80, 80), 10)
        frames.append(frame)
    
    # 阶段3: 列车实体通过
    for i in range(30):
        frame = np.full((height, width, 3), 120, dtype=np.uint8)
        cv2.line(frame, (100, 300), (700, 300), (80, 80, 80), 10)
        cv2.line(frame, (100, 320), (700, 320), (80, 80, 80), 10)
        # 添加列车
        train_x = 200 + i * 15
        cv2.rectangle(frame, (train_x, 250), (train_x + 100, 350), (180, 180, 180), -1)
        frames.append(frame)
    
    return frames

def test_strategy_with_equalization():
    """测试使用直方图均衡化的策略"""
    print("\n🔧 测试策略1: 使用直方图均衡化")
    
    frames = simulate_light_change_scenario()
    
    # 使用直方图均衡化的KNN
    knn = cv2.createBackgroundSubtractorKNN(
        history=200,
        dist2Threshold=400,
        detectShadows=False
    )
    
    foreground_ratios = []
    light_change_detected = []
    train_detected = []
    
    for i, frame in enumerate(frames):
        # 应用直方图均衡化
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        processed_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        # 应用KNN
        fg_mask = knn.apply(processed_frame)
        
        # 计算前景比例
        fg_ratio = np.sum(fg_mask > 0) / fg_mask.size
        foreground_ratios.append(fg_ratio)
        
        # 分析检测情况
        if 30 <= i < 50:  # 光影变化阶段
            light_change_detected.append(fg_ratio > 0.01)
        elif i >= 50:  # 列车通过阶段
            train_detected.append(fg_ratio > 0.05)
    
    # 统计结果
    light_change_detection_rate = np.mean(light_change_detected) if light_change_detected else 0
    train_detection_rate = np.mean(train_detected) if train_detected else 0
    
    print(f"   光影变化检测率: {light_change_detection_rate:.2f}")
    print(f"   列车检测率: {train_detection_rate:.2f}")
    print(f"   平均前景比例: {np.mean(foreground_ratios):.4f}")
    
    return {
        'strategy': '直方图均衡化',
        'light_change_detection': light_change_detection_rate,
        'train_detection': train_detection_rate,
        'avg_foreground': np.mean(foreground_ratios)
    }

def test_strategy_treat_light_as_foreground():
    """测试将光影变化视为前景的策略"""
    print("\n🔧 测试策略2: 光影变化视为前景")
    
    frames = simulate_light_change_scenario()
    
    # 使用更敏感的KNN参数
    knn = cv2.createBackgroundSubtractorKNN(
        history=100,           # 较短历史，更快适应变化
        dist2Threshold=300,    # 较低阈值，更敏感
        detectShadows=False    # 不区分阴影
    )
    
    foreground_ratios = []
    light_change_detected = []
    train_detected = []
    
    for i, frame in enumerate(frames):
        # 直接应用KNN，不进行均衡化
        fg_mask = knn.apply(frame)
        
        # 计算前景比例
        fg_ratio = np.sum(fg_mask > 0) / fg_mask.size
        foreground_ratios.append(fg_ratio)
        
        # 分析检测情况
        if 30 <= i < 50:  # 光影变化阶段
            light_change_detected.append(fg_ratio > 0.01)
        elif i >= 50:  # 列车通过阶段
            train_detected.append(fg_ratio > 0.05)
    
    # 统计结果
    light_change_detection_rate = np.mean(light_change_detected) if light_change_detected else 0
    train_detection_rate = np.mean(train_detected) if train_detected else 0
    
    print(f"   光影变化检测率: {light_change_detection_rate:.2f}")
    print(f"   列车检测率: {train_detection_rate:.2f}")
    print(f"   平均前景比例: {np.mean(foreground_ratios):.4f}")
    
    return {
        'strategy': '光影变化视为前景',
        'light_change_detection': light_change_detection_rate,
        'train_detection': train_detection_rate,
        'avg_foreground': np.mean(foreground_ratios)
    }

def test_hybrid_strategy():
    """测试混合策略"""
    print("\n🔧 测试策略3: 混合策略")
    
    frames = simulate_light_change_scenario()
    
    # 使用中等参数的KNN
    knn = cv2.createBackgroundSubtractorKNN(
        history=150,
        dist2Threshold=350,
        detectShadows=True     # 启用阴影检测
    )
    
    foreground_ratios = []
    light_change_detected = []
    train_detected = []
    
    for i, frame in enumerate(frames):
        # 轻度高斯模糊去噪，但不进行直方图均衡化
        processed_frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # 应用KNN
        fg_mask = knn.apply(processed_frame)
        
        # 计算前景比例（排除阴影）
        foreground_pixels = np.sum(fg_mask == 255)  # 只计算确定的前景
        fg_ratio = foreground_pixels / fg_mask.size
        foreground_ratios.append(fg_ratio)
        
        # 分析检测情况
        if 30 <= i < 50:  # 光影变化阶段
            light_change_detected.append(fg_ratio > 0.005)  # 较低阈值
        elif i >= 50:  # 列车通过阶段
            train_detected.append(fg_ratio > 0.03)  # 中等阈值
    
    # 统计结果
    light_change_detection_rate = np.mean(light_change_detected) if light_change_detected else 0
    train_detection_rate = np.mean(train_detected) if train_detected else 0
    
    print(f"   光影变化检测率: {light_change_detection_rate:.2f}")
    print(f"   列车检测率: {train_detection_rate:.2f}")
    print(f"   平均前景比例: {np.mean(foreground_ratios):.4f}")
    
    return {
        'strategy': '混合策略',
        'light_change_detection': light_change_detection_rate,
        'train_detection': train_detection_rate,
        'avg_foreground': np.mean(foreground_ratios)
    }

def analyze_real_video_strategies():
    """在真实视频上测试不同策略"""
    print("\n🎬 在真实视频上测试策略")
    
    video_path = "data/test_videos/train_enter_station.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ 测试视频不存在: {video_path}")
        return None
    
    strategies = [
        {
            'name': '均衡化策略',
            'params': {
                'algorithm': 'KNN',
                'history': 200,
                'dist2_threshold': 400,
                'use_histogram_equalization': True,
                'use_median_blur': True
            }
        },
        {
            'name': '敏感策略', 
            'params': {
                'algorithm': 'KNN',
                'history': 100,
                'dist2_threshold': 300,
                'use_histogram_equalization': False,
                'use_median_blur': False
            }
        },
        {
            'name': '混合策略',
            'params': {
                'algorithm': 'KNN', 
                'history': 150,
                'dist2_threshold': 350,
                'use_histogram_equalization': False,
                'use_median_blur': True,
                'gaussian_kernel': (3, 3)
            }
        }
    ]
    
    results = []
    
    for strategy in strategies:
        print(f"\n   测试: {strategy['name']}")
        
        try:
            gmm = GMMBackgroundSubtractor(**strategy['params'])
            gmm.setup_track_roi([(300, 200), (800, 900)])
            
            stats = gmm.process_video(video_path, max_frames=50, show_visualization=False)
            
            results.append({
                'strategy': strategy['name'],
                'avg_foreground_ratio': stats['avg_foreground_ratio'],
                'std_foreground_ratio': stats['std_foreground_ratio'],
                'frames_processed': stats['total_frames']
            })
            
            print(f"     平均前景比例: {stats['avg_foreground_ratio']:.4f}")
            print(f"     前景比例标准差: {stats['std_foreground_ratio']:.4f}")
            
        except Exception as e:
            print(f"     失败: {e}")
    
    return results

def run_comprehensive_analysis():
    """运行综合分析"""
    print("=" * 60)
    print("🤔 铁轨光影变化处理策略分析")
    print("=" * 60)
    
    # 模拟场景测试
    print("\n📊 模拟场景测试结果:")
    results_sim = [
        test_strategy_with_equalization(),
        test_strategy_treat_light_as_foreground(),
        test_hybrid_strategy()
    ]
    
    # 真实视频测试
    print("\n📊 真实视频测试结果:")
    results_real = analyze_real_video_strategies()
    
    # 输出建议
    print("\n" + "=" * 60)
    print("💡 策略选择建议:")
    print("=" * 60)
    
    if results_sim:
        best_sim = max(results_sim, key=lambda x: x['train_detection'])
        print(f"模拟场景推荐: {best_sim['strategy']}")
        print(f"  - 列车检测率: {best_sim['train_detection']:.2f}")
        print(f"  - 光影变化检测率: {best_sim['light_change_detection']:.2f}")
    
    if results_real:
        best_real = min(results_real, key=lambda x: x['std_foreground_ratio'])
        print(f"真实视频推荐: {best_real['strategy']}")
        print(f"  - 平均前景比例: {best_real['avg_foreground_ratio']:.4f}")
        print(f"  - 稳定性(标准差): {best_real['std_foreground_ratio']:.4f}")
    
    print("\n🎯 最终建议:")
    print("1. 如果光影变化是列车进出的重要指标 → 选择敏感策略")
    print("2. 如果追求稳定的背景模型 → 选择均衡化策略") 
    print("3. 如果场景复杂多变 → 选择混合策略")
    print("4. 建议在实际场景中测试多种策略")

if __name__ == "__main__":
    run_comprehensive_analysis()