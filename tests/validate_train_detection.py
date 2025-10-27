#!/usr/bin/env python3
"""
验证火车进站检测效果
"""

import cv2
import numpy as np
from models.background_subtractors.gmm_model_old1 import GMMBackgroundSubtractor

def validate_train_detection(video_path: str):
    """验证GMM对火车进站的检测效果"""
    
    print("🚂 验证火车进站检测效果...")
    
    # 测试三种配置
    configs = [
        {'name': '保守', 'history': 500, 'var_threshold': 16},
        {'name': '平衡', 'history': 200, 'var_threshold': 10},
        {'name': '敏感', 'history': 100, 'var_threshold': 8},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n🔧 测试 {config['name']} 配置...")
        
        gmm = GMMBackgroundSubtractor(
            'MOG2',
            history=config['history'],
            var_threshold=config['var_threshold'],
            detect_shadows=False
        )
        
        # 处理视频并分析帧间变化
        cap = cv2.VideoCapture(video_path)
        foreground_pixels = []
        
        prev_frame = None
        frame_changes = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 转换为灰度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 应用GMM
            fg_mask = gmm.apply(gray)
            foreground_pixels.append(np.sum(fg_mask > 0))
            
            # 分析帧间变化（用于检测火车进入ROI）
            if prev_frame is not None:
                frame_diff = cv2.absdiff(prev_frame, gray)
                change_ratio = np.sum(frame_diff > 25) / (1080 * 1116)
                frame_changes.append(change_ratio)
            
            prev_frame = gray
        
        cap.release()
        
        # 分析结果
        avg_foreground = np.mean(foreground_pixels)
        max_foreground = np.max(foreground_pixels)
        foreground_ratio = avg_foreground / (1080 * 1116)
        
        # 检测显著运动帧（可能表示火车进入）
        significant_changes = [x for x in frame_changes if x > 0.02]  # 2%的变化阈值
        train_enter_frames = len(significant_changes)
        
        results[config['name']] = {
            'avg_foreground': avg_foreground,
            'max_foreground': max_foreground,
            'foreground_ratio': foreground_ratio,
            'train_enter_events': train_enter_frames,
            'sensitivity': "高" if foreground_ratio > 0.08 else "中" if foreground_ratio > 0.06 else "低"
        }
        
        print(f"  前景比例: {foreground_ratio:.3f}")
        print(f"  最大前景: {max_foreground}")
        print(f"  检测到显著运动帧: {train_enter_frames}")
        print(f"  灵敏度: {results[config['name']]['sensitivity']}")
    
    # 输出推荐
    print("\n🎯 配置推荐:")
    best_config = max(results.items(), key=lambda x: x[1]['foreground_ratio'])
    print(f"  推荐使用: {best_config[0]} 配置")
    print(f"  理由: 前景比例最高 ({best_config[1]['foreground_ratio']:.3f})")
    
    return results

if __name__ == "__main__":
    results = validate_train_detection("data/test_videos/train_enter_station.mp4")