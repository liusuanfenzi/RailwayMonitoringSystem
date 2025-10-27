#!/usr/bin/env python3
"""
GMM演示脚本 - 三个独立窗口可视化
"""

import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.background_subtractors.gmm_model_old1 import GMMBackgroundSubtractor

def main():
    print("🎬 GMM视频处理演示（三个独立窗口）")
    
    video_path = "data/test_videos/train_enter_station.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    # 测试不同算法和参数
    (x1,y1)=[60,350]
    (x2,y2)=[590,900]
    test_configs = [
        {
            'name': 'MOG2-默认',
            'algorithm': 'MOG2',
            'params': {'history': 100, 'var_threshold': 16},
            'roi': [(x1,y1), (x2,y2)]
        },
        # {
        #     'name': 'MOG2-敏感', 
        #     'algorithm': 'MOG2',
        #     'params': {'history': 200, 'var_threshold': 8},
        #     'roi': [(x1,y1), (x2,y2)]
        # },
        # {
        #     'name': 'KNN-默认',
        #     'algorithm': 'KNN', 
        #     'params': {'history': 500, 'dist2_threshold': 400},
        #     'roi': [(x1,y1), (x2,y2)]
        # }
    ]
    
    print(f"📹 处理视频: {video_path}")
    print("💡 将显示三个独立窗口:")
    print("   - Original + ROI: 原图+ROI区域")
    print("   - Global Foreground Mask: 全局前景掩码") 
    print("   - ROI Foreground Mask: ROI区域前景掩码")
    print("💡 按 'q' 退出当前配置的处理")
    print("=" * 60)
    
    for config in test_configs:
        print(f"\n🔧 测试配置: {config['name']}")
        print(f"   算法: {config['algorithm']}")
        print(f"   参数: {config['params']}")
        
        try:
            gmm = GMMBackgroundSubtractor(config['algorithm'], **config['params'])
            gmm.setup_track_roi(config['roi'])
            
            # 处理视频并显示可视化
            stats = gmm.process_video(video_path, max_frames=700, show_visualization=True)
            
            print(f"   📊 处理帧数: {stats['total_frames']}")
            print(f"   📊 平均前景比例: {stats['avg_foreground_ratio']:.4f}")
            print(f"   📊 最大前景比例: {stats['max_foreground_ratio']:.4f}")
            print(f"   📊 最小前景比例: {stats['min_foreground_ratio']:.4f}")
            
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 视频处理演示完成")

if __name__ == "__main__":
    main()

