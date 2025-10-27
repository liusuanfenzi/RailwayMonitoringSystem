#!/usr/bin/env python3
"""
基于真实视频的铁轨光影变化策略实验
"""

from utils.video.video_utils import VideoReader
from models.background_subtractors.gmm_model_old1 import GMMBackgroundSubtractor
import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LightChangeExperiment:
    """光影变化策略实验"""

    def __init__(self, video_path):
        self.video_path = video_path
        self.results = {}

    def analyze_light_changes(self, frame_sequence):
        """分析帧序列中的亮度变化"""
        brightness_changes = []

        for frame in frame_sequence:
            if frame is not None:
                # 转换为灰度图计算平均亮度
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(gray)
                brightness_changes.append(avg_brightness)

        return brightness_changes

    def detect_significant_light_changes(self, brightness_changes, threshold=10):
        """检测显著的亮度变化"""
        changes = []
        for i in range(1, len(brightness_changes)):
            change = abs(brightness_changes[i] - brightness_changes[i-1])
            if change > threshold:
                changes.append(
                    (i, change, brightness_changes[i] - brightness_changes[i-1]))
        return changes

    def run_strategy_comparison(self, max_frames=700):
        """运行策略比较实验"""
        print("🎬 开始真实视频策略比较实验...")

        if not Path(self.video_path).exists():
            print(f"❌ 视频文件不存在: {self.video_path}")
            return

        # 策略配置
        strategies = [
            {
                'name': 'KNN-equalization',
                'algorithm': 'KNN',
                'params': {
                    'history': 200,
                    'dist2_threshold': 400,
                    'use_histogram_equalization': True,
                    'use_median_blur': True,
                    'gaussian_kernel': (5, 5)
                },
                'roi': [(50, 200), (600, 900)]
            },
            {
                'name': 'KNN-sensitive',
                'algorithm': 'KNN',
                'params': {
                    'history': 100,
                    'dist2_threshold': 300,
                    'use_histogram_equalization': False,
                    'use_median_blur': False
                },
                'roi': [(50, 200), (600, 900)]
            },
            {
                'name': 'KNN-mix',
                'algorithm': 'KNN',
                'params': {
                    'history': 150,
                    'dist2_threshold': 350,
                    'use_histogram_equalization': False,
                    'use_median_blur': True,
                    'gaussian_kernel': (3, 3)
                },
                'roi': [(50, 200), (600, 900)]
            },
            {
                'name': 'MOG2-comparison',
                'algorithm': 'MOG2',
                'params': {
                    'history': 200,
                    'var_threshold': 16,
                    'use_histogram_equalization': False,
                    'use_median_blur': True
                },
                'roi': [(50, 200), (600, 900)]
            }
        ]

        reader = VideoReader(self.video_path)
        frames = []
        frame_count = 0

        # 读取帧序列
        print("📥 读取视频帧...")
        while frame_count < max_frames:
            frame = reader.read_frame()
            if frame is None:
                break
            frames.append(frame)
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"   已读取 {frame_count} 帧")

        reader.release()

        if not frames:
            print("❌ 无法读取视频帧")
            return

        # 分析亮度变化
        print("📊 分析视频亮度变化...")
        brightness_changes = self.analyze_light_changes(frames)
        significant_changes = self.detect_significant_light_changes(
            brightness_changes)

        print(f"   总帧数: {len(frames)}")
        print(f"   检测到显著亮度变化: {len(significant_changes)} 次")

        # 运行各策略
        for strategy in strategies:
            print(f"\n🔧 测试策略: {strategy['name']}")

            try:
                detector = GMMBackgroundSubtractor(
                    algorithm=strategy['algorithm'],
                    **strategy['params']
                )
                detector.setup_track_roi(strategy['roi'])

                foreground_ratios = []
                processing_times = []

                for i, frame in enumerate(frames):
                    start_time = time.time()

                    result = detector.apply_with_roi_analysis(frame)

                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)

                    foreground_ratio = result['full_frame']['foreground_ratio']
                    foreground_ratios.append(foreground_ratio)

                # 计算策略表现指标
                avg_foreground = np.mean(foreground_ratios)
                std_foreground = np.std(foreground_ratios)
                avg_processing_time = np.mean(processing_times)

                # 检测亮度变化期间的响应
                light_change_responses = []
                for change_frame, change_amount, _ in significant_changes:
                    if change_frame < len(foreground_ratios):
                        light_change_responses.append(
                            foreground_ratios[change_frame])

                avg_light_response = np.mean(
                    light_change_responses) if light_change_responses else 0

                self.results[strategy['name']] = {
                    'avg_foreground': avg_foreground,
                    'std_foreground': std_foreground,
                    'avg_processing_time': avg_processing_time,
                    'avg_light_response': avg_light_response,
                    'foreground_ratios': foreground_ratios,
                    'processing_times': processing_times
                }

                print(f"   ✅ 平均前景比例: {avg_foreground:.4f}")
                print(f"   ✅ 前景稳定性: {std_foreground:.4f}")
                print(f"   ✅ 平均处理时间: {avg_processing_time*1000:.2f}ms")
                print(f"   ✅ 亮度变化响应: {avg_light_response:.4f}")

            except Exception as e:
                print(f"   ❌ 策略测试失败: {e}")

        return self.results, brightness_changes, significant_changes

    def visualize_results(self, results, brightness_changes, significant_changes):
        """可视化实验结果"""
        print("\n📈 生成实验结果可视化...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 亮度变化趋势
        ax1.plot(brightness_changes, label='avg_brightness',
                 color='blue', alpha=0.7)
        ax1.set_title('video_brightness_trend')
        ax1.set_xlabel('frame_index')
        ax1.set_ylabel('brightness')
        ax1.grid(True, alpha=0.3)

        # 标记显著变化点
        for change_frame, change_amount, direction in significant_changes:
            color = 'red' if direction > 0 else 'green'
            ax1.axvline(x=change_frame, color=color, alpha=0.5, linestyle='--')

        # 2. 各策略前景比例对比
        for strategy_name, result in results.items():
            ax2.plot(result['foreground_ratios'],
                     label=strategy_name, alpha=0.7)
        ax2.set_title('each_strategy_foreground_ratio')
        ax2.set_xlabel('frame_index')
        ax2.set_ylabel('foreground_ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 性能对比
        strategy_names = list(results.keys())
        processing_times = [
            results[name]['avg_processing_time'] * 1000 for name in strategy_names]
        bars = ax3.bar(strategy_names, processing_times, alpha=0.7)
        ax3.set_title('avg_processing_time')
        ax3.set_ylabel('processing_time (ms)')

        # 在柱状图上添加数值
        for bar, time_val in zip(bars, processing_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{time_val:.1f}ms', ha='center', va='bottom')

        # 4. 综合表现雷达图
        metrics = ['detect_sensitivity', 'stability', 'processing_time', 'light_response']
        strategy_performance = {}

        for strategy_name, result in results.items():
            # 归一化各项指标
            sensitivity = result['avg_foreground']  # 前景比例越高，灵敏度越高
            stability = 1 / (result['std_foreground'] + 0.001)  # 标准差越小，稳定性越好
            speed = 1 / (result['avg_processing_time'] + 0.001)  # 处理时间越短，速度越快
            light_response = result['avg_light_response']  # 亮度变化响应

            # 归一化到0-1范围
            max_sensitivity = max([r['avg_foreground']
                                  for r in results.values()])
            max_stability = max([1/(r['std_foreground']+0.001)
                                for r in results.values()])
            max_speed = max([1/(r['avg_processing_time']+0.001)
                            for r in results.values()])
            max_light_response = max([r['avg_light_response']
                                     for r in results.values()])

            normalized_performance = [
                sensitivity / max_sensitivity,
                stability / max_stability,
                speed / max_speed,
                light_response / (max_light_response + 0.001)
            ]

            strategy_performance[strategy_name] = normalized_performance

        # 绘制雷达图
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        for strategy_name, performance in strategy_performance.items():
            performance += performance[:1]  # 闭合图形
            ax4.plot(angles, performance, 'o-', label=strategy_name)
            ax4.fill(angles, performance, alpha=0.1)

        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics)
        ax4.set_title('strategy_performance_radar')
        ax4.legend(loc='upper right')

        plt.tight_layout()

        # 保存图像
        output_dir = Path("outputs/experiments")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "light_change_strategy_comparison.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

        print(f"💾 可视化结果已保存到: {output_dir}")

    def generate_report(self, results, significant_changes):
        """生成实验报告"""
        print("\n📄 生成实验报告...")

        report = {
            "实验时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            "测试视频": self.video_path,
            "显著亮度变化次数": len(significant_changes),
            "策略比较结果": {}
        }

        # 找出最佳策略
        best_sensitivity = max(
            results.items(), key=lambda x: x[1]['avg_foreground'])
        best_stability = min(
            results.items(), key=lambda x: x[1]['std_foreground'])
        best_speed = min(
            results.items(), key=lambda x: x[1]['avg_processing_time'])
        best_light_response = max(
            results.items(), key=lambda x: x[1]['avg_light_response'])

        for strategy_name, result in results.items():
            report["策略比较结果"][strategy_name] = {
                "平均前景比例": f"{result['avg_foreground']:.4f}",
                "前景稳定性(标准差)": f"{result['std_foreground']:.4f}",
                "平均处理时间(ms)": f"{result['avg_processing_time']*1000:.2f}",
                "亮度变化响应": f"{result['avg_light_response']:.4f}"
            }

        report["推荐策略"] = {
            "最高灵敏度": best_sensitivity[0],
            "最稳定": best_stability[0],
            "最快速度": best_speed[0],
            "最佳亮度响应": best_light_response[0]
        }

        # 保存报告
        import json
        report_file = Path(
            "outputs/experiments/light_change_experiment_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"💾 实验报告已保存到: {report_file}")

        # 打印摘要
        print("\n" + "=" * 60)
        print("📋 实验报告摘要")
        print("=" * 60)
        print(f"测试视频: {Path(self.video_path).name}")
        print(f"显著亮度变化: {len(significant_changes)} 次")
        print("\n策略表现排名:")
        print("1. 最高灵敏度:", best_sensitivity[0])
        print("2. 最稳定:", best_stability[0])
        print("3. 最快速度:", best_speed[0])
        print("4. 最佳亮度响应:", best_light_response[0])


def main():
    """主实验函数"""
    video_path = "data/test_videos/train_enter_station.mp4"

    if not Path(video_path).exists():
        print(f"❌ 视频文件不存在: {video_path}")
        print("💡 请确保测试视频文件存在")
        return

    print("=" * 60)
    print("🎬 铁轨光影变化策略真实视频实验")
    print("=" * 60)

    experiment = LightChangeExperiment(video_path)

    # 运行实验
    results, brightness_changes, significant_changes = experiment.run_strategy_comparison(
        max_frames=700)

    if results:
        # 可视化结果
        experiment.visualize_results(
            results, brightness_changes, significant_changes)

        # 生成报告
        experiment.generate_report(results, significant_changes)

        print("\n🎉 实验完成！")
    else:
        print("❌ 实验失败，无有效结果")


if __name__ == "__main__":
    main()
