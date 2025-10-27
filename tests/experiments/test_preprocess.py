#!/usr/bin/env python3
"""
预处理技术测试脚本 - 测试效果和资源消耗
"""

import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.background_subtractors.gmm_model_old1 import GMMBackgroundSubtractor

def create_test_frame_with_noise(width=800, height=600):
    """
    创建带噪声的测试帧
    
    Returns:
        带噪声的测试帧
    """
    # 创建基础图像
    frame = np.full((height, width, 3), 100, dtype=np.uint8)
    
    # 添加一些前景物体
    cv2.rectangle(frame, (200, 150), (400, 350), (200, 200, 200), -1)
    cv2.circle(frame, (600, 300), 80, (150, 150, 150), -1)
    
    # 添加高斯噪声
    noise = np.random.normal(0, 25, frame.shape).astype(np.uint8)
    frame_noisy = cv2.add(frame, noise)
    
    # 添加椒盐噪声
    salt_pepper_prob = 0.01
    noise_mask = np.random.random(frame_noisy.shape[:2])
    frame_noisy[noise_mask < salt_pepper_prob/2] = 0    # 椒噪声
    frame_noisy[noise_mask > 1 - salt_pepper_prob/2] = 255  # 盐噪声
    
    return frame_noisy

def test_gaussian_blur():
    """测试高斯模糊效果"""
    print("🔍 测试高斯模糊...")
    
    frame = create_test_frame_with_noise()
    
    start_time = time.time()
    
    # 不同核大小的高斯模糊
    kernels = [(3, 3), (5, 5), (7, 7)]
    
    for kernel_size in kernels:
        single_start = time.time()
        blurred = cv2.GaussianBlur(frame, kernel_size, 0)
        single_time = time.time() - single_start
        
        noise_reduction = np.std(frame) - np.std(blurred)
        print(f"   核大小 {kernel_size}: {single_time*1000:.2f}ms, 噪声减少: {noise_reduction:.2f}")
    
    total_time = time.time() - start_time
    print(f"   ✅ 高斯模糊测试完成 - 总耗时: {total_time*1000:.2f}ms")
    
    return True

def test_median_blur():
    """测试中值滤波效果"""
    print("🔍 测试中值滤波...")
    
    frame = create_test_frame_with_noise()
    
    start_time = time.time()
    
    # 不同核大小的中值滤波
    kernel_sizes = [3, 5, 7]
    
    for ksize in kernel_sizes:
        single_start = time.time()
        median = cv2.medianBlur(frame, ksize)
        single_time = time.time() - single_start
        
        # 计算椒盐噪声减少程度
        salt_pepper_pixels_original = np.sum((frame == 0) | (frame == 255))
        salt_pepper_pixels_filtered = np.sum((median == 0) | (median == 255))
        noise_reduction = salt_pepper_pixels_original - salt_pepper_pixels_filtered
        
        print(f"   核大小 {ksize}: {single_time*1000:.2f}ms, 椒盐噪声减少: {noise_reduction}像素")
    
    total_time = time.time() - start_time
    print(f"   ✅ 中值滤波测试完成 - 总耗时: {total_time*1000:.2f}ms")
    
    return True

def test_bilateral_filter():
    """测试双边滤波效果"""
    print("🔍 测试双边滤波...")
    
    frame = create_test_frame_with_noise()
    
    start_time = time.time()
    
    # 不同参数的双边滤波
    params = [
        {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75},
        {'d': 15, 'sigmaColor': 100, 'sigmaSpace': 100},
        {'d': 5, 'sigmaColor': 50, 'sigmaSpace': 50}
    ]
    
    for param in params:
        single_start = time.time()
        bilateral = cv2.bilateralFilter(frame, param['d'], param['sigmaColor'], param['sigmaSpace'])
        single_time = time.time() - single_start
        
        # 计算边缘保持度（通过计算梯度变化）
        original_grad = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        filtered_grad = cv2.Laplacian(cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        edge_preservation = filtered_grad / original_grad
        
        print(f"   参数 {param}: {single_time*1000:.2f}ms, 边缘保持度: {edge_preservation:.3f}")
    
    total_time = time.time() - start_time
    print(f"   ✅ 双边滤波测试完成 - 总耗时: {total_time*1000:.2f}ms")
    
    return True

def test_histogram_equalization():
    """测试直方图均衡化效果"""
    print("🔍 测试直方图均衡化...")
    
    frame = create_test_frame_with_noise()
    
    start_time = time.time()
    
    # 测试不同的直方图均衡化方法
    methods = [
        ('全局均衡化', lambda img: cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))),
        ('YUV通道均衡化', lambda img: 
            cv2.cvtColor(
                cv2.merge([cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 0]), 
                          cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 1], 
                          cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 2]]), 
                cv2.COLOR_YUV2BGR)
        ),
        ('CLAHE', lambda img: 
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            .apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        )
    ]
    
    for method_name, method_func in methods:
        single_start = time.time()
        try:
            if 'YUV' in method_name:
                result = method_func(frame)
                contrast_improvement = np.std(result) - np.std(frame)
            else:
                result = method_func(frame)
                contrast_improvement = np.std(result) - np.std(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            
            single_time = time.time() - single_start
            print(f"   {method_name}: {single_time*1000:.2f}ms, 对比度提升: {contrast_improvement:.2f}")
            
        except Exception as e:
            print(f"   {method_name}: 失败 - {e}")
    
    total_time = time.time() - start_time
    print(f"   ✅ 直方图均衡化测试完成 - 总耗时: {total_time*1000:.2f}ms")
    
    return True

def test_preprocessing_combinations():
    """测试预处理组合效果"""
    print("🔍 测试预处理组合...")
    
    frame = create_test_frame_with_noise()
    
    combinations = [
        {
            'name': '仅高斯模糊',
            'func': lambda img: cv2.GaussianBlur(img, (5, 5), 0)
        },
        {
            'name': '高斯+中值',
            'func': lambda img: cv2.medianBlur(cv2.GaussianBlur(img, (3, 3), 0), 3)
        },
        {
            'name': '中值+双边',
            'func': lambda img: cv2.bilateralFilter(cv2.medianBlur(img, 3), 9, 75, 75)
        },
        {
            'name': '均衡化+高斯+中值',
            'func': lambda img: 
                cv2.medianBlur(
                    cv2.GaussianBlur(
                        cv2.cvtColor(
                            cv2.merge([cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 0]), 
                                      cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 1], 
                                      cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 2]]), 
                            cv2.COLOR_YUV2BGR
                        ), (3, 3), 0
                    ), 3
                )
        }
    ]
    
    for combo in combinations:
        start_time = time.time()
        
        try:
            processed = combo['func'](frame)
            process_time = time.time() - start_time
            
            # 计算质量指标
            original_noise = np.std(frame)
            processed_noise = np.std(processed)
            noise_reduction = original_noise - processed_noise
            
            # 计算PSNR（峰值信噪比）
            mse = np.mean((frame - processed) ** 2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            
            print(f"   {combo['name']}:")
            print(f"     耗时: {process_time*1000:.2f}ms")
            print(f"     噪声减少: {noise_reduction:.2f}")
            print(f"     PSNR: {psnr:.2f} dB")
            
        except Exception as e:
            print(f"   {combo['name']}: 失败 - {e}")
    
    print("   ✅ 预处理组合测试完成")
    return True

def test_preprocessing_impact_on_gmm():
    """测试预处理对GMM算法的影响"""
    print("🔍 测试预处理对GMM的影响...")
    
    video_path = "data/test_videos/train_enter_station.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ 测试视频不存在: {video_path}")
        return False
    
    # 不同的预处理配置
    preprocess_configs = [
        {
            'name': '无预处理',
            'params': {
                'gaussian_kernel': (0, 0),
                'use_median_blur': False,
                'use_bilateral_filter': False,
                'use_histogram_equalization': False
            }
        },
        {
            'name': '基础去噪',
            'params': {
                'gaussian_kernel': (5, 5),
                'use_median_blur': True,
                'use_bilateral_filter': False,
                'use_histogram_equalization': False
            }
        },
        {
            'name': '增强去噪',
            'params': {
                'gaussian_kernel': (3, 3),
                'use_median_blur': True,
                'use_bilateral_filter': True,
                'use_histogram_equalization': False
            }
        },
        {
            'name': '完整预处理',
            'params': {
                'gaussian_kernel': (3, 3),
                'use_median_blur': True,
                'use_bilateral_filter': True,
                'use_histogram_equalization': True
            }
        }
    ]
    
    results = []
    
    for config in preprocess_configs:
        print(f"\n   测试配置: {config['name']}")
        
        try:
            start_time = time.time()
            
            gmm = GMMBackgroundSubtractor(
                'MOG2',
                history=200,
                var_threshold=10,
                **config['params']
            )
            
            gmm.setup_track_roi([(300, 200), (800, 900)])
            
            # 处理少量帧进行测试
            stats = gmm.process_video(video_path, max_frames=30, show_visualization=False)
            
            process_time = time.time() - start_time
            
            results.append({
                'name': config['name'],
                'avg_foreground_ratio': stats['avg_foreground_ratio'],
                'process_time': process_time,
                'frames_processed': stats['total_frames']
            })
            
            print(f"     平均前景比例: {stats['avg_foreground_ratio']:.4f}")
            print(f"     处理时间: {process_time:.2f}s")
            print(f"     帧率: {stats['total_frames']/process_time:.2f} FPS")
            
        except Exception as e:
            print(f"     失败: {e}")
    
    # 输出比较结果
    print("\n📊 GMM预处理效果比较:")
    for result in results:
        print(f"   {result['name']}:")
        print(f"     前景比例: {result['avg_foreground_ratio']:.4f}")
        print(f"     处理速度: {result['frames_processed']/result['process_time']:.2f} FPS")
    
    return True

def visualize_preprocessing_effects():
    """可视化预处理效果"""
    print("🎨 可视化预处理效果...")
    
    frame = create_test_frame_with_noise(400, 300)  # 小尺寸用于显示
    
    # 不同的预处理方法
    methods = [
        ('原图', lambda img: img),
        ('高斯模糊', lambda img: cv2.GaussianBlur(img, (5, 5), 0)),
        ('中值滤波', lambda img: cv2.medianBlur(img, 3)),
        ('双边滤波', lambda img: cv2.bilateralFilter(img, 9, 75, 75)),
        ('组合去噪', lambda img: cv2.medianBlur(cv2.GaussianBlur(img, (3, 3), 0), 3))
    ]
    
    try:
        for i, (name, func) in enumerate(methods):
            processed = func(frame)
            cv2.imshow(f'Preprocessing: {name}', processed)
        
        print("   💡 按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("   ✅ 可视化完成")
        return True
        
    except Exception as e:
        print(f"   ❌ 可视化失败: {e}")
        return False

def run_all_preprocess_tests():
    """运行所有预处理测试"""
    print("=" * 60)
    print("🧪 预处理技术全面测试")
    print("=" * 60)
    
    tests = [
        ("高斯模糊测试", test_gaussian_blur),
        ("中值滤波测试", test_median_blur),
        ("双边滤波测试", test_bilateral_filter),
        ("直方图均衡化测试", test_histogram_equalization),
        ("预处理组合测试", test_preprocessing_combinations),
        ("预处理对GMM影响测试", test_preprocessing_impact_on_gmm),
        ("预处理效果可视化", visualize_preprocessing_effects),
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
    print(f"📊 预处理测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有预处理测试通过！")
    else:
        print("⚠️  部分测试失败")
    
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    run_all_preprocess_tests()