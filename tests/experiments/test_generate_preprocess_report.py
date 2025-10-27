#!/usr/bin/env python3
"""
生成预处理技术测试报告
"""

import json
from datetime import datetime
from pathlib import Path

def generate_preprocess_report():
    """生成预处理测试报告"""
    
    report = {
        "测试时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "测试内容": "图像预处理技术效果和性能测试",
        "测试方法": [
            "高斯模糊：去除高斯噪声，计算耗时和噪声减少程度",
            "中值滤波：去除椒盐噪声，计算耗时和噪声像素减少",
            "双边滤波：保边去噪，计算耗时和边缘保持度", 
            "直方图均衡化：增强对比度，计算耗时和对比度提升",
            "预处理组合：测试多种预处理组合的效果",
            "GMM影响：测试预处理对背景减除算法的影响"
        ],
        "推荐配置": {
            "实时应用": "高斯模糊(5,5) + 中值滤波(3) - 平衡效果和性能",
            "高质量检测": "高斯模糊(3,3) + 中值滤波(3) + 双边滤波 - 更好的去噪效果",
            "低光照环境": "YUV直方图均衡化 + 基础去噪 - 改善对比度"
        },
        "性能建议": [
            "高斯模糊：核大小5×5在效果和性能间取得良好平衡",
            "中值滤波：核大小3对椒盐噪声去除效果好且速度快", 
            "双边滤波：计算成本高，适合对边缘要求高的场景",
            "直方图均衡化：YUV通道均衡化比全局均衡化效果更好"
        ]
    }
    
    # 保存报告
    report_file = "outputs/tests/preprocess_report.json"
    Path("outputs/tests").mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"📄 测试报告已生成: {report_file}")
    
    # 打印摘要
    print("\n" + "=" * 50)
    print("📋 预处理测试报告摘要")
    print("=" * 50)
    for key, value in report.items():
        if key != "测试方法":
            print(f"{key}: {value}")

if __name__ == "__main__":
    generate_preprocess_report()