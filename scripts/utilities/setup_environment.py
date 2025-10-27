#!/usr/bin/env python3
"""
环境自动配置脚本 - 国内镜像源版
"""

import subprocess
import sys
import os


def setup_chinese_mirrors():
    """配置国内镜像源"""
    mirrors = {
        'tsinghua': 'https://pypi.tuna.tsinghua.edu.cn/simple',
        'aliyun': 'https://mirrors.aliyun.com/pypi/simple',
        'douban': 'https://pypi.douban.com/simple',
        'huawei': 'https://mirrors.huaweicloud.com/repository/pypi/simple'
    }

    # 选择镜像源（清华源最稳定）
    selected_mirror = mirrors['tsinghua']
    print(f"🎯 使用镜像源: {selected_mirror}")

    return selected_mirror


def install_packages_with_mirror(mirror_url):
    """使用镜像源安装包"""
    packages = [
        "opencv-python==4.8.1.78",
        "numpy==1.24.3",
        "matplotlib==3.7.2",
        "scipy==1.11.3",
        "scikit-learn==1.3.0",
        "pandas==2.0.3",
        "jupyter==1.0.0",
        "filterpy==1.4.5",
        "scikit-image==0.21.0",
        "deep-sort-realtime==1.3.2"
    ]

    # PyTorch需要单独处理（有官方中国源）
    torch_packages = [
        "torch==2.0.1",
        "torchvision==0.15.2"
    ]

    # ultralytics也需要单独处理
    ultralytics_package = "ultralytics==8.0.186"

    success_count = 0
    total_count = len(packages) + len(torch_packages) + 1

    print("🚀 开始安装基础包...")
    for package in packages:
        try:
            # 使用国内镜像源安装
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                package, "-i", mirror_url, "--trusted-host",
                mirror_url.split('//')[1].split('/')[0]
            ])
            print(f"✅ {package} 安装成功")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} 安装失败: {e}")
            # 尝试使用默认源重试
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} 使用默认源安装成功")
                success_count += 1
            except:
                print(f"❌ {package} 完全安装失败")

    print("\n🔬 开始安装PyTorch（使用官方中国源）...")
    for torch_pkg in torch_packages:
        try:
            # PyTorch官方中国源
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                torch_pkg, "-f", "https://download.pytorch.org/whl/torch_stable.html"
            ])
            print(f"✅ {torch_pkg} 安装成功")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ {torch_pkg} 安装失败: {e}")

    print("\n🤖 开始安装ultralytics（YOLOv8）...")
    try:
        # ultralytics使用国内源
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            ultralytics_package, "-i", mirror_url, "--trusted-host",
            mirror_url.split('//')[1].split('/')[0]
        ])
        print(f"✅ {ultralytics_package} 安装成功")
        success_count += 1
    except subprocess.CalledProcessError as e:
        print(f"❌ {ultralytics_package} 安装失败: {e}")

    return success_count, total_count


def create_requirements_file():
    """创建requirements.txt文件备用"""
    requirements_content = """opencv-python==4.8.1.78
numpy==1.24.3
matplotlib==3.7.2
scipy==1.11.3
scikit-learn==1.3.0
pandas==2.0.3
jupyter==1.0.0
filterpy==1.4.5
scikit-image==0.21.0
deep-sort-realtime==1.3.2
torch==2.0.1
torchvision==0.15.2
ultralytics==8.0.186
"""

    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements_content)
    print("📁 已创建requirements.txt文件")


def main():
    print("=" * 50)
    print("🤖 智能视频分析项目 - 环境自动配置脚本")
    print("=" * 50)

    # 配置镜像源
    mirror_url = setup_chinese_mirrors()

    # 安装包
    success_count, total_count = install_packages_with_mirror(mirror_url)

    # 创建requirements文件
    create_requirements_file()

    # 输出结果
    print("\n" + "=" * 50)
    print("📊 安装结果统计:")
    print(f"✅ 成功安装: {success_count}/{total_count} 个包")

    if success_count == total_count:
        print("🎉 所有包安装成功！环境配置完成！")
    else:
        print("⚠️  部分包安装失败，请检查网络连接后重试")
        print("💡 提示: 可以手动运行: pip install -r requirements.txt")

    print("=" * 50)


if __name__ == "__main__":
    main()
