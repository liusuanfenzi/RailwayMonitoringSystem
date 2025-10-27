# #!/usr/bin/env python3
# """
# 铁路监控系统主程序
# """

# import argparse
# from src.core.system_controller import SystemController

# def main():
#     parser = argparse.ArgumentParser(description='铁路智能监控系统')
#     parser.add_argument('--config', default='config/system/main.yaml', help='配置文件路径')
#     parser.add_argument('--camera', help='指定相机ID')
#     parser.add_argument('--debug', action='store_true', help='调试模式')
    
#     args = parser.parse_args()
    
#     # 启动系统
#     controller = SystemController(args.config)
#     controller.start()
    
#     print("🚄 铁路监控系统启动成功！")

# if __name__ == "__main__":
#     main()
