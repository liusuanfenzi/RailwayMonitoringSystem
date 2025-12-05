# # 创建一个极简的YOLO测试脚本 test_yolo_simple.py
# from ultralytics import YOLO
# import cv2
# import numpy as np

# print("测试基础YOLO模型...")
# model = YOLO('yolov8n.pt')

# # 创建测试图像
# test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# print("开始推理测试...")
# try:
#     results = model(test_img, verbose=True)
#     print("✅ 基础YOLO推理成功")
#     print(f"检测结果: {len(results)}")
# except Exception as e:
#     print(f"❌ 基础YOLO推理失败: {e}")


#=================================================
# 检查模型是否需要显式转换为TensorRT
# def check_model_format():
#     from ultralytics import YOLO
#     import os
    
#     model_path = 'yolov8n.pt'
#     print(f"模型路径: {model_path}")
#     print(f"模型存在: {os.path.exists(model_path)}")
    
#     # 尝试导出为TensorRT格式
#     try:
#         model = YOLO(model_path)
#         print("尝试导出为TensorRT格式...")
#         model.export(format='engine', imgsz=640)
#         print("✅ TensorRT导出成功")
#     except Exception as e:
#         print(f"❌ TensorRT导出失败: {e}")

# check_model_format()


#=================================================
# 检查内存使用情况
def check_system_status():
    import psutil
    import torch
    
    # 系统内存
    memory = psutil.virtual_memory()
    print(f"系统内存: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent}%)")
    
    # GPU内存
    if torch.cuda.is_available():
        print(f"GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f}GB 已分配")
        print(f"GPU缓存: {torch.cuda.memory_reserved()/1024**3:.2f}GB 已保留")
    
    # 交换空间
    swap = psutil.swap_memory()
    print(f"交换空间: {swap.used/1024**3:.1f}GB / {swap.total/1024**3:.1f}GB")

check_system_status()