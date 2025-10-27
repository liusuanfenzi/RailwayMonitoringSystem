import cv2
import numpy as np
from matplotlib import pyplot as plt

def global_threshold_binarization(frame):
    """全局阈值二值化"""
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 使用Otsu's方法自动计算最佳阈值
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def adaptive_threshold_binarization(frame):
    """自适应阈值二值化"""
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 使用高斯自适应阈值
    adaptive_binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return adaptive_binary

def histogram_based_binarization(frame):
    """基于直方图的二值化"""
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # 找到直方图的峰值作为阈值
    # 这里使用简单的方法：找到直方图最大值的索引
    peak_threshold = np.argmax(hist[1:]) + 1  # 跳过0值
    
    # 应用阈值
    _, binary = cv2.threshold(gray, peak_threshold, 255, cv2.THRESH_BINARY)
    
    return binary, hist

def process_video(video_path, max_frames=100):
    """
    处理视频并显示二值化结果
    
    参数:
    video_path: 视频文件路径
    max_frames: 最大处理帧数（避免处理整个长视频）
    """
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("错误：无法打开视频文件")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret or frame_count >= max_frames:
            break
            
        # 调整帧大小以便显示
        frame = cv2.resize(frame, (640, 480))
        
        # 应用三种二值化方法
        global_binary = global_threshold_binarization(frame)
        adaptive_binary = adaptive_threshold_binarization(frame)
        hist_binary, hist = histogram_based_binarization(frame)
        
        # 创建显示图像
        display_frame = create_display_image(frame, global_binary, adaptive_binary, hist_binary)
        
        # 显示结果
        cv2.imshow('二值化方法比较', display_frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

def create_display_image(original, global_bin, adaptive_bin, hist_bin):
    """创建包含所有结果的显示图像"""
    
    # 将二值图像转换为3通道以便与原始图像拼接
    global_bin_color = cv2.cvtColor(global_bin, cv2.COLOR_GRAY2BGR)
    adaptive_bin_color = cv2.cvtColor(adaptive_bin, cv2.COLOR_GRAY2BGR)
    hist_bin_color = cv2.cvtColor(hist_bin, cv2.COLOR_GRAY2BGR)
    
    # 添加标签
    def add_label(img, text):
        labeled_img = img.copy()
        cv2.putText(labeled_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        return labeled_img
    
    original = add_label(original, "Original")
    global_bin_color = add_label(global_bin_color, "Global Threshold")
    adaptive_bin_color = add_label(adaptive_bin_color, "Adaptive Threshold")
    hist_bin_color = add_label(hist_bin_color, "Histogram Based")
    
    # 水平拼接
    top_row = np.hstack([original, global_bin_color])
    bottom_row = np.hstack([adaptive_bin_color, hist_bin_color])
    
    # 垂直拼接
    result = np.vstack([top_row, bottom_row])
    
    return result

def analyze_single_frame(video_path, frame_number=0):
    """分析单帧图像的详细二值化效果"""
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("错误：无法读取指定帧")
        return
    
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 应用二值化方法
    global_binary = global_threshold_binarization(frame)
    adaptive_binary = adaptive_threshold_binarization(frame)
    hist_binary, hist = histogram_based_binarization(frame)
    
    # 使用matplotlib显示详细结果
    plt.figure(figsize=(15, 10))
    
    # 显示原始图像和灰度图
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')
    
    # 显示直方图
    plt.subplot(2, 3, 3)
    plt.plot(hist)
    plt.title('Gray Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    # 显示二值化结果
    plt.subplot(2, 3, 4)
    plt.imshow(global_binary, cmap='gray')
    plt.title('Global Threshold (Otsu)')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(adaptive_binary, cmap='gray')
    plt.title('Adaptive Threshold')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(hist_binary, cmap='gray')
    plt.title('Histogram Based')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 使用方法1：使用摄像头实时处理
    # process_video(0)  # 0表示默认摄像头
    
    # 使用方法2：处理视频文件
    video_file = "data/test_videos/train_enter_station.mp4"  # 请替换为您的视频文件路径
    
    # 如果视频文件存在，进行处理
    try:
        # 实时视频处理比较
        print("开始视频处理，按'q'键退出...")
        process_video(video_file, max_frames=700)
        
        # 详细分析第一帧
        print("显示详细分析结果...")
        analyze_single_frame(video_file, 0)
        
    except Exception as e:
        print(f"处理视频时出错: {e}")
        print("请确保视频文件路径正确，或者使用摄像头：")
        print("取消注释 process_video(0) 来使用摄像头")