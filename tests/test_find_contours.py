import cv2
import numpy as np
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

# 添加模块路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.video.video_utils import ROIManager
from models.background_subtractors.gmm_model import GMMBackgroundSubtractor

class DebrisContourDetector:
    """杂物/异物轮廓检测器"""
    
    def __init__(self, min_contour_area=100, max_contour_area=5000, 
                 aspect_ratio_threshold=5.0, roi_manager=None):
        """
        初始化轮廓检测器
        
        Args:
            min_contour_area: 最小轮廓面积阈值
            max_contour_area: 最大轮廓面积阈值  
            aspect_ratio_threshold: 长宽比阈值，过滤细长轮廓
            roi_manager: ROI管理器实例
        """
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.roi_manager = roi_manager
        
        # 背景减除器
        self.background_subtractor = GMMBackgroundSubtractor(
            algorithm='MOG2',
            preprocess_mode='basic',
            history=200,
            var_threshold=16,
            detect_shadows=False,
            noise_reduction='light'
        )
        
        self.roi_name = 'debris_detection_roi'
        self.roi_points = None
        
        print("✅ 杂物轮廓检测器初始化完成")
        print(f"   - 轮廓面积范围: {min_contour_area} ~ {max_contour_area} 像素")
        print(f"   - 长宽比阈值: {aspect_ratio_threshold}")
    
    def setup_roi(self, points, roi_name='debris_detection_roi'):
        """设置检测ROI区域"""
        self.roi_points = points
        self.roi_name = roi_name
        self.background_subtractor.setup_single_roi(points, roi_name)
        if self.roi_manager:
            self.roi_manager.add_roi(roi_name, points)
        print(f"🎯 设置杂物检测ROI: {points}")
    
    def process_frame(self, frame, learning_rate=0.005):
        """
        处理单帧图像，检测杂物轮廓
        
        Args:
            frame: 输入图像
            learning_rate: 背景模型学习率
            
        Returns:
            original_with_roi: 带ROI标记的原图
            roi_foreground_mask: ROI区域的前景掩码
            roi_contour_frame: ROI区域的轮廓检测结果
            contours: 检测到的轮廓列表
        """
        try:
            # 1. 复制原图用于绘制ROI
            original_with_roi = frame.copy()
            
            # 2. 应用背景减除获取前景掩码
            bg_results = self.background_subtractor.apply_with_roi_analysis(
                frame, learning_rate=learning_rate
            )
            
            # 3. 获取ROI区域的前景掩码
            if self.roi_name in bg_results:
                roi_foreground_mask = bg_results[self.roi_name]['mask']
                
                # 4. 轮廓检测
                contours = self._extract_contours(roi_foreground_mask)
                
                # 5. 过滤和验证轮廓
                filtered_contours = self._filter_contours(contours)
                
                # 6. 创建ROI区域的轮廓可视化
                roi_contour_frame = self._create_roi_contour_visualization(
                    frame, roi_foreground_mask, filtered_contours
                )
                
                # 7. 在原图上绘制ROI区域
                original_with_roi = self._draw_roi_on_original(original_with_roi)
                
                # 8. 输出检测信息
                self._print_detection_info(filtered_contours, bg_results[self.roi_name])
                
                return original_with_roi, roi_foreground_mask, roi_contour_frame, filtered_contours
            else:
                print("⚠️ 未找到ROI区域的前景掩码")
                # 返回空的ROI区域图像
                h, w = frame.shape[:2]
                empty_mask = np.zeros((h, w), dtype=np.uint8)
                empty_contour = np.zeros((h, w, 3), dtype=np.uint8)
                return original_with_roi, empty_mask, empty_contour, []
            
        except Exception as e:
            print(f"❌ 轮廓检测错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回错误状态下的默认图像
            h, w = frame.shape[:2]
            original_with_roi = frame.copy()
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            empty_contour = np.zeros((h, w, 3), dtype=np.uint8)
            return original_with_roi, empty_mask, empty_contour, []
    
    def _extract_contours(self, foreground_mask):
        """从前景掩码中提取轮廓"""
        # 使用findContours提取轮廓
        contours, hierarchy = cv2.findContours(
            foreground_mask, 
            cv2.RETR_EXTERNAL,  # 只检测外部轮廓
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        return contours
    
    def _filter_contours(self, contours):
        """过滤和验证轮廓"""
        filtered_contours = []
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 面积过滤
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
            
            # 计算轮廓的边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 长宽比过滤（过滤细长轮廓，可能是噪声）
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            if aspect_ratio > self.aspect_ratio_threshold:
                continue
            
            # 轮廓复杂度过滤（使用轮廓近似）
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 如果近似后的点数太少，可能是噪声
            if len(approx) < 3:
                continue
            
            filtered_contours.append(contour)
        
        return filtered_contours
    
    def _create_roi_contour_visualization(self, frame, foreground_mask, contours):
        """创建ROI区域的轮廓可视化"""
        # 裁剪ROI区域
        if self.roi_points:
            x1, y1 = self.roi_points[0]
            x2, y2 = self.roi_points[1]
            roi_frame = frame[y1:y2, x1:x2].copy()
        else:
            roi_frame = frame.copy()
        
        # 创建轮廓可视化
        contour_frame = roi_frame.copy()
        
        # 绘制检测到的轮廓
        for i, contour in enumerate(contours):
            # 计算轮廓的边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # 绘制轮廓
            cv2.drawContours(contour_frame, [contour], -1, (0, 255, 0), 2)
            
            # 绘制边界矩形
            cv2.rectangle(contour_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # 添加标签
            label = f"Debris {i+1}: {area:.0f}px"
            cv2.putText(contour_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 添加统计信息
        cv2.putText(contour_frame, f"Detected: {len(contours)} debris", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return contour_frame
    
    def _draw_roi_on_original(self, frame):
        """在原图上绘制ROI区域"""
        result_frame = frame.copy()
        
        if self.roi_points:
            # 绘制绿色矩形ROI
            cv2.rectangle(result_frame, self.roi_points[0], self.roi_points[1], (0, 255, 0), 2)
            
            # 添加ROI标签
            label = f"ROI: {self.roi_name}"
            cv2.putText(result_frame, label, 
                       (self.roi_points[0][0], self.roi_points[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return result_frame
    
    def _print_detection_info(self, contours, roi_info):
        """输出检测信息"""
        if contours:
            print(f"🔍 检测到 {len(contours)} 个潜在杂物轮廓")
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                print(f"   - 轮廓 {i+1}: 面积={area:.0f}px, 位置=({x},{y}), 尺寸={w}x{h}")
        else:
            print("🔍 未检测到杂物轮廓")
        
        # 输出前景统计
        if 'foreground_ratio' in roi_info:
            fg_ratio = roi_info['foreground_ratio']
            print(f"📊 前景像素比例: {fg_ratio:.4f}")
    
    def reset_background_model(self):
        """重置背景模型"""
        self.background_subtractor.reset_model()
        print("🔄 背景模型已重置")

def main():
    """主测试函数"""
    print("🚀 启动杂物/异物轮廓检测测试")
    print("=" * 50)
    
    # 视频文件路径
    # video_path = "data/test_videos/trash_in_area/1 (online-video-cutter.com) (1).mp4"
    video_path = "data/test_videos/trash_in_area/14.mp4"
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        print("请确保视频文件路径正确")
        return
    
    # 初始化检测器
    roi_manager = ROIManager()
    debris_detector = DebrisContourDetector(
        min_contour_area=100,      # 最小轮廓面积
        max_contour_area=3000,    # 最大轮廓面积
        aspect_ratio_threshold=8, # 长宽比阈值
        roi_manager=roi_manager
    )
    
    # 设置ROI区域（根据实际视频调整）
    # 这里设置一个示例ROI，你需要根据实际视频内容调整
    # roi_points = [(300, 300), (700, 600)]
    roi_points = [(600, 300), (900, 600)]
    debris_detector.setup_roi(roi_points)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 视频信息: {width}x{height}, {fps:.1f} FPS, 总帧数: {total_frames}")
    print("✅ 系统准备就绪")
    print("🎮 控制说明:")
    print("  - 按 'q' 键退出")
    print("  - 按 'r' 键重置背景模型")
    print("  - 按 'p' 键暂停/继续")
    print("  - 按 's' 键保存当前帧")
    print("=" * 50)
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("📹 视频流结束")
                break
            
            frame_count += 1
            
            # 处理帧
            original_with_roi, roi_foreground_mask, roi_contour_frame, contours = debris_detector.process_frame(
                frame, learning_rate=0.001
            )
            
            # 显示三个可视化结果
            cv2.imshow('1. Original with ROI', original_with_roi)
            cv2.imshow('2. ROI Foreground Mask', roi_foreground_mask)
            cv2.imshow('3. ROI Contour Detection', roi_contour_frame)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            debris_detector.reset_background_model()
        elif key == ord('p'):
            paused = not paused
            print(f"⏸️ {'暂停' if paused else '继续'}")
        elif key == ord('s'):
            # 保存当前帧
            timestamp = int(time.time())
            cv2.imwrite(f"original_with_roi_{timestamp}.jpg", original_with_roi)
            cv2.imwrite(f"roi_foreground_mask_{timestamp}.jpg", roi_foreground_mask)
            cv2.imwrite(f"roi_contour_frame_{timestamp}.jpg", roi_contour_frame)
            print("💾 当前帧已保存")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 测试结束")

if __name__ == "__main__":
    main()