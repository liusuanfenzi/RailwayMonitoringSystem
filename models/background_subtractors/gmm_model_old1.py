# """
# 简化的GMM背景减除模型 - 纯背景建模功能
# """
from pathlib import Path
from typing import Optional, Dict
import cv2
import numpy as np
from utils.video.video_utils import ROIManager

class GMMBackgroundSubtractor:
    """GMM背景减除器 - 专注于背景建模和前景提取"""

    def __init__(self, algorithm: str = 'MOG2', **kwargs):
        """
        初始化背景减除器

        Args:
            algorithm: 'MOG2' 或 'KNN'
            **kwargs: 算法参数
        """
        self.algorithm = algorithm.upper()
        self.roi_manager = ROIManager()

        if self.algorithm == 'MOG2':
            self.history = kwargs.get('history', 500)
            self.var_threshold = kwargs.get('var_threshold', 16)
            self.detect_shadows = kwargs.get('detect_shadows', False)

            self.back_sub = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows
            )
        elif self.algorithm == 'KNN':
            self.history = kwargs.get('history', 500)
            self.dist2_threshold = kwargs.get('dist2_threshold', 400)
            self.detect_shadows = kwargs.get('detect_shadows', False)

            self.back_sub = cv2.createBackgroundSubtractorKNN(
                history=self.history,
                dist2Threshold=self.dist2_threshold,
                detectShadows=self.detect_shadows
            )
        else:
            raise ValueError("算法必须是 'MOG2' 或 'KNN'")
        
        print(f"✅ {self.algorithm}背景减除器初始化成功")

    def setup_rois(self, rois_config: Dict[str, list]):
        """
        设置多个ROI区域
        
        Args:
            rois_config: ROI配置字典 {roi_name: [(x1,y1), (x2,y2)]}
        """
        for roi_name, points in rois_config.items():
            if len(points) != 2:
                raise ValueError(f"ROI {roi_name} 点必须是两个点 [(x1,y1), (x2,y2)]")
            self.roi_manager.add_roi(roi_name, points)
            print(f"🎯 设置ROI区域 {roi_name}: {points}")

    def setup_single_roi(self, points: list, roi_name: str = 'detection_region'):
        """
        设置单个ROI区域

        Args:
            points: 矩形区域点列表 [(x1,y1), (x2,y2)]
            roi_name: ROI区域名称
        """
        if len(points) != 2:
            raise ValueError("ROI点必须是两个点 [(x1,y1), (x2,y2)]")
        self.roi_manager.add_roi(roi_name, points)
        print(f"🎯 设置ROI区域 {roi_name}: {points}")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        预处理帧 - 按照最优顺序处理
        
        Args:
            frame: 输入帧

        Returns:
            预处理后的帧
        """
        if frame is None:
            raise ValueError("输入帧不能为None")

        # Step 1: 转换为灰度图（放在最前面）
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame.copy()

        # Step 2: 应用多个ROI掩膜
        if self.roi_manager.rois:
            # 创建组合ROI掩膜
            mask = np.zeros(gray_frame.shape[:2], dtype=np.uint8)
            
            # 为所有ROI区域创建掩膜
            for points in self.roi_manager.rois.values():
                cv2.rectangle(mask, points[0], points[1], 255, -1)

            # 应用ROI掩膜
            gray_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

        # Step 3: 高斯模糊降噪
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        return blurred_frame
    
    def _postprocess_mask(self, fg_mask: np.ndarray) -> np.ndarray:
        """
        后处理前景掩码 - 按照最优顺序处理
        
        Args:
            fg_mask: 前景掩码

        Returns:
            后处理后的掩码
        """
        # Step 1: 二值化处理
        # 对于MOG2/KNN输出，使用适当阈值去除阴影和弱检测
        _, binary_mask = cv2.threshold(fg_mask, 100, 255, cv2.THRESH_BINARY)

        # Step 2: 形态学开运算去除小噪声
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

        # Step 3: 中值滤波进一步降噪
        filtered_mask = cv2.medianBlur(opened_mask, 5)

        # 可选步骤: 闭运算填充小孔洞
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        return final_mask
    
    def apply(self, frame: np.ndarray, learning_rate: float = 0.005) -> np.ndarray:
        """
        应用背景减除
        
        Args:
            frame: 输入帧
            learning_rate: 学习率

        Returns:
            处理后的前景掩码
        """
        preprocessed_frame = self._preprocess_frame(frame)
        fg_mask = self.back_sub.apply(preprocessed_frame, learningRate=learning_rate)
        processed_mask = self._postprocess_mask(fg_mask)
        return processed_mask

    def apply_with_roi_analysis(self, frame: np.ndarray, learning_rate: float = 0.005) -> dict:
        """
        应用背景减除并分析所有ROI区域

        Args:
            frame: 输入帧
            learning_rate: 学习率

        Returns:
            包含所有ROI区域结果的字典
        """
        fg_mask = self.apply(frame, learning_rate)

        # 计算完整帧统计
        full_foreground_pixels = np.sum(fg_mask > 0)
        full_foreground_ratio = full_foreground_pixels / fg_mask.size

        results = {
            'full_frame': {
                'mask': fg_mask,
                'foreground_pixels': full_foreground_pixels,
                'foreground_ratio': full_foreground_ratio
            }
        }

        # 计算每个ROI区域的统计
        for roi_name in self.roi_manager.rois.keys():
            try:
                roi_mask = self.roi_manager.crop_roi(fg_mask, roi_name)
                
                roi_size = roi_mask.shape[0] * roi_mask.shape[1]
                roi_foreground_pixels = np.sum(roi_mask > 0)
                roi_foreground_ratio = roi_foreground_pixels / roi_size if roi_size > 0 else 0

                results[roi_name] = {
                    'mask': roi_mask,
                    'foreground_pixels': roi_foreground_pixels,
                    'foreground_ratio': roi_foreground_ratio,
                    'roi_size': roi_size
                }
            except Exception as e:
                print(f"⚠️ ROI分析失败 {roi_name}: {e}")

        return results

    def get_background_model(self) -> np.ndarray:
        """获取当前背景模型"""
        return self.back_sub.getBackgroundImage()

    def reset_model(self):
        """重置背景模型"""
        if self.algorithm == 'MOG2':
            self.back_sub = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows
            )
        else:
            self.back_sub = cv2.createBackgroundSubtractorKNN(
                history=self.history,
                dist2Threshold=self.dist2_threshold,
                detectShadows=self.detect_shadows
            )
        print("🔄 背景模型已重置")