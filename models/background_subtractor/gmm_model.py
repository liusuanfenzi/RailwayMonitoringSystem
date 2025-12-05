# models/jetson_gmm_model.py
import cv2
import numpy as np
from typing import Literal, Dict
from utils.utils import ROIManager, PerformanceMonitor

class GMMBackgroundSubtractor:
    """Jetson优化的GMM背景减除器 - 支持多种预处理模式"""

    def __init__(self, algorithm: str = 'MOG2', preprocess_mode: str = 'basic', **kwargs):
        """
        Jetson优化初始化

        Args:
            algorithm: 'MOG2' 或 'KNN'
            preprocess_mode: 预处理模式 - 'basic', 'enhance_dark'
            **kwargs: 算法参数
        """
        self.algorithm = algorithm.upper()
        self.preprocess_mode = preprocess_mode
        self.roi_manager = ROIManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Jetson优化参数
        if self.algorithm == 'MOG2':
            self.history = kwargs.get('history', 200)
            self.var_threshold = kwargs.get('var_threshold', 16)
            self.detect_shadows = kwargs.get('detect_shadows', False)

            self.back_sub = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows
            )
        elif self.algorithm == 'KNN':
            self.history = kwargs.get('history', 200)
            self.dist2_threshold = kwargs.get('dist2_threshold', 400)
            self.detect_shadows = kwargs.get('detect_shadows', False)

            self.back_sub = cv2.createBackgroundSubtractorKNN(
                history=self.history,
                dist2Threshold=self.dist2_threshold,
                detectShadows=self.detect_shadows
            )
        else:
            raise ValueError("算法必须是 'MOG2' 或 'KNN'")

        print(f"✅ Jetson {self.algorithm}背景减除器初始化完成 - 模式: {self.preprocess_mode}")

    def setup_single_roi(self, points: list, roi_name: str = 'detection_region'):
        """设置单个ROI区域"""
        self.roi_manager.add_roi(roi_name, points)

    def _basic_preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        基础预处理 - 灰度化 + ROI掩码
        性能最优，适合明亮环境
        """
        # 转换为灰度图
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame.copy()

        # 应用ROI掩膜
        if self.roi_manager.rois:
            mask = np.zeros(gray_frame.shape[:2], dtype=np.uint8)
            for points in self.roi_manager.rois.values():
                cv2.rectangle(mask, points[0], points[1], 255, -1)
            gray_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

        # 轻微高斯模糊减少噪声
        blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        return blurred_frame

    def _enhance_dark_preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        增强暗部预处理 - 针对列车暗色区域优化
        平衡效果和性能，适合暗光环境
        """
        # 转换为灰度图
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame.copy()

        # 应用ROI掩膜
        if self.roi_manager.rois:
            mask = np.zeros(gray_frame.shape[:2], dtype=np.uint8)
            for points in self.roi_manager.rois.values():
                cv2.rectangle(mask, points[0], points[1], 255, -1)
            gray_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

        # Jetson优化的暗部增强处理链
        # 1. 直方图均衡化 - 增强对比度
        equalized_frame = cv2.equalizeHist(gray_frame)
        
        # 2. 限制对比度的自适应直方图均衡化 - 针对暗部优化
        # 使用较小的网格和适中的clipLimit以平衡性能
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        clahe_frame = clahe.apply(equalized_frame)
        
        # 3. 伽马校正 - 增强暗部细节 (优化版本)
        gamma_corrected = self._optimized_gamma_correction(clahe_frame, gamma=1.3)
        
        # 4. 高斯模糊减少噪声
        blurred_frame = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)

        return blurred_frame

    def _optimized_gamma_correction(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Jetson优化的伽马校正
        使用查找表避免重复计算
        """
        # 构建伽马校正查找表
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in range(256)]).astype("uint8")
        
        # 应用查找表
        return cv2.LUT(image, table)

    def _jetson_preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Jetson优化的预处理 - 支持多种模式
        """
        if frame is None:
            raise ValueError("输入帧不能为None")

        if self.preprocess_mode == 'basic':
            return self._basic_preprocess(frame)
        elif self.preprocess_mode == 'enhance_dark':
            return self._enhance_dark_preprocess(frame)
        else:
            raise ValueError(f"不支持的预处理模式: {self.preprocess_mode}")

    def _jetson_postprocess(self, fg_mask: np.ndarray) -> np.ndarray:
        """
        Jetson优化的后处理
        根据预处理模式调整参数
        """
        # 自适应二值化阈值
        if self.preprocess_mode == 'enhance_dark':
            # 增强暗部模式下使用更高阈值
            _, binary_mask = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)
        else:
            _, binary_mask = cv2.threshold(fg_mask, 100, 255, cv2.THRESH_BINARY)

        # Jetson优化：根据模式调整形态学操作强度
        if self.preprocess_mode == 'enhance_dark':
            # 增强模式下使用更强的噪声抑制
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        else:
            # 基础模式下使用较轻的噪声抑制
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # 形态学处理
        opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        final_mask = cv2.medianBlur(closed_mask, 3)

        return final_mask

    def set_preprocess_mode(self, mode: Literal['basic', 'enhance_dark']):
        """动态设置预处理模式"""
        if mode not in ['basic', 'enhance_dark']:
            raise ValueError("预处理模式必须是 'basic' 或 'enhance_dark'")
        
        self.preprocess_mode = mode
        print(f"🔄 预处理模式已切换为: {mode}")

    def apply(self, frame: np.ndarray, learning_rate: float = 0.005) -> np.ndarray:
        """
        应用背景减除 - Jetson优化版本
        """
        start_time = self.performance_monitor.start_timing()
        
        preprocessed_frame = self._jetson_preprocess(frame)
        fg_mask = self.back_sub.apply(preprocessed_frame, learningRate=learning_rate)
        processed_mask = self._jetson_postprocess(fg_mask)
        
        self.performance_monitor.end_timing(start_time, f"背景减除[{self.preprocess_mode}]")
        return processed_mask

    def apply_with_roi_analysis(self, frame: np.ndarray, learning_rate: float = 0.005) -> Dict:
        """
        应用背景减除并分析ROI区域 - Jetson优化版本
        """
        fg_mask = self.apply(frame, learning_rate)

        # 计算完整帧统计
        full_foreground_pixels = np.sum(fg_mask > 0)
        full_foreground_ratio = full_foreground_pixels / fg_mask.size

        results = {
            'full_frame': {
                'mask': fg_mask,
                'foreground_pixels': full_foreground_pixels,
                'foreground_ratio': full_foreground_ratio,
                'preprocess_mode': self.preprocess_mode
            }
        }

        # 计算ROI区域的统计
        if self.roi_manager.rois:
            roi_name = list(self.roi_manager.rois.keys())[0]
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

    def get_performance_stats(self):
        """获取性能统计"""
        return self.performance_monitor.get_performance_stats()

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