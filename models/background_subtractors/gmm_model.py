from typing import Literal
import cv2
import numpy as np
from utils.video.video_utils import ROIManager


class GMMBackgroundSubtractor:
    """GMM背景减除器 - 专注于单个ROI区域的背景建模和前景提取"""

    def __init__(self, algorithm: str = 'MOG2', preprocess_mode: str = 'basic', **kwargs):
        """
        初始化背景减除器

        Args:
            algorithm: 'MOG2' 或 'KNN'
            preprocess_mode: 预处理模式 - 'basic', 'enhance_dark', 或 'none'
            **kwargs: 算法参数
        """
        self.algorithm = algorithm.upper()
        self.preprocess_mode = preprocess_mode
        self.roi_manager = ROIManager()

        # 噪声抑制级别
        self.noise_reduction_level = kwargs.get('noise_reduction', 'medium')

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

        print(f"✅ {self.algorithm}背景减除器初始化成功 - 预处理模式: {self.preprocess_mode}")

    def setup_single_roi(self, points: list, roi_name: str = 'detection_region'):
        """设置单个ROI区域"""
        if len(points) != 2:
            raise ValueError("ROI点必须是两个点 [(x1,y1), (x2,y2)]")
        self.roi_manager.add_roi(roi_name, points)
        print(f"🎯 设置单个ROI区域 {roi_name}: {points}")

    def _basic_preprocess(self, frame: np.ndarray) -> np.ndarray:
        """基础预处理 - 只保留灰度转换和ROI掩码"""
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame.copy()

        if self.roi_manager.rois:
            mask = np.zeros(gray_frame.shape[:2], dtype=np.uint8)
            for points in self.roi_manager.rois.values():
                cv2.rectangle(mask, points[0], points[1], 255, -1)
            gray_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

        return gray_frame

    def _enhance_dark_preprocess(self, frame: np.ndarray) -> np.ndarray:
        """增强暗色区域的预处理 - 专门针对列车暗色区域"""
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

        # 增强暗部处理链
        equalized_frame = cv2.equalizeHist(gray_frame)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_frame = clahe.apply(equalized_frame)
        gamma_corrected = self._gamma_correction(clahe_frame, gamma=1.5)
        blurred_frame = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)

        return blurred_frame

    def _gamma_correction(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """伽马校正"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """根据选择的预处理模式处理帧"""
        if frame is None:
            raise ValueError("输入帧不能为None")

        if self.preprocess_mode == 'none':
            if len(frame.shape) == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame.copy()
        elif self.preprocess_mode == 'basic':
            return self._basic_preprocess(frame)
        elif self.preprocess_mode == 'enhance_dark':
            return self._enhance_dark_preprocess(frame)
        else:
            raise ValueError(f"不支持的预处理模式: {self.preprocess_mode}")

    def _postprocess_mask(self, fg_mask: np.ndarray, noise_reduction_level: str = None) -> np.ndarray:
        """
        增强的后处理前景掩码 - 多级别噪声抑制
        """
        if noise_reduction_level is None:
            noise_reduction_level = self.noise_reduction_level

        # 自适应二值化阈值
        if self.preprocess_mode == 'enhance_dark':
            _, binary_mask = cv2.threshold(
                fg_mask, 150, 255, cv2.THRESH_BINARY)
        else:
            _, binary_mask = cv2.threshold(
                fg_mask, 100, 255, cv2.THRESH_BINARY)

        # 噪声抑制参数配置
        if noise_reduction_level == 'light':
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            median_kernel = 3
            area_threshold = 30
        elif noise_reduction_level == 'medium':
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            median_kernel = 5
            area_threshold = 100
        else:  # 'strong'
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            median_kernel = 7
            area_threshold = 200

        # 形态学处理流程
        opened_mask = cv2.morphologyEx(
            binary_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        closed_mask = cv2.morphologyEx(
            opened_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        final_mask = median_filtered = cv2.medianBlur(
            closed_mask, median_kernel)
        # final_mask = self._remove_small_components(median_filtered, area_threshold)

        return final_mask

    def set_preprocess_mode(self, mode: Literal['none', 'basic', 'enhance_dark']):
        """动态设置预处理模式"""
        self.preprocess_mode = mode
        print(f"🔄 预处理模式已切换为: {mode}")

    def set_noise_reduction_level(self, level: Literal['light', 'medium', 'strong']):
        """动态设置噪声抑制级别"""
        self.noise_reduction_level = level
        print(f"🔄 噪声抑制级别已切换为: {level}")

    def apply(self, frame: np.ndarray, learning_rate: float = 0.005,
              noise_reduction_level: str = None) -> np.ndarray:
        """
        应用背景减除

        Args:
            frame: 输入帧
            learning_rate: 学习率
            noise_reduction_level: 可选，覆盖默认噪声抑制级别
        """
        preprocessed_frame = self._preprocess_frame(frame)
        fg_mask = self.back_sub.apply(
            preprocessed_frame, learningRate=learning_rate)
        processed_mask = self._postprocess_mask(fg_mask, noise_reduction_level)
        return processed_mask

    def apply_with_roi_analysis(self, frame: np.ndarray, learning_rate: float = 0.005) -> dict:
        """
        应用背景减除并分析ROI区域

        Args:
            frame: 输入帧
            learning_rate: 学习率

        Returns:
            包含ROI区域结果的字典
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
            roi_name = list(self.roi_manager.rois.keys())[0]  # 获取唯一的ROI名称
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
