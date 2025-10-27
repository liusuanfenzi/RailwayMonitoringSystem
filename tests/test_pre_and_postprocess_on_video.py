# #!/usr/bin/env python3
# """
# 背景减除处理策略对比测试
# 比较三种策略在列车进站视频上的效果：
# 1. 不加预处理与后处理
# 2. 只加后处理
# 3. 加预处理与后处理
# """

# import cv2
# import numpy as np
# from pathlib import Path

# class GMMBackgroundSubtractor:
#     """GMM背景减除器 - 支持不同处理策略"""
    
#     def __init__(self, algorithm: str = 'MOG2', **kwargs):
#         self.algorithm = algorithm.upper()
#         self.roi_manager = ROIManager()

#         if self.algorithm == 'MOG2':
#             self.back_sub = cv2.createBackgroundSubtractorMOG2(
#                 history=kwargs.get('history', 500),
#                 varThreshold=kwargs.get('varThreshold', 16),
#                 detectShadows=kwargs.get('detectShadows', False)
#             )
#         else:
#             self.back_sub = cv2.createBackgroundSubtractorKNN(
#                 history=kwargs.get('history', 500),
#                 dist2Threshold=kwargs.get('dist2Threshold', 400),
#                 detectShadows=kwargs.get('detectShadows', False)
#             )
        
#         print(f"✅ {self.algorithm}背景减除器初始化成功")

#     def setup_single_roi(self, points: list, roi_name: str = 'detection_region'):
#         """设置单个ROI区域"""
#         if len(points) != 2:
#             raise ValueError("ROI点必须是两个点 [(x1,y1), (x2,y2)]")
#         self.roi_manager.add_roi(roi_name, points)
#         print(f"🎯 设置ROI区域 {roi_name}: {points}")

#     def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
#         """预处理帧 - 降低去噪强度"""
#         if frame is None:
#             raise ValueError("输入帧不能为None")

#         # 转换为灰度图
#         if len(frame.shape) == 3:
#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         else:
#             gray_frame = frame.copy()

#         # 应用ROI掩膜
#         if self.roi_manager.rois:
#             mask = np.zeros(gray_frame.shape[:2], dtype=np.uint8)
#             for points in self.roi_manager.rois.values():
#                 cv2.rectangle(mask, points[0], points[1], 255, -1)
#             gray_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

#         # 轻微高斯模糊降噪
#         blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0.5)
#         return blurred_frame

#     def _postprocess_mask(self, fg_mask: np.ndarray) -> np.ndarray:
#         """后处理前景掩码 - 降低去噪强度"""
#         # 降低二值化阈值
#         _, binary_mask = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)

#         # 减少形态学开运算的强度
#         kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

#         # 使用更小的中值滤波核
#         filtered_mask = cv2.medianBlur(opened_mask, 3)
#         return filtered_mask

#     def apply_no_processing(self, frame: np.ndarray, learning_rate: float = 0.005) -> dict:
#         """
#         策略1: 不加预处理与后处理
#         """
#         # 直接应用背景减除，不做任何处理
#         fg_mask = self.back_sub.apply(frame, learningRate=learning_rate)
        
#         # 计算ROI区域统计
#         return self._analyze_results(frame, fg_mask, "no_preprocess + no_postprocess")

#     def apply_postprocessing_only(self, frame: np.ndarray, learning_rate: float = 0.005) -> dict:
#         """
#         策略2: 只加后处理
#         """
#         # 直接应用背景减除
#         fg_mask = self.back_sub.apply(frame, learningRate=learning_rate)
        
#         # 只进行后处理
#         processed_mask = self._postprocess_mask(fg_mask)
        
#         return self._analyze_results(frame, processed_mask, "no_preprocess + postprocess")

#     def apply_full_processing(self, frame: np.ndarray, learning_rate: float = 0.005) -> dict:
#         """
#         策略3: 加预处理与后处理
#         """
#         # 进行预处理
#         preprocessed_frame = self._preprocess_frame(frame)
        
#         # 应用背景减除
#         fg_mask = self.back_sub.apply(preprocessed_frame, learningRate=learning_rate)
        
#         # 进行后处理
#         processed_mask = self._postprocess_mask(fg_mask)
        
#         return self._analyze_results(frame, processed_mask, "npreproces + postprocess")

#     def _analyze_results(self, original_frame: np.ndarray, fg_mask: np.ndarray, strategy_name: str) -> dict:
#         """分析结果并返回统计信息"""
#         # 计算完整帧统计
#         full_foreground_pixels = np.sum(fg_mask > 0)
#         full_foreground_ratio = full_foreground_pixels / fg_mask.size

#         results = {
#             'strategy': strategy_name,
#             'full_frame': {
#                 'mask': fg_mask,
#                 'foreground_pixels': full_foreground_pixels,
#                 'foreground_ratio': full_foreground_ratio
#             },
#             'roi_data': {}
#         }

#         # 计算ROI区域的统计
#         if self.roi_manager.rois:
#             roi_name = list(self.roi_manager.rois.keys())[0]
#             try:
#                 roi_mask = self.roi_manager.crop_roi(fg_mask, roi_name)
#                 roi_original = self.roi_manager.crop_roi(original_frame, roi_name)
                
#                 roi_size = roi_mask.shape[0] * roi_mask.shape[1]
#                 roi_foreground_pixels = np.sum(roi_mask > 0)
#                 roi_foreground_ratio = roi_foreground_pixels / roi_size if roi_size > 0 else 0

#                 results['roi_data'] = {
#                     'mask': roi_mask,
#                     'original': roi_original,
#                     'foreground_pixels': roi_foreground_pixels,
#                     'foreground_ratio': roi_foreground_ratio,
#                     'roi_size': roi_size,
#                     'roi_name': roi_name
#                 }
#             except Exception as e:
#                 print(f"⚠️ ROI分析失败 {roi_name}: {e}")

#         return results

# class ROIManager:
#     """ROI管理器"""
    
#     def __init__(self):
#         self.rois = {}
    
#     def add_roi(self, roi_name: str, points: list):
#         self.rois = {roi_name: points}
    
#     def crop_roi(self, image: np.ndarray, roi_name: str) -> np.ndarray:
#         if roi_name not in self.rois:
#             raise ValueError(f"ROI {roi_name} 不存在")
        
#         points = self.rois[roi_name]
#         x1, y1 = points[0]
#         x2, y2 = points[1]
        
#         h, w = image.shape[:2]
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(w, x2), min(h, y2)
        
#         return image[y1:y2, x1:x2]

# def create_original_display_frame(original_frame: np.ndarray, roi_points: list) -> np.ndarray:
#     """创建原图显示帧，包含ROI区域标注"""
#     display_frame = original_frame.copy()
    
#     # 在原图上绘制ROI区域
#     if roi_points and len(roi_points) == 2:
#         cv2.rectangle(display_frame, roi_points[0], roi_points[1], (0, 255, 0), 2)
        
#         # 添加ROI标签
#         label = "Detection ROI"
#         text_x = roi_points[0][0]
#         text_y = roi_points[0][1] - 10
#         if text_y < 20:
#             text_y = roi_points[0][1] + 25
        
#         cv2.putText(display_frame, label, (text_x, text_y),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
#     # 添加窗口标题
#     cv2.putText(display_frame, "Original Frame with ROI", (10, 30),
#                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
#     return display_frame

# def create_roi_mask_display_frame(results: dict, strategy_name: str) -> np.ndarray:
#     """创建ROI掩码显示帧，只显示ROI区域的前景掩码"""
#     # 创建显示画布
#     display_height = 400
#     display_width = 600
    
#     # 创建黑色背景
#     display_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    
#     # 获取ROI数据
#     roi_data = results.get('roi_data', {})
    
#     if roi_data and 'mask' in roi_data:
#         roi_mask = roi_data['mask']
#         foreground_ratio = roi_data.get('foreground_ratio', 0)
        
#         # 调整掩码大小以适合显示窗口
#         if len(roi_mask.shape) == 2:
#             # 灰度图转彩色
#             roi_mask_color = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
#         else:
#             roi_mask_color = roi_mask
        
#         # 计算缩放比例，保持宽高比
#         h, w = roi_mask_color.shape[:2]
#         scale = min(500 / w, 300 / h)  # 最大显示尺寸 500x300
#         new_w = int(w * scale)
#         new_h = int(h * scale)
        
#         mask_resized = cv2.resize(roi_mask_color, (new_w, new_h))
        
#         # 居中显示掩码
#         x_offset = (display_width - new_w) // 2
#         y_offset = (display_height - new_h) // 2
#         display_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = mask_resized
        
#         # 添加策略名称和统计信息
#         cv2.putText(display_frame, f"Strategy: {strategy_name}", 
#                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
#         cv2.putText(display_frame, f"ROI FG Ratio: {foreground_ratio:.4f}", 
#                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         # 添加完整帧统计
#         full_frame_data = results.get('full_frame', {})
#         full_ratio = full_frame_data.get('foreground_ratio', 0)
#         cv2.putText(display_frame, f"Full Frame FG Ratio: {full_ratio:.4f}", 
#                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         # 显示ROI尺寸信息
#         roi_size = roi_data.get('roi_size', 0)
#         cv2.putText(display_frame, f"ROI Size: {roi_size} pixels", 
#                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
#     else:
#         # 如果没有ROI数据，显示提示信息
#         cv2.putText(display_frame, f"Strategy: {strategy_name}", 
#                    (display_width//2-100, display_height//2-30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#         cv2.putText(display_frame, "No ROI Data Available", 
#                    (display_width//2-120, display_height//2+30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
#     # 添加边框
#     cv2.rectangle(display_frame, (0, 0), (display_width-1, display_height-1), 
#                  (100, 100, 100), 2)
    
#     return display_frame

# def main():
#     """主测试函数"""
#     video_path = "data/test_videos/train_enter_station.mp4"
    
#     # 检查视频文件是否存在
#     if not Path(video_path).exists():
#         print(f"❌ 视频文件不存在: {video_path}")
#         print("请确保视频文件路径正确")
#         return
    
#     # 定义ROI区域（根据列车进站的典型区域调整）
#     roi_points = [(200, 200), (600, 700)]  # [(x1,y1), (x2,y2)]
    
#     # 初始化三个背景减除器（使用相同的参数确保公平比较）
#     bg_subtractors = {
#         'no_processing': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16),
#         'post_only': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16),
#         #'full_processing': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16)
#     }
    
#     # 为所有减除器设置相同的ROI
#     for bg_sub in bg_subtractors.values():
#         bg_sub.setup_single_roi(roi_points, 'train_detection_roi')
    
#     # 打开视频文件
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"❌ 无法打开视频文件: {video_path}")
#         return
    
#     print("🚀 开始背景减除策略对比测试")
#     print("📊 比较三种处理策略:")
#     print("   1. 无预处理 + 无后处理")
#     print("   2. 无预处理 + 有后处理") 
#     #print("   3. 有预处理 + 有后处理")
#     print("🎯 按 'q' 退出，按 'p' 暂停/继续，按 'r' 重置背景模型")
    
#     paused = False
#     frame_count = 0
    
#     try:
#         while True:
#             if not paused:
#                 ret, frame = cap.read()
#                 if not ret:
#                     print("✅ 视频播放完毕")
#                     break
                
#                 frame_count += 1
                
#                 # 应用三种处理策略
#                 results = {
#                     'no_processing': bg_subtractors['no_processing'].apply_no_processing(frame),
#                     'post_only': bg_subtractors['post_only'].apply_postprocessing_only(frame),
#                     #'full_processing': bg_subtractors['full_processing'].apply_full_processing(frame)
#                 }
                
#                 # 创建并显示原图窗口（带ROI标注）
#                 original_display = create_original_display_frame(frame, roi_points)
#                 cv2.imshow('1. Original Frame with ROI', original_display)
                
#                 # 创建并显示三个策略的ROI掩码窗口
#                 strategy_displays = {
#                     'no_processing': create_roi_mask_display_frame(
#                         results['no_processing'], "无预处理+无后处理"
#                     ),
#                     'post_only': create_roi_mask_display_frame(
#                         results['post_only'], "无预处理+有后处理"
#                     ),
#                     # 'full_processing': create_roi_mask_display_frame(
#                     #     results['full_processing'], "有预处理+有后处理"
#                     # )
#                 }
                
#                 cv2.imshow('2. Strategy: No Pre+No Post', strategy_displays['no_processing'])
#                 cv2.imshow('3. Strategy: No Pre+With Post', strategy_displays['post_only'])
#                 #cv2.imshow('4. Strategy: With Pre+With Post', strategy_displays['full_processing'])
                
#                 # 每50帧输出一次统计信息
#                 if frame_count % 50 == 0:
#                     print(f"\n📈 帧 {frame_count} 统计:")
#                     for strategy, result in results.items():
#                         full_ratio = result['full_frame']['foreground_ratio']
#                         roi_ratio = result.get('roi_data', {}).get('foreground_ratio', 0)
#                         print(f"   {result['strategy']}:")
#                         print(f"      Full Frame FG Ratio: {full_ratio:.4f}")
#                         print(f"      ROI FG Ratio: {roi_ratio:.4f}")
            
#             # 键盘控制
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord('p'):
#                 paused = not paused
#                 print(f"{'⏸️ 暂停' if paused else '▶️ 继续'}")
#             elif key == ord('r'):  # 重置背景模型
#                 for name, bg_sub in bg_subtractors.items():
#                     bg_sub.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)
#                 print("🔄 所有背景模型已重置")
    
#     except Exception as e:
#         print(f"❌ 程序执行出错: {e}")
#         import traceback
#         traceback.print_exc()
    
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
    
#     print(f"\n✅ 测试完成")
#     print(f"📊 总共处理帧数: {frame_count}")
#     print("🎯 分析建议:")
#     print("   - 窗口1: 原图+ROI区域标注")
#     print("   - 窗口2: 无预处理+无后处理效果")
#     print("   - 窗口3: 无预处理+有后处理效果") 
#     #print("   - 窗口4: 有预处理+有后处理效果")
#     print("   - 观察三种策略的前景检测完整度和噪声抑制效果")

# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# """
# 四种处理策略对比测试
# 1. 完全不处理
# 2. 只加预处理
# 3. 只加后处理
# 4. 预处理+后处理
# """

# import cv2
# import numpy as np
# from pathlib import Path

# class GMMBackgroundSubtractor:
#     """GMM背景减除器 - 支持四种处理策略"""
    
#     def __init__(self, algorithm: str = 'MOG2', **kwargs):
#         self.algorithm = algorithm.upper()
#         self.roi_manager = ROIManager()

#         if self.algorithm == 'MOG2':
#             self.back_sub = cv2.createBackgroundSubtractorMOG2(
#                 history=kwargs.get('history', 500),
#                 varThreshold=kwargs.get('varThreshold', 16),
#                 detectShadows=kwargs.get('detectShadows', False)
#             )
#         else:
#             self.back_sub = cv2.createBackgroundSubtractorKNN(
#                 history=kwargs.get('history', 500),
#                 dist2Threshold=kwargs.get('dist2Threshold', 400),
#                 detectShadows=kwargs.get('detectShadows', False)
#             )
        
#         print(f"✅ {self.algorithm}背景减除器初始化成功")

#     def setup_single_roi(self, points: list, roi_name: str = 'detection_region'):
#         """设置单个ROI区域"""
#         if len(points) != 2:
#             raise ValueError("ROI点必须是两个点 [(x1,y1), (x2,y2)]")
#         self.roi_manager.add_roi(roi_name, points)
#         print(f"🎯 设置ROI区域 {roi_name}: {points}")

#     def _enhance_dark_regions(self, frame: np.ndarray) -> np.ndarray:
#         """增强暗部区域"""
#         if len(frame.shape) == 3:
#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         else:
#             gray_frame = frame.copy()
        
#         # 应用伽马校正增强暗部
#         gamma = 0.8  # 小于1的值增强暗部
#         inv_gamma = 1.0 / gamma
#         table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#         enhanced_frame = cv2.LUT(gray_frame, table)
        
#         return enhanced_frame

#     def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
#         """预处理帧 - 包含暗部增强和降噪"""
#         if frame is None:
#             raise ValueError("输入帧不能为None")

#         # Step 1: 增强暗部区域
#         enhanced_frame = self._enhance_dark_regions(frame)

#         # Step 2: 应用ROI掩膜
#         if self.roi_manager.rois:
#             mask = np.zeros(enhanced_frame.shape[:2], dtype=np.uint8)
#             for points in self.roi_manager.rois.values():
#                 cv2.rectangle(mask, points[0], points[1], 255, -1)
#             enhanced_frame = cv2.bitwise_and(enhanced_frame, enhanced_frame, mask=mask)

#         # Step 3: 轻微高斯模糊降噪
#         blurred_frame = cv2.GaussianBlur(enhanced_frame, (3, 3), 0.5)

#         return blurred_frame

#     def _postprocess_mask(self, fg_mask: np.ndarray) -> np.ndarray:
#         """后处理前景掩码 - 降低去噪强度"""
#         # Step 1: 降低二值化阈值
#         _, binary_mask = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)

#         # Step 2: 减少形态学开运算的强度
#         kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
#         opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

#         # Step 3: 使用更小的中值滤波核
#         filtered_mask = cv2.medianBlur(opened_mask, 3)

#         return filtered_mask

#     def apply_no_processing(self, frame: np.ndarray, learning_rate: float = 0.005) -> np.ndarray:
#         """
#         策略1: 完全不处理
#         """
#         # 直接应用背景减除，不做任何处理
#         fg_mask = self.back_sub.apply(frame, learningRate=learning_rate)
#         return fg_mask

#     def apply_preprocessing_only(self, frame: np.ndarray, learning_rate: float = 0.005) -> np.ndarray:
#         """
#         策略2: 只加预处理
#         """
#         # 进行预处理（包含暗部增强）
#         preprocessed_frame = self._preprocess_frame(frame)
        
#         # 应用背景减除
#         fg_mask = self.back_sub.apply(preprocessed_frame, learningRate=learning_rate)
        
#         return fg_mask

#     def apply_postprocessing_only(self, frame: np.ndarray, learning_rate: float = 0.005) -> np.ndarray:
#         """
#         策略3: 只加后处理
#         """
#         # 直接应用背景减除
#         fg_mask = self.back_sub.apply(frame, learningRate=learning_rate)
        
#         # 只进行后处理
#         processed_mask = self._postprocess_mask(fg_mask)
        
#         return processed_mask

#     def apply_full_processing(self, frame: np.ndarray, learning_rate: float = 0.005) -> np.ndarray:
#         """
#         策略4: 预处理+后处理
#         """
#         # 进行预处理（包含暗部增强）
#         preprocessed_frame = self._preprocess_frame(frame)
        
#         # 应用背景减除
#         fg_mask = self.back_sub.apply(preprocessed_frame, learningRate=learning_rate)
        
#         # 进行后处理
#         processed_mask = self._postprocess_mask(fg_mask)
        
#         return processed_mask

# class ROIManager:
#     """ROI管理器"""
    
#     def __init__(self):
#         self.rois = {}
    
#     def add_roi(self, roi_name: str, points: list):
#         self.rois = {roi_name: points}
    
#     def crop_roi(self, image: np.ndarray, roi_name: str) -> np.ndarray:
#         if roi_name not in self.rois:
#             raise ValueError(f"ROI {roi_name} 不存在")
        
#         points = self.rois[roi_name]
#         x1, y1 = points[0]
#         x2, y2 = points[1]
        
#         h, w = image.shape[:2]
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(w, x2), min(h, y2)
        
#         return image[y1:y2, x1:x2]

# def get_frame_300(video_path: str) -> np.ndarray:
#     """获取视频的第300帧"""
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError(f"无法打开视频文件: {video_path}")
    
#     # 设置到第300帧
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 299)  # 0-based index
    
#     ret, frame = cap.read()
#     cap.release()
    
#     if not ret:
#         raise ValueError("无法读取第300帧")
    
#     return frame

# def test_four_strategies_on_frame_300():
#     """在第300帧上测试四种处理策略"""
#     video_path = "data/test_videos/train_enter_station.mp4"
    
#     # 检查视频文件是否存在
#     if not Path(video_path).exists():
#         print(f"❌ 视频文件不存在: {video_path}")
#         return
    
#     # 定义ROI区域
#     roi_points = [(200, 200), (600, 700)]
    
#     # 初始化四个背景减除器（使用相同的参数确保公平比较）
#     bg_subtractors = {
#         'no_processing': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16),
#         'pre_only': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16),
#         'post_only': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16),
#         'full_processing': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16)
#     }
    
#     # 为所有减除器设置相同的ROI
#     for bg_sub in bg_subtractors.values():
#         bg_sub.setup_single_roi(roi_points, 'train_detection_roi')
    
#     print("🚀 开始四种处理策略对比测试")
#     print("📊 比较四种处理策略:")
#     print("   1. 完全不处理")
#     print("   2. 只加预处理（暗部增强+降噪）")
#     print("   3. 只加后处理（形态学滤波）")
#     print("   4. 预处理+后处理")
    
#     try:
#         # 获取第300帧
#         print("\n🎯 正在读取第300帧...")
#         frame_300 = get_frame_300(video_path)
#         print(f"✅ 成功读取第300帧，尺寸: {frame_300.shape}")
        
#         # 应用四种处理策略
#         print("\n🔧 应用四种处理策略...")
#         masks = {
#             '1. No Processing': bg_subtractors['no_processing'].apply_no_processing(frame_300),
#             '2. Preprocessing Only': bg_subtractors['pre_only'].apply_preprocessing_only(frame_300),
#             '3. Postprocessing Only': bg_subtractors['post_only'].apply_postprocessing_only(frame_300),
#             '4. Full Processing': bg_subtractors['full_processing'].apply_full_processing(frame_300)
#         }
        
#         # 计算每种策略的前景比例
#         stats = {}
#         for strategy_name, mask in masks.items():
#             foreground_pixels = np.sum(mask > 0)
#             total_pixels = mask.shape[0] * mask.shape[1]
#             foreground_ratio = foreground_pixels / total_pixels
#             stats[strategy_name] = foreground_ratio
        
#         # 显示原图和四种策略的结果
#         print("\n🖼️ 显示结果窗口...")
        
#         # 显示原图（带ROI标注）
#         original_with_roi = frame_300.copy()
#         cv2.rectangle(original_with_roi, roi_points[0], roi_points[1], (0, 255, 0), 2)
#         cv2.putText(original_with_roi, "Original Frame (Frame 300)", (10, 30),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         cv2.imshow('0. Original Frame (Frame 300)', original_with_roi)
        
#         # 显示四种策略的前景掩码
#         for i, (strategy_name, mask) in enumerate(masks.items(), 1):
#             # 将灰度掩码转换为彩色以便显示
#             if len(mask.shape) == 2:
#                 mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#             else:
#                 mask_display = mask.copy()
            
#             # 添加标题和统计信息
#             foreground_ratio = stats[strategy_name]
#             cv2.putText(mask_display, f"{strategy_name}", (10, 30),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#             cv2.putText(mask_display, f"FG Ratio: {foreground_ratio:.4f}", (10, 60),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
#             # 显示窗口
#             cv2.imshow(f'{i}. {strategy_name}', mask_display)
        
#         # 输出统计信息
#         print("\n📈 第300帧处理结果统计:")
#         for strategy_name, ratio in stats.items():
#             print(f"   {strategy_name}: {ratio:.4f}")
        
#         print("\n🎯 按任意键关闭窗口...")
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        
#         # 保存结果图像（可选）
#         save_results = input("\n💾 是否保存结果图像？(y/n): ").lower().strip()
#         if save_results == 'y':
#             # 创建保存目录
#             save_dir = Path("test_results")
#             save_dir.mkdir(exist_ok=True)
            
#             # 保存原图
#             cv2.imwrite(str(save_dir / "original_frame_300.jpg"), original_with_roi)
            
#             # 保存各策略结果
#             for strategy_name, mask in masks.items():
#                 filename = f"strategy_{strategy_name.replace(' ', '_').replace('.', '')}.jpg"
#                 cv2.imwrite(str(save_dir / filename), mask)
            
#             print(f"✅ 结果已保存到 {save_dir} 目录")
        
#     except Exception as e:
#         print(f"❌ 测试过程中出错: {e}")
#         import traceback
#         traceback.print_exc()

# def test_four_strategies_on_video():
#     """在完整视频上测试四种处理策略"""
#     video_path = "data/test_videos/train_enter_station.mp4"
    
#     if not Path(video_path).exists():
#         print(f"❌ 视频文件不存在: {video_path}")
#         return
    
#     roi_points = [(200, 200), (600, 700)]
    
#     # 初始化四个背景减除器
#     bg_subtractors = {
#         'no_processing': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16),
#         'pre_only': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16),
#         'post_only': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16),
#         'full_processing': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16)
#     }
    
#     for bg_sub in bg_subtractors.values():
#         bg_sub.setup_single_roi(roi_points, 'train_detection_roi')
    
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"❌ 无法打开视频文件: {video_path}")
#         return
    
#     print("🚀 开始视频流四种处理策略对比测试")
#     print("🎯 按 'q' 退出，按 'p' 暂停/继续，按 'r' 重置背景模型")
    
#     paused = False
#     frame_count = 0
    
#     try:
#         while True:
#             if not paused:
#                 ret, frame = cap.read()
#                 if not ret:
#                     print("✅ 视频播放完毕")
#                     break
                
#                 frame_count += 1
                
#                 # 应用四种处理策略
#                 masks = {
#                     '1. No Processing': bg_subtractors['no_processing'].apply_no_processing(frame),
#                     '2. Preprocessing Only': bg_subtractors['pre_only'].apply_preprocessing_only(frame),
#                     '3. Postprocessing Only': bg_subtractors['post_only'].apply_postprocessing_only(frame),
#                     '4. Full Processing': bg_subtractors['full_processing'].apply_full_processing(frame)
#                 }
                
#                 # 显示原图
#                 original_with_roi = frame.copy()
#                 cv2.rectangle(original_with_roi, roi_points[0], roi_points[1], (0, 255, 0), 2)
#                 cv2.putText(original_with_roi, f"Original Frame - Frame {frame_count}", (10, 30),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#                 cv2.imshow('0. Original Frame', original_with_roi)
                
#                 # 显示四种策略的前景掩码
#                 for i, (strategy_name, mask) in enumerate(masks.items(), 1):
#                     if len(mask.shape) == 2:
#                         mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#                     else:
#                         mask_display = mask.copy()
                    
#                     # 计算前景比例
#                     foreground_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
                    
#                     cv2.putText(mask_display, f"{strategy_name}", (10, 30),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#                     cv2.putText(mask_display, f"FG Ratio: {foreground_ratio:.4f}", (10, 60),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
#                     cv2.imshow(f'{i}. {strategy_name}', mask_display)
                
#                 # 每100帧输出统计信息
#                 if frame_count % 100 == 0:
#                     print(f"\n📈 帧 {frame_count} 统计:")
#                     for strategy_name, mask in masks.items():
#                         foreground_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
#                         print(f"   {strategy_name}: {foreground_ratio:.4f}")
            
#             # 键盘控制
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord('p'):
#                 paused = not paused
#                 print(f"{'⏸️ 暂停' if paused else '▶️ 继续'}")
#             elif key == ord('r'):
#                 for bg_sub in bg_subtractors.values():
#                     bg_sub.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)
#                 print("🔄 所有背景模型已重置")
    
#     except Exception as e:
#         print(f"❌ 测试过程中出错: {e}")
#         import traceback
#         traceback.print_exc()
    
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
    
#     print(f"\n✅ 视频测试完成，总共处理 {frame_count} 帧")

# if __name__ == "__main__":
    # print("请选择测试模式:")
    # print("1. 测试第300帧")
    # print("2. 测试完整视频")
    
    # choice = input("请输入选择 (1 或 2): ").strip()
    
    # if choice == "1":
    #     test_four_strategies_on_frame_300()
    # elif choice == "2":
    #     test_four_strategies_on_video()
    # else:
    #     print("❌ 无效选择")

#!/usr/bin/env python3
"""
四种处理策略对比测试
1. 无预处理 + 无后处理
2. 只加ROI掩码（预处理）
3. 只加后处理
4. ROI掩码 + 后处理
"""

import cv2
import numpy as np
from pathlib import Path

class ROIManager:
    """ROI管理器"""
    
    def __init__(self):
        self.rois = {}
    
    def add_roi(self, roi_name: str, points: list):
        self.rois = {roi_name: points}
    
    def crop_roi(self, image: np.ndarray, roi_name: str) -> np.ndarray:
        if roi_name not in self.rois:
            raise ValueError(f"ROI {roi_name} 不存在")
        
        points = self.rois[roi_name]
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        return image[y1:y2, x1:x2]

class GMMBackgroundSubtractor:
    """GMM背景减除器 - 支持四种处理策略"""
    
    def __init__(self, algorithm: str = 'MOG2', **kwargs):
        self.algorithm = algorithm.upper()
        self.roi_manager = ROIManager()

        if self.algorithm == 'MOG2':
            self.back_sub = cv2.createBackgroundSubtractorMOG2(
                history=kwargs.get('history', 500),
                varThreshold=kwargs.get('varThreshold', 16),
                detectShadows=kwargs.get('detectShadows', False)
            )
        else:
            self.back_sub = cv2.createBackgroundSubtractorKNN(
                history=kwargs.get('history', 500),
                dist2Threshold=kwargs.get('dist2Threshold', 400),
                detectShadows=kwargs.get('detectShadows', False)
            )
        
        print(f"✅ {self.algorithm}背景减除器初始化成功")

    def setup_single_roi(self, points: list, roi_name: str = 'detection_region'):
        """设置单个ROI区域"""
        if len(points) != 2:
            raise ValueError("ROI点必须是两个点 [(x1,y1), (x2,y2)]")
        self.roi_manager.add_roi(roi_name, points)
        print(f"🎯 设置ROI区域 {roi_name}: {points}")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        预处理帧 - 只保留ROI掩码应用
        """
        if frame is None:
            raise ValueError("输入帧不能为None")

        # Step 1: 转换为灰度图
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame.copy()

        # Step 2: 应用ROI掩膜
        if self.roi_manager.rois:
            mask = np.zeros(gray_frame.shape[:2], dtype=np.uint8)
            for points in self.roi_manager.rois.values():
                cv2.rectangle(mask, points[0], points[1], 255, -1)
            gray_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

        return gray_frame
    
    def _postprocess_mask(self, fg_mask: np.ndarray) -> np.ndarray:
        """
        后处理前景掩码
        """
        # Step 1: 降低二值化阈值检测暗色前景
        _, binary_mask = cv2.threshold(fg_mask, 100, 255, cv2.THRESH_BINARY)

        # Step 2: 形态学闭运算填充孔洞
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # Step 3: 形态学开运算去除小噪声
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # Step 4: 中值滤波降噪
        filtered_mask = cv2.medianBlur(opened_mask, 3)
        return filtered_mask
    
    def apply_no_processing(self, frame: np.ndarray, learning_rate: float = 0.005) -> np.ndarray:
        """
        策略1: 无预处理 + 无后处理
        """
        # 直接应用背景减除，不做任何处理
        fg_mask = self.back_sub.apply(frame, learningRate=learning_rate)
        return fg_mask
    
    def apply_roi_only(self, frame: np.ndarray, learning_rate: float = 0.005) -> np.ndarray:
        """
        策略2: 只加ROI掩码（预处理）
        """
        # 只进行ROI掩码预处理，无后处理
        preprocessed_frame = self._preprocess_frame(frame)
        fg_mask = self.back_sub.apply(preprocessed_frame, learningRate=learning_rate)
        return fg_mask
    
    def apply_post_only(self, frame: np.ndarray, learning_rate: float = 0.005) -> np.ndarray:
        """
        策略3: 只加后处理
        """
        # 无预处理，只进行后处理
        fg_mask = self.back_sub.apply(frame, learningRate=learning_rate)
        processed_mask = self._postprocess_mask(fg_mask)
        return processed_mask
    
    def apply_full_processing(self, frame: np.ndarray, learning_rate: float = 0.005) -> np.ndarray:
        """
        策略4: ROI掩码 + 后处理
        """
        # 完整处理：ROI掩码 + 后处理
        preprocessed_frame = self._preprocess_frame(frame)
        fg_mask = self.back_sub.apply(preprocessed_frame, learningRate=learning_rate)
        processed_mask = self._postprocess_mask(fg_mask)
        return processed_mask

def get_frame_300(video_path: str) -> np.ndarray:
    """获取视频的第300帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 设置到第300帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, 299)  # 0-based index
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError("无法读取第300帧")
    
    return frame

def calculate_mask_statistics(mask: np.ndarray) -> dict:
    """计算掩码的统计信息"""
    foreground_pixels = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]
    foreground_ratio = foreground_pixels / total_pixels
    
    return {
        'foreground_pixels': foreground_pixels,
        'total_pixels': total_pixels,
        'foreground_ratio': foreground_ratio
    }

def create_mask_display(mask: np.ndarray, title: str, stats: dict) -> np.ndarray:
    """创建掩码显示图像"""
    # 将灰度掩码转换为彩色
    if len(mask.shape) == 2:
        display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        display = mask.copy()
    
    # 添加标题和统计信息
    cv2.putText(display, title, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display, f"FG Ratio: {stats['foreground_ratio']:.4f}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, f"FG Pixels: {stats['foreground_pixels']}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return display

def test_four_strategies_on_frame_300():
    """在第300帧上测试四种处理策略"""
    video_path = "data/test_videos/train_enter_station.mp4"
    
    # 检查视频文件是否存在
    if not Path(video_path).exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    # 定义ROI区域
    roi_points = [(200, 200), (600, 700)]
    
    # 初始化四个背景减除器（使用相同的参数确保公平比较）
    bg_subtractors = {
        'no_processing': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16),
        'roi_only': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16),
        'post_only': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16),
        'full_processing': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16)
    }
    
    # 为所有减除器设置相同的ROI
    for bg_sub in bg_subtractors.values():
        bg_sub.setup_single_roi(roi_points, 'train_detection_roi')
    
    print("🚀 开始四种处理策略对比测试")
    print("📊 比较四种处理策略:")
    print("   1. 无预处理 + 无后处理")
    print("   2. 只加ROI掩码（预处理）")
    print("   3. 只加后处理")
    print("   4. ROI掩码 + 后处理")
    
    try:
        # 获取第300帧
        print("\n🎯 正在读取第300帧...")
        frame_300 = get_frame_300(video_path)
        print(f"✅ 成功读取第300帧，尺寸: {frame_300.shape}")
        
        # 应用四种处理策略
        print("\n🔧 应用四种处理策略...")
        masks = {
            '1. No Pre + No Post': bg_subtractors['no_processing'].apply_no_processing(frame_300),
            '2. ROI Only': bg_subtractors['roi_only'].apply_roi_only(frame_300),
            '3. Post Only': bg_subtractors['post_only'].apply_post_only(frame_300),
            '4. ROI + Post': bg_subtractors['full_processing'].apply_full_processing(frame_300)
        }
        
        # 计算统计信息
        stats = {}
        for strategy_name, mask in masks.items():
            stats[strategy_name] = calculate_mask_statistics(mask)
        
        # 显示原图
        original_with_roi = frame_300.copy()
        cv2.rectangle(original_with_roi, roi_points[0], roi_points[1], (0, 255, 0), 2)
        cv2.putText(original_with_roi, "Original Frame (Frame 300)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('0. Original Frame (Frame 300)', original_with_roi)
        
        # 显示四种策略的前景掩码
        for strategy_name, mask in masks.items():
            display = create_mask_display(mask, strategy_name, stats[strategy_name])
            cv2.imshow(strategy_name, display)
        
        # 输出统计信息
        print("\n📈 第300帧处理结果统计:")
        for strategy_name, stat in stats.items():
            print(f"   {strategy_name}:")
            print(f"     前景比例: {stat['foreground_ratio']:.4f}")
            print(f"     前景像素: {stat['foreground_pixels']}")
            print(f"     总像素: {stat['total_pixels']}")
        
        print("\n🎯 按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

def test_four_strategies_on_video():
    """在完整视频上测试四种处理策略"""
    video_path = "data/test_videos/train_enter_station.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    roi_points = [(200, 200), (600, 700)]
    
    # 初始化四个背景减除器
    bg_subtractors = {
        #'no_processing': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16),
        'roi_only': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16),
        #'post_only': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16),
        'full_processing': GMMBackgroundSubtractor(algorithm='MOG2', history=500, varThreshold=16)
    }
    
    for bg_sub in bg_subtractors.values():
        bg_sub.setup_single_roi(roi_points, 'train_detection_roi')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return
    
    print("🚀 开始视频流四种处理策略对比测试")
    print("🎯 按 'q' 退出，按 'p' 暂停/继续，按 'r' 重置背景模型")
    
    paused = False
    frame_count = 0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("✅ 视频播放完毕")
                    break
                
                frame_count += 1
                
                # 应用四种处理策略
                masks = {
                    #'1. No Pre + No Post': bg_subtractors['no_processing'].apply_no_processing(frame),
                    '2. ROI Only': bg_subtractors['roi_only'].apply_roi_only(frame),
                    #'3. Post Only': bg_subtractors['post_only'].apply_post_only(frame),
                    '4. ROI + Post': bg_subtractors['full_processing'].apply_full_processing(frame)
                }
                
                # 显示原图
                original_with_roi = frame.copy()
                cv2.rectangle(original_with_roi, roi_points[0], roi_points[1], (0, 255, 0), 2)
                cv2.putText(original_with_roi, f"Original Frame - Frame {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('0. Original Frame', original_with_roi)
                
                # 显示四种策略的前景掩码
                for strategy_name, mask in masks.items():
                    stats = calculate_mask_statistics(mask)
                    display = create_mask_display(mask, f"{strategy_name} - Frame {frame_count}", stats)
                    cv2.imshow(strategy_name, display)
                
                # 每50帧输出统计信息
                if frame_count % 50 == 0:
                    print(f"\n📈 帧 {frame_count} 统计:")
                    for strategy_name, mask in masks.items():
                        stats = calculate_mask_statistics(mask)
                        print(f"   {strategy_name}: {stats['foreground_ratio']:.4f}")
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'⏸️ 暂停' if paused else '▶️ 继续'}")
            elif key == ord('r'):
                for bg_sub in bg_subtractors.values():
                    bg_sub.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)
                print("🔄 所有背景模型已重置")
    
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"\n✅ 视频测试完成，总共处理 {frame_count} 帧")

if __name__ == "__main__":
    print("请选择测试模式:")
    print("1. 测试第300帧")
    print("2. 测试完整视频")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        test_four_strategies_on_frame_300()
    elif choice == "2":
        test_four_strategies_on_video()
    else:
        print("❌ 无效选择")