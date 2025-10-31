import numpy as np
import cv2
import torch

class MultiObjectTracker:
    """多目标跟踪器封装（基于DeepSORT算法）"""
    
    def __init__(self, max_age=70, min_hits=3, iou_threshold=0.3, 
                 max_cosine_distance=0.2, nn_budget=None, use_gpu=True):
        """
        初始化DeepSORT跟踪器
        
        Args:
            max_age: 目标丢失多少帧后删除
            min_hits: 需要多少帧连续检测才创建跟踪
            iou_threshold: IOU匹配阈值
            max_cosine_distance: 外观特征余弦距离阈值
            nn_budget: 外观特征缓存大小
            use_gpu: 是否使用GPU进行特征提取
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.use_gpu = use_gpu
        
        # ROI相关
        self.roi_points = None
        self.roi_active = False
        
        # 初始化DeepSORT跟踪器
        self.tracker = self._create_deepsort_tracker()
        
        print("✅ DeepSORT跟踪器初始化完成")
        
    def _create_deepsort_tracker(self):
        """创建DeepSORT跟踪器实例"""
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            
            # 检查可用的设备
            device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
            print(f"🔧 使用设备: {device}")

            # 初始化DeepSORT
            tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.min_hits,
                max_iou_distance=self.iou_threshold,
                max_cosine_distance=self.max_cosine_distance,
                nn_budget=self.nn_budget,
                nms_max_overlap=0.8,  # NMS重叠阈值
                embedder="mobilenet",  # 使用轻量级特征提取器
                half=True if device == 'cuda' else False,  # 仅在GPU上使用半精度
                bgr=True,
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None
            )
            
            return tracker
            
        except ImportError:
            print("❌ 无法导入DeepSORT，请安装: pip install deep-sort-realtime")
            raise ImportError("DeepSORT库未安装")
    
    def set_roi(self, points):
        """设置ROI区域"""
        if len(points) == 2:
            self.roi_points = points
            self.roi_active = True
            print(f"🎯 设置跟踪器ROI: {points}")
        else:
            print("⚠️ ROI点必须是两个点 [(x1,y1), (x2,y2)]")
    
    def disable_roi(self):
        """禁用ROI跟踪"""
        self.roi_active = False
        print("🔓 禁用ROI跟踪，使用全图跟踪")
    
    def update(self, detections, frame=None):
        """
        更新跟踪器（可选ROI区域）
        
        Args:
            detections: 检测结果数组 [[x1, y1, x2, y2, confidence, class_id], ...]
            frame: 当前帧图像（用于提取外观特征）
            
        Returns:
            tracked_objects: 跟踪结果 [[x1, y1, x2, y2, track_id], ...]
        """
        if frame is None:
            print("⚠️ DeepSORT需要帧图像进行特征提取，使用空结果")
            return []
        
        # 如果启用了ROI且检测结果为空，检查ROI区域内是否有需要跟踪的目标
        if self.roi_active and len(detections) == 0:
            tracks = self.tracker.update_tracks([], frame=frame)
            # 过滤ROI区域外的跟踪结果
            return self._filter_tracks_by_roi(tracks)
        
        if len(detections) == 0:
            # 没有检测结果时更新跟踪器
            tracks = self.tracker.update_tracks([], frame=frame)
            if self.roi_active:
                return self._filter_tracks_by_roi(tracks)
            return self._parse_tracks(tracks)
        
        # 转换检测结果为DeepSORT格式
        deepsort_detections = []
        for det in detections:
            if len(det) >= 6:
                try:
                    # 确保所有值都是数字类型
                    x1, y1, x2, y2, confidence, class_id = det[:6]
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    confidence = float(confidence)
                    class_id = int(class_id)
                    
                    bbox = [x1, y1, x2-x1, y2-y1]  # DeepSORT使用[x,y,w,h]格式
                    deepsort_detections.append((bbox, confidence, class_id))
                except (ValueError, TypeError) as e:
                    print(f"⚠️ 检测格式错误: {e}，跳过该检测")
                    continue
        
        # 更新跟踪器
        try:
            tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
            
            # 如果启用了ROI，过滤ROI区域外的跟踪结果
            if self.roi_active:
                return self._filter_tracks_by_roi(tracks)
            else:
                return self._parse_tracks(tracks)
                
        except Exception as e:
            print(f"❌ DeepSORT更新失败: {e}")
            return []
    
    def _filter_tracks_by_roi(self, tracks):
        """过滤ROI区域外的跟踪结果"""
        filtered_tracks = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            bbox = track.to_tlbr()  # 获取[x1, y1, x2, y2]格式的边界框
            if len(bbox) >= 4:
                try:
                    x1, y1, x2, y2 = map(float, bbox)
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # 检查中心点是否在ROI内
                    if self._point_in_roi(center_x, center_y):
                        filtered_tracks.append(track)
                except (ValueError, TypeError):
                    continue
        
        return self._parse_tracks(filtered_tracks)
    
    def _point_in_roi(self, x, y):
        """检查点是否在ROI内"""
        if not self.roi_active or self.roi_points is None:
            return True
            
        x1, y1 = self.roi_points[0]
        x2, y2 = self.roi_points[1]
        
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def _parse_tracks(self, tracks):
        """解析DeepSORT跟踪结果"""
        tracked_objects = []
        seen_track_ids = set()  # 用于检查track_id是否重复
    
        for track in tracks:
            if not track.is_confirmed():  # 只有达到min_hits的检测结果才被confirmed
                continue
    
            track_id = track.track_id
            if track_id in seen_track_ids:  # 检查track_id是否重复
                continue
    
            bbox = track.to_tlbr()  # 获取[x1, y1, x2, y2]格式的边界框
            if len(bbox) != 4:  # 确保边界框有4个值
                continue
    
            try:
                x1, y1, x2, y2 = map(float, bbox)  # 转换为浮点数
                if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:  # 检查边界框合法性
                    tracked_objects.append([x1, y1, x2, y2, int(track_id)])
                    seen_track_ids.add(track_id)  # 记录已处理的track_id
            except (ValueError, TypeError) as e:
                print(f"⚠️ 坐标转换错误: {e}，跳过该轨迹")
                continue
    
        return tracked_objects if tracked_objects else []
        
    # def visualize_tracking(self, frame, tracked_objects, staying_objects=None):
    #     """
    #     可视化跟踪结果
        
    #     Args:
    #         frame: 输入图像
    #         tracked_objects: 跟踪结果
    #         staying_objects: 停留对象集合
            
    #     Returns:
    #         visualized_frame: 可视化后的图像
    #     """
    #     display_frame = frame.copy()
    #     staying_objects = staying_objects or set()
        
    #     # 绘制ROI区域（如果启用）
    #     if self.roi_active and self.roi_points:
    #         cv2.rectangle(display_frame, 
    #                      self.roi_points[0], self.roi_points[1],
    #                      (0, 255, 0), 2)
    #         cv2.putText(display_frame, "Detection ROI", 
    #                    (self.roi_points[0][0], self.roi_points[0][1] - 10),
    #                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    #     for obj in tracked_objects:
    #         if len(obj) >= 5:
    #             try:
    #                 # 确保所有值都是数字类型
    #                 x1, y1, x2, y2, track_id = obj[:5]
    #                 x1 = float(x1)
    #                 y1 = float(y1)
    #                 x2 = float(x2)
    #                 y2 = float(y2)
    #                 track_id = int(float(track_id))  # 确保track_id是整数
                    
    #                 # 转换为整数用于绘制
    #                 x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
                    
    #                 # 设置颜色和标签
    #                 color = (0, 255, 0)  # 绿色-正常
    #                 label = f"ID:{track_id}"
                    
    #                 if track_id in staying_objects:
    #                     color = (0, 0, 255)  # 红色-停留
    #                     label = f"ID:{track_id} (STAY)"
                    
    #                 # 绘制边界框和标签
    #                 cv2.rectangle(display_frame, (x1_int, y1_int), (x2_int, y2_int), color, 2)
    #                 cv2.putText(display_frame, label, (x1_int, y1_int-10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
    #             except (ValueError, TypeError) as e:
    #                 print(f"⚠️ 可视化错误: {e}，跳过该对象")
    #                 continue
        
    #     return display_frame

    def visualize_tracking(self, frame, tracked_objects, staying_objects=None, alerted_objects=None):
        """
        可视化跟踪结果
        
        Args:
            frame: 输入图像
            tracked_objects: 跟踪结果
            staying_objects: 当前停留对象集合
            alerted_objects: 曾经报警过的对象集合
            
        Returns:
            visualized_frame: 可视化后的图像
        """
        display_frame = frame.copy()
        staying_objects = staying_objects or set()
        alerted_objects = alerted_objects or set()
        
        # 绘制ROI区域（如果启用）
        if self.roi_active and self.roi_points:
            cv2.rectangle(display_frame, 
                        self.roi_points[0], self.roi_points[1],
                        (0, 255, 0), 2)
            cv2.putText(display_frame, "Detection ROI", 
                    (self.roi_points[0][0], self.roi_points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        for obj in tracked_objects:
            if len(obj) >= 5:
                try:
                    # 确保所有值都是数字类型
                    x1, y1, x2, y2, track_id = obj[:5]
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    track_id = int(float(track_id))  # 确保track_id是整数
                    
                    # 转换为整数用于绘制
                    x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
                    
                    # 设置颜色和标签
                    if track_id in staying_objects:
                        color = (0, 0, 255)  # 红色 - 当前停留
                        status = "STAYING"
                    elif track_id in alerted_objects:
                        color = (0, 0, 255)  # 蓝色 - 曾经停留过
                        status = "ALERTED"
                    else:
                        color = (0, 255, 0)  # 绿色 - 正常
                        status = "TRACKING"
                    
                    label = f"ID:{track_id} - {status}"
                    
                    # 绘制边界框和标签
                    cv2.rectangle(display_frame, (x1_int, y1_int), (x2_int, y2_int), color, 2)
                    cv2.putText(display_frame, label, (x1_int, y1_int-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                except (ValueError, TypeError) as e:
                    print(f"⚠️ 可视化错误: {e}，跳过该对象")
                    continue
        
        return display_frame