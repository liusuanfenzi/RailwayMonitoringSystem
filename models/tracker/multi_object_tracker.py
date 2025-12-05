# multi_object_tracker_tensorrt.py  (零 PyTorch版)
import numpy as np
import cv2
import time
from collections import deque, defaultdict

class MultiObjectTracker:
    """TensorRT优化的多目标跟踪器（IoU-Only，零PyTorch）"""

    def __init__(self, max_age=70, min_hits=3, iou_threshold=0.3,
                 max_cosine_distance=0.2, nn_budget=None, use_gpu=True):
        # 只用 IoU，不用 ReID，因此 max_cosine_distance / nn_budget 作废
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.roi_points = None
        self.roi_active = False
        self.track_count = 0
        self.memory_cleanup_interval = 50
        
        # 性能监控
        self._last_t = time.time() if 'time' in globals() else 0
        
        # ---- IoU-Only 跟踪器 ----
        self.tracker = self._create_iou_tracker()
        print("✅ IoU-Only TensorRT跟踪器初始化完成")

    # ---------- 核心：纯 NumPy IoU 跟踪 ----------
    def _create_iou_tracker(self):
        """纯 NumPy 版 IoU-DeepSORT（无外观）"""
        return IOUTracker(max_age=self.max_age,
                          min_hits=self.min_hits,
                          iou_th=self.iou_threshold)

    # ---------- 备用：极简回退 ----------
    def _create_fallback_tracker(self):
        print("🔄 使用备用 IoU 跟踪器")
        return IOUTracker(max_age=50, min_hits=3, iou_th=0.3)

    # ---------- 对外接口 ----------
    def set_roi(self, points):
        if len(points) == 2:
            self.roi_points = points
            self.roi_active = True
            print(f"🎯 设置跟踪器ROI: {points}")
            
    def disable_roi(self):
        self.roi_active = False
        print("🔓 禁用ROI跟踪")

    def update(self, detections, frame=None):
        """
        输入: detections = [[x1,y1,x2,y2,conf,class_id], ...]
        输出: [[x1,y1,x2,y2,track_id], ...]  （与旧接口一致）
        """
        self.track_count += 1
        # 1. 空检测快速路径
        if len(detections) == 0:
            tracks = self.tracker.update(np.empty((0, 5)))
            return self._filter_tracks_by_roi(tracks) if self.roi_active else tracks

        # 2. 转格式 → [bbox, conf, class] （IoUTracker 只用 bbox）
        dets_np = np.array(detections)
        if dets_np.size > 0:
            dets_np = dets_np[:, :5].astype(np.float32)  # 只取前5列
        else:
            dets_np = np.empty((0, 5), dtype=np.float32)
            
        tracks = self.tracker.update(dets_np)

        # 3. ROI 过滤
        return self._filter_tracks_by_roi(tracks) if self.roi_active else tracks

    # ---------- ROI 过滤 ----------
    def _filter_tracks_by_roi(self, tracks):
        filtered = []
        for track in tracks:
            if len(track) >= 5:  # 确保有足够的元素
                x1, y1, x2, y2, tid = track[:5]
                cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                if self._point_in_roi(cx, cy):
                    filtered.append([x1, y1, x2, y2, tid])
        return filtered

    def _point_in_roi(self, x, y):
        if not self.roi_active or self.roi_points is None:
            return True
        (x1, y1), (x2, y2) = self.roi_points
        return x1 <= x <= x2 and y1 <= y <= y2

    # ---------- 可视化 ----------
    def visualize_tracking(self, frame, tracked_objects, staying_objects=None, alerted_objects=None):
        staying_objects = staying_objects or set()
        alerted_objects = alerted_objects or set()
        vis = frame.copy()
        
        # 绘制ROI
        if self.roi_active and self.roi_points:
            cv2.rectangle(vis, self.roi_points[0], self.roi_points[1], (0, 255, 0), 2)
            
        # 绘制跟踪框
        for obj in tracked_objects:
            if len(obj) < 5:
                continue
            x1, y1, x2, y2, tid = obj[:5]
            color = (0, 0, 255) if tid in staying_objects else (0, 255, 0)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(vis, f'ID:{int(tid)}', (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return vis

    def reset(self):
        """重置跟踪器状态"""
        self.tracker = self._create_iou_tracker()
        print("🔄 跟踪器状态已重置")


# ---------- 内部：纯 NumPy IoUTracker ----------
class IOUTracker:
    def __init__(self, max_age=50, min_hits=2, iou_th=0.3):
        self.max_age, self.min_hits, self.iou_th = max_age, min_hits, iou_th
        self.tracks = []          # list of dict
        self.next_id = 1

    def update(self, dets):
        # dets: [[x1,y1,x2,y2,conf], ...]  conf not used in iou
        for t in self.tracks:
            t['age'] += 1
            
        if dets.size == 0:
            # 没有检测时，只更新年龄
            self.tracks = [t for t in self.tracks if t['age'] < self.max_age]
            return [(t['box'][0], t['box'][1], t['box'][2], t['box'][3], t['id'])
                    for t in self.tracks if t['hits'] >= self.min_hits]
        
        matched, unmatched_dets, unmatched_trks = self._match(dets[:, :4])
        
        # 更新匹配的track
        for idx_trk, idx_det in matched:
            self.tracks[idx_trk]['box'] = dets[idx_det][:4]
            self.tracks[idx_trk]['hits'] += 1
            self.tracks[idx_trk]['age'] = 0
            
        # 为未匹配的检测创建新track
        for i in unmatched_dets:
            self.tracks.append({
                'id': self.next_id, 
                'box': dets[i][:4],
                'age': 0, 
                'hits': 1
            })
            self.next_id += 1
            
        # 清理过期的track
        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]
        
        # 返回满足最小命中次数的track
        return [(t['box'][0], t['box'][1], t['box'][2], t['box'][3], t['id'])
                for t in self.tracks if t['hits'] >= self.min_hits]

    def _match(self, boxes):
        if not self.tracks or boxes.shape[0] == 0:
            return [], list(range(boxes.shape[0])), []
            
        iou_mat = self._iou_batch([t['box'] for t in self.tracks], boxes)
        matched = []
        
        while iou_mat.max() > self.iou_th:
            idx_trk, idx_det = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            matched.append((idx_trk, idx_det))
            iou_mat[idx_trk, :] = -1
            iou_mat[:, idx_det] = -1
            
        unmatched_d = list(set(range(boxes.shape[0])) - set([m[1] for m in matched]))
        unmatched_t = list(set(range(len(self.tracks))) - set([m[0] for m in matched]))
        
        return matched, unmatched_d, unmatched_t

    @staticmethod
    def _iou_batch(boxes, dets):
        # boxes: List[array], dets: ndarray[N,4]
        if len(boxes) == 0 or len(dets) == 0:
            return np.array([])
            
        boxes = np.array(boxes)          # [M,4]
        x11, y11, x12, y12 = np.split(boxes, 4, axis=1)
        x21, y21, x22, y22 = np.split(dets, 4, axis=1)
        
        xA = np.maximum(x11, x21.T)
        xB = np.minimum(x12, x22.T)
        yA = np.maximum(y11, y21.T)
        yB = np.minimum(y12, y22.T)
        
        inter = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)
        union = area1 + area2.T - inter
        
        return inter / (union + 1e-7)