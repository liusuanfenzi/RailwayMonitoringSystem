# utils/jetson_state_manager.py
from enum import Enum
import time


class TrainState(Enum):
    """列车状态枚举"""
    NO_TRAIN = 0      # 无列车
    TRAIN_ENTERING = 1  # 列车进站中


class TrainStateManager:
    """Jetson优化的列车状态管理器"""
    def __init__(self,
                 spatial_threshold: float = 0.05,
                 temporal_frames: int = 50,  # 减少帧数，适应Jetson性能
                 temporal_threshold: int = 45):  # 相应调整阈值

        self.state = TrainState.NO_TRAIN
        self.spatial_threshold = spatial_threshold
        self.temporal_frames = temporal_frames
        self.temporal_threshold = temporal_threshold

        # 优化数据结构，使用固定大小数组
        self.detection_buffer = [False] * temporal_frames
        self.buffer_index = 0
        self.trigger_frame = -1
        self.entry_frame = -1
        self.entry_count = 0

        print(f"🚂 Jetson列车状态管理器初始化")
        print(f"   空域阈值: {spatial_threshold}")
        print(f"   时域帧数: {temporal_frames}")
        print(f"   时域阈值: {temporal_threshold}")

    def update_state(self, confidence: float, frame_index: int) -> dict:
        """优化状态更新逻辑"""
        spatial_detected = confidence > self.spatial_threshold

        result = {
            'state': self.state,
            'spatial_detected': spatial_detected,
            'confidence': confidence,
            'event_triggered': False,
        }

        if self.state == TrainState.NO_TRAIN:
            if spatial_detected:
                # 更新循环缓冲区
                self.detection_buffer[self.buffer_index] = True
                self.buffer_index = (self.buffer_index +
                                     1) % self.temporal_frames

                # 检查时域条件
                true_count = sum(self.detection_buffer)
                temporal_condition = true_count >= self.temporal_threshold

                if temporal_condition:
                    result.update({
                        'state': TrainState.TRAIN_ENTERING,
                        'event_triggered': True,
                        'event_type': 'train_entered',
                        'true_count': true_count,
                    })
                    self.entry_count += 1
                    self.entry_frame = frame_index
                    self.state = TrainState.TRAIN_ENTERING
                    # 重置缓冲区
                    self.detection_buffer = [False] * self.temporal_frames
                    self.buffer_index = 0
            else:
                # 更新缓冲区为False
                self.detection_buffer[self.buffer_index] = False
                self.buffer_index = (self.buffer_index +
                                     1) % self.temporal_frames

        elif self.state == TrainState.TRAIN_ENTERING:
            # 简化状态退出逻辑
            frames_in_state = frame_index - self.entry_frame
            if frames_in_state >= self.temporal_frames * 3 or not spatial_detected:
                result.update({'state': TrainState.NO_TRAIN})
                self.state = TrainState.NO_TRAIN

        return result

    def get_status(self) -> dict:
        """获取状态信息"""
        return {
            'current_state': self.state.name,
            'entry_count': self.entry_count,
            'buffer_fill': sum(self.detection_buffer),
            'spatial_threshold': self.spatial_threshold
        }

    def reset(self):
        """重置状态"""
        self.state = TrainState.NO_TRAIN
        self.detection_buffer = [False] * self.temporal_frames
        self.buffer_index = 0
        self.trigger_frame = -1
        self.entry_frame = -1
        print("🔄 状态管理器已重置")
