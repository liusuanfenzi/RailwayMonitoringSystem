#!/usr/bin/env python3
"""
简化的列车检测状态管理器 - 基于空域+时域双重判断
"""

from enum import Enum
import time

class TrainState(Enum):
    """列车状态枚举"""
    NO_TRAIN = 0      # 无列车
    TRAIN_ENTERING = 1  # 列车进站中

class TrainStateManager:
    """简化的列车状态管理器 - 基于空域+时域双重判断"""

    def __init__(self,
                 spatial_threshold: float = 0.05,
                 temporal_frames: int = 100,
                 temporal_threshold: int = 90):

        self.state = TrainState.NO_TRAIN
        self.spatial_threshold = spatial_threshold  # 空域阈值
        self.temporal_frames = temporal_frames      # 时域帧数
        self.temporal_threshold = temporal_threshold  # 时域阈值

        # 时域判断变量
        self.detection_buffer = []  # 检测缓冲区
        self.trigger_frame = -1     # 触发帧索引
        self.entry_frame = -1       # 进站事件触发帧索引

        # 事件计数
        self.entry_count = 0

        print(f"🚂 简化列车状态管理器初始化完成")
        print(f"   空域阈值: {spatial_threshold}")
        print(f"   时域帧数: {temporal_frames}")
        print(f"   时域阈值: {temporal_threshold}")

    def update_state(self, confidence: float, frame_index: int) -> dict:
        """
        基于置信度的状态转移逻辑
        
        核心逻辑：
        1. 单帧置信度必须超过空域阈值
        2. 连续多帧满足时域条件才判定进站
        3. 进站状态持续一段时间后自动回到无列车状态
        """
        # 空域判断 bool
        spatial_detected = confidence > self.spatial_threshold
        
        # 初始化结果
        result = {
            'state': self.state,
            'spatial_detected': spatial_detected,
            'confidence': confidence,
            'event_triggered': False,
            'buffer_size': len(self.detection_buffer)
        }

        # 状态转移逻辑
        if self.state == TrainState.NO_TRAIN:
            # 无列车状态下，只有当单帧置信度超过阈值时才考虑时域判断
            if spatial_detected:
                # 如果是首次检测到，初始化缓冲区
                if self.trigger_frame == -1:
                    self.trigger_frame = frame_index
                    self.detection_buffer = [True]
                else:
                    # 更新检测缓冲区
                    self.detection_buffer.append(True)
            
                    # 保持缓冲区大小
                    if len(self.detection_buffer) > self.temporal_frames:
                        self.detection_buffer.pop(0)
                
                # 检查时域条件
                if len(self.detection_buffer) >= self.temporal_frames:
                    true_count = len(self.detection_buffer)  # 因为只记录了True
                    temporal_condition = true_count >= self.temporal_threshold
                    
                    if temporal_condition:
                        # 时域条件满足，判定进站
                        result.update({
                            'state': TrainState.TRAIN_ENTERING,
                            'event_triggered': True,
                            'event_type': 'train_entered',
                            'true_count': true_count,
                            'temporal_condition': True
                        })
                        self.entry_count += 1
                        self.entry_frame = frame_index
                        
                        # 重置检测状态（保持TRAIN_ENTERING状态）
                        self.detection_buffer = []
                        self.trigger_frame = -1
                        
                        # 更新内部状态
                        self.state = TrainState.TRAIN_ENTERING
            else:
                # 当前帧不满足空域条件，重置检测状态
                self.detection_buffer = []
                self.trigger_frame = -1

        elif self.state == TrainState.TRAIN_ENTERING:
            # 进站状态下，检查是否需要回到无列车状态
            # 简单的超时机制：进站状态持续一定帧数后自动结束
            frames_in_state = frame_index - self.entry_frame
            
            # 如果进站状态持续超过时域帧数的2倍，则回到无列车状态
            if frames_in_state >= self.temporal_frames * 2:
                result.update({
                    'state': TrainState.NO_TRAIN,
                    'state_changed': True
                })
                self.state = TrainState.NO_TRAIN
                print(f"🔄 进站状态结束，回到无列车状态 (持续了 {frames_in_state} 帧)")
            
            # 或者，如果连续多帧置信度低于阈值，也回到无列车状态
            elif not spatial_detected:
                # 可以添加一个计数器，连续多帧低于阈值才切换状态
                # 这里简化处理：如果当前帧不满足条件，直接切换
                result.update({
                    'state': TrainState.NO_TRAIN,
                    'state_changed': True
                })
                self.state = TrainState.NO_TRAIN
                print(f"🔄 置信度低于阈值，回到无列车状态")

        return result

    def get_status(self) -> dict:
        """获取当前状态信息"""
        return {
            'current_state': self.state.name,
            'entry_count': self.entry_count,
            'buffer_size': len(self.detection_buffer),
            'spatial_threshold': self.spatial_threshold
        }

    def reset(self):
        """重置状态"""
        self.state = TrainState.NO_TRAIN
        self.detection_buffer = []
        self.trigger_frame = -1
        self.entry_frame = -1
        print("🔄 状态管理器已重置")