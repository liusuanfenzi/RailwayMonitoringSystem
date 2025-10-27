#!/usr/bin/env python3
"""
列车检测状态管理器
"""

from enum import Enum
from typing import Optional
import time

class TrainState(Enum):
    """列车状态枚举"""
    NO_TRAIN = 0      # 无列车
    ENTERING = 1      # 进站中
    IN_STATION = 2    # 在站内
    EXITING = 3       # 出站中

class TrainStateManager:
    """列车状态管理器"""
    
    def __init__(self, 
             min_stay_duration: float = 5.0,
             cooldown_duration: float = 3.0,
             entering_timeout: float = 15.0,
             exiting_timeout: float = 15.0,
             min_entering_duration: float = 3.0):  # 新增：最小进站时间
        
        self.state = TrainState.NO_TRAIN
        self.min_stay_duration = min_stay_duration
        self.cooldown_duration = cooldown_duration
        self.entering_timeout = entering_timeout
        self.exiting_timeout = exiting_timeout
        self.min_entering_duration = min_entering_duration
        
        # 时间记录
        self.state_start_time = time.time()
        self.last_event_time = 0.0
        self.enter_start_time: Optional[float] = None
        self.exit_start_time: Optional[float] = None
        
        # 事件计数
        self.entry_count = 0
        self.exit_count = 0
        
        print(f"🚂 列车状态管理器初始化完成")
        print(f"   最小停留时间: {min_stay_duration}秒")
        print(f"   事件冷却时间: {cooldown_duration}秒")

    def update_state(self, 
                entry_detected: bool, 
                exit_detected: bool, 
                frame_timestamp: float) -> dict:
        """
        基于检测信号的状态转移逻辑
        """
        current_time = frame_timestamp
        state_duration = current_time - self.state_start_time

        # 检查冷却时间 - 修复初始化问题
        if self.last_event_time == 0:  # 初始状态，没有发生过事件
            in_cooldown = False
        else:
            time_since_last_event = current_time - self.last_event_time
            in_cooldown = time_since_last_event < self.cooldown_duration

        result = {
            'state': self.state,
            'state_changed': False,
            'new_state': None,
            'event_triggered': False,
            'event_type': None,
            'in_cooldown': in_cooldown,
            'state_duration': state_duration
        }
        
        # 状态转移逻辑 - 基于检测信号的时序关系
        if self.state == TrainState.NO_TRAIN:
            if entry_detected and not exit_detected and not in_cooldown:
                # 只有进站区域有检测，出站区域没有 → 列车开始进站
                self._transition_to(TrainState.ENTERING, current_time)
                result.update({
                    'state_changed': True,
                    'new_state': TrainState.ENTERING,
                    'event_triggered': True,
                    'event_type': 'train_entering_start'
                })
                self.enter_start_time = current_time
            
        elif self.state == TrainState.ENTERING:
            # 检查进站超时
            if state_duration > self.entering_timeout:
                self._transition_to(TrainState.NO_TRAIN, current_time)
                result.update({
                    'state_changed': True,
                    'new_state': TrainState.NO_TRAIN,
                    'event_triggered': True,
                    'event_type': 'entering_timeout'
                })
                
                # 关键逻辑：进站完成的条件
            elif entry_detected and exit_detected:
                # 两个区域都有检测 → 列车正在站内
                if state_duration >= self.min_entering_duration:  # 确保进站过程持续了一段时间
                    self._transition_to(TrainState.IN_STATION, current_time)
                    result.update({
                        'state_changed': True,
                        'new_state': TrainState.IN_STATION,
                        'event_triggered': True,
                        'event_type': 'train_entered'
                    })
                    self.entry_count += 1
                
            elif not entry_detected and exit_detected:
                # 进站区域消失，出站区域有检测 → 可能进站完成
                if state_duration >= 2.0:  # 短暂延迟确认
                    self._transition_to(TrainState.IN_STATION, current_time)
                    result.update({
                        'state_changed': True,
                        'new_state': TrainState.IN_STATION,
                        'event_triggered': True,
                        'event_type': 'train_entered'
                    })
                    self.entry_count += 1
            
        elif self.state == TrainState.IN_STATION:
            # 检查最小停留时间
            min_stay_met = state_duration >= self.min_stay_duration
                
            if exit_detected and not entry_detected and min_stay_met:
                # 只有出站区域有检测 → 列车开始出站
                self._transition_to(TrainState.EXITING, current_time)
                result.update({
                    'state_changed': True,
                    'new_state': TrainState.EXITING,
                    'event_triggered': True,
                    'event_type': 'train_exiting_start'
                })
                self.exit_start_time = current_time
                
            elif not exit_detected and not entry_detected and min_stay_met:
                # 两个区域都没有检测 → 列车可能已出站
                if state_duration >= self.min_stay_duration + 2.0:  # 额外确认时间
                    self._transition_to(TrainState.NO_TRAIN, current_time)
                    result.update({
                        'state_changed': True,
                        'new_state': TrainState.NO_TRAIN,
                        'event_triggered': True,
                        'event_type': 'train_exited'
                    })
                    self.exit_count += 1
                    self.last_event_time = current_time
            
        elif self.state == TrainState.EXITING:
            # 检查出站超时
            if state_duration > self.exiting_timeout:
                self._transition_to(TrainState.NO_TRAIN, current_time)
                result.update({
                    'state_changed': True,
                    'new_state': TrainState.NO_TRAIN,
                    'event_triggered': True,
                    'event_type': 'exiting_timeout'
                })
                
                # 关键逻辑：出站完成的条件
            elif not exit_detected and not entry_detected:
                # 两个区域都没有检测 → 列车完全出站
                if state_duration >= 2.0:  # 短暂延迟确认
                    self._transition_to(TrainState.NO_TRAIN, current_time)
                    result.update({
                        'state_changed': True,
                        'new_state': TrainState.NO_TRAIN,
                        'event_triggered': True,
                        'event_type': 'train_exited'
                    })
                    self.exit_count += 1
                    self.last_event_time = current_time
                
            elif not exit_detected and entry_detected:
                # 出站区域消失，进站区域有检测 → 可能是新列车
                if state_duration >= 3.0:  # 需要更长时间确认
                    self._transition_to(TrainState.ENTERING, current_time)
                    result.update({
                        'state_changed': True,
                        'new_state': TrainState.ENTERING,
                        'event_triggered': True,
                        'event_type': 'train_entering_start'
                    })
                    self.enter_start_time = current_time
        
        return result
    
    def _transition_to(self, new_state: TrainState, timestamp: float):
        """执行状态转移"""
        old_state = self.state
        self.state = new_state
        self.state_start_time = timestamp
        
        print(f"🔄 状态转移: {old_state.name} -> {new_state.name}")
    
    def get_status(self) -> dict:
        """获取当前状态信息"""
        return {
            'current_state': self.state.name,
            'entry_count': self.entry_count,
            'exit_count': self.exit_count,
            'state_duration': time.time() - self.state_start_time,
            'time_since_last_event': time.time() - self.last_event_time
        }
    
    def reset(self):
        """重置状态"""
        self.state = TrainState.NO_TRAIN
        self.state_start_time = time.time()
        self.last_event_time = time.time()
        self.enter_start_time = None
        self.exit_start_time = None
        print("🔄 状态管理器已重置")

    