# utils/output_manager.py
import cv2
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import numpy as np

class OutputManager:
    """输出管理器，负责保存事件截图和检测结果"""
    
    def __init__(self, base_output_dir: Optional[Union[str, Path]] = None):
        """
        初始化输出管理器
        
        Args:
            base_output_dir: 基础输出目录，如果为None则使用默认目录
        """
        if base_output_dir is None:
            # 默认输出目录为项目根目录下的 output 文件夹
            self.base_output_dir = Path(__file__).parent.parent / "alerts"
        else:
            self.base_output_dir = Path(base_output_dir)
        
        # 确保基础目录存在
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 快速修复：确保快照/日志/结果目录存在
        # 为了让模块直接保存到 `alerts/<subfolder>`，我们将快照目录设为 base_output_dir
        self.snapshots_dir = self.base_output_dir
        self.logs_dir = self.base_output_dir / "logs"
        self.results_dir = self.base_output_dir / "results"

        for directory in [self.snapshots_dir, self.logs_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ 输出管理器初始化完成")
        print(f"📁 输出目录: {self.base_output_dir.absolute()}")
    
    def save_event_frame(self, 
                        frame: np.ndarray, 
                        event_type: str, 
                        confidence: float, 
                        frame_index: int,
                        subfolder: Optional[str] = None) -> bool:
        """
        保存事件截图
        
        Args:
            frame: 要保存的图像帧
            event_type: 事件类型 ('entry', 'exit', 等)
            confidence: 置信度
            frame_index: 帧索引
            subfolder: 子文件夹名称
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 创建时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # 构建文件名
            filename = f"{event_type}_{timestamp}_f{frame_index:06d}_c{confidence:.3f}.jpg"
            
            # 确定保存目录
            if subfolder:
                save_dir = self.snapshots_dir / subfolder
            else:
                save_dir = self.snapshots_dir / event_type
            
            # 确保目录存在
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 完整文件路径
            file_path = save_dir / filename
            
            # 保存图像
            success = cv2.imwrite(str(file_path), frame)
            
            if success:
                print(f"💾 保存事件截图: {file_path.name}")
                return True
            else:
                print(f"⚠️ 保存图像失败: {file_path}")
                return False
                
        except Exception as e:
            print(f"❌ 保存事件截图时出错: {e}")
            return False
    
    def save_detection_results(self, 
                             results: dict, 
                             filename: Optional[str] = None) -> str:
        """
        保存检测结果到JSON文件
        
        Args:
            results: 检测结果字典
            filename: 文件名，如果为None则自动生成
            
        Returns:
            str: 保存的文件路径
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"detection_results_{timestamp}.json"
            
            file_path = self.results_dir / filename
            
            # 添加保存时间戳
            results['save_timestamp'] = datetime.now().isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 保存检测结果: {file_path.name}")
            return str(file_path)
            
        except Exception as e:
            print(f"❌ 保存检测结果时出错: {e}")
            return ""
    
    def log_event(self, 
                 event_type: str, 
                 frame_index: int, 
                 confidence: float,
                 additional_info: Optional[dict] = None):
        """
        记录事件到日志文件
        
        Args:
            event_type: 事件类型
            frame_index: 帧索引
            confidence: 置信度
            additional_info: 附加信息
        """
        try:
            # 每日一个日志文件
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = self.logs_dir / f"events_{date_str}.log"
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'frame_index': frame_index,
                'confidence': confidence
            }
            
            if additional_info:
                log_entry.update(additional_info)
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            print(f"❌ 记录事件日志时出错: {e}")
    
    def get_output_summary(self) -> dict:
        """
        获取输出目录摘要信息
        
        Returns:
            dict: 摘要信息
        """
        try:
            summary = {
                'base_directory': str(self.base_output_dir.absolute()),
                'snapshots_count': self._count_files(self.snapshots_dir),
                'log_files_count': self._count_files(self.logs_dir),
                'result_files_count': self._count_files(self.results_dir),
                'subfolders': {}
            }
            
            # 统计子文件夹
            if self.snapshots_dir.exists():
                for subdir in self.snapshots_dir.iterdir():
                    if subdir.is_dir():
                        summary['subfolders'][subdir.name] = self._count_files(subdir)
            
            return summary
            
        except Exception as e:
            print(f"❌ 获取输出摘要时出错: {e}")
            return {}
    
    def _count_files(self, directory: Path) -> int:
        """统计目录中的文件数量"""
        if not directory.exists():
            return 0
        return len([f for f in directory.iterdir() if f.is_file()])
    
    def cleanup_old_files(self, 
                         days_old: int = 30, 
                         keep_min_snapshots: int = 100):
        """
        清理旧文件
        
        Args:
            days_old: 保留多少天内的文件
            keep_min_snapshots: 至少保留的截图数量
        """
        try:
            cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            deleted_count = 0
            
            # 清理截图文件（保留最少数量）
            snapshot_files = []
            for file_path in self.snapshots_dir.rglob('*.jpg'):
                snapshot_files.append((file_path, file_path.stat().st_mtime))
            
            # 按时间排序，保留最新的
            snapshot_files.sort(key=lambda x: x[1], reverse=True)
            
            for file_path, mtime in snapshot_files[keep_min_snapshots:]:
                if mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
            
            print(f"🧹 清理了 {deleted_count} 个旧文件")
            return deleted_count
            
        except Exception as e:
            print(f"❌ 清理文件时出错: {e}")
            return 0
    
    def create_test_image(self, 
                         text: str = "Test Output",
                         size: tuple = (640, 480)) -> np.ndarray:
        """
        创建测试图像（用于调试）
        
        Args:
            text: 显示的文本
            size: 图像尺寸 (宽, 高)
            
        Returns:
            np.ndarray: 测试图像
        """
        width, height = size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        cv2.putText(image, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
        
        # 添加边框
        cv2.rectangle(image, (10, 10), (width-10, height-10), (0, 255, 0), 2)
        
        return image

# 简单的单例模式，方便全局访问
_default_output_manager = None

def get_output_manager() -> OutputManager:
    """获取默认的输出管理器实例"""
    global _default_output_manager
    if _default_output_manager is None:
        _default_output_manager = OutputManager()
    return _default_output_manager