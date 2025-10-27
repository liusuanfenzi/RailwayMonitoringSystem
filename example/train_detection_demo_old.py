from models.background_subtractors.gmm_model_old1 import GMMBackgroundSubtractor
from models.detectors.train_station_detector_old import TrainStationDetector


# 使用示例
def main():
    # 初始化背景减除器
    bg_subtractor = GMMBackgroundSubtractor(algorithm='MOG2', history=500)
    
    # 设置ROI区域
    rois_config = {
        'entry_region': [(400, 0), (700, 300)],
        'exit_region': [(100, 450), (520, 900)]
    }
    bg_subtractor.setup_rois(rois_config)
    
    # 初始化列车检测器
    train_detector = TrainStationDetector(
        min_stay_duration=5.0,
        cooldown_duration=2.0,
        # entry_threshold=0.3,
        # exit_threshold=0.2
        entry_threshold=0.06,
        exit_threshold=0.06
    )
    
    # 处理视频
    stats = train_detector.process_video_with_detection(
        video_path="data/test_videos/train_enter_station.mp4",
        bg_subtractor=bg_subtractor,
        max_frames=1500,
        show_visualization=True
    )
    
    return stats

if __name__ == "__main__":
    main()