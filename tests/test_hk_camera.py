import cv2
import numpy as np
import time
import sys

def simple_hikvision_stream():
    IP = "192.168.1.64"
    USERNAME = "admin"
    RAW_PASSWORD = "13221953816wjy!"
    ENCODED_PASSWORD = RAW_PASSWORD.replace("!", "%21")
    PORT = 554

    # 候选URL列表
    RTSP_URLS = [
        f"rtsp://{USERNAME}:{ENCODED_PASSWORD}@{IP}:{PORT}/Streaming/Channels/101",
        f"rtsp://{USERNAME}:{ENCODED_PASSWORD}@{IP}:{PORT}/Streaming/Channels/102",  # 子码流，更稳定
        f"rtsp://{USERNAME}:{ENCODED_PASSWORD}@{IP}:{PORT}/ISAPI/Streaming/channels/101",
        f"rtsp://{USERNAME}:{ENCODED_PASSWORD}@{IP}:{PORT}/h264/ch1/main/av_stream",
    ]

    cap = None
    selected_url = None

    print("正在尝试连接摄像头...")
    for url in RTSP_URLS:
        safe_url = url.replace(ENCODED_PASSWORD, '******')
        print(f"  尝试: {safe_url}")
        # 直接尝试用OpenCV打开，不经过ffprobe
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            selected_url = url
            print(f"✅ 连接成功！")
            break
        else:
            cap = None

    if cap is None:
        print("❌ 所有URL尝试均失败。")
        print("请检查：1. 摄像头是否供电并联网；2. 用电脑VLC播放器测试同一URL")
        return

    # 设置降低延迟的参数
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # 尝试设置帧率，部分摄像头支持
    # cap.set(cv2.CAP_PROP_FPS, 30)

    print("开始拉流，按 'q' 退出...")
    last_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取帧失败，可能连接中断。")
                # 简单重连逻辑
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(selected_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    print("重连失败，退出。")
                    break
                continue

            # 计算显示FPS
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_time = now
                print(f"实时FPS: {fps}", end='\r')  # 同行更新显示

            # 显示画面
            cv2.imshow('Hikvision Camera Stream', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("\n程序结束。")

if __name__ == "__main__":
    simple_hikvision_stream()
