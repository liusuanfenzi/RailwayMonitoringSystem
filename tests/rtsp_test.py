# rtsp_test.py
import cv2
import time

def test_rtsp_connection(rtsp_url, test_duration=10):
    """测试RTSP连接"""
    print(f"🔧 开始测试RTSP连接: {rtsp_url}")
    print(f"⏱️  测试持续时间: {test_duration}秒")
    
    cap = None
    try:
        # 尝试连接
        print("🔄 尝试连接RTSP流...")
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            print("❌ 无法打开RTSP流")
            return False
        
        # 获取流信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ RTSP连接成功!")
        print(f"📊 流信息:")
        print(f"   分辨率: {width}x{height}")
        print(f"   帧率: {fps:.1f} FPS")
        
        # 测试帧捕获
        print("\n🎬 开始捕获测试帧...")
        start_time = time.time()
        frame_count = 0
        failed_frames = 0
        
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            
            if ret:
                frame_count += 1
                # 每秒钟显示一次状态
                if frame_count % int(fps or 30) == 0:
                    elapsed = time.time() - start_time
                    print(f"  已捕获 {frame_count} 帧，用时 {elapsed:.1f}秒")
            else:
                failed_frames += 1
            
            # 显示第一帧
            if frame_count == 1 and frame is not None:
                cv2.imshow('RTSP Test - First Frame', frame)
                cv2.waitKey(1000)  # 显示1秒
                cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        actual_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n📊 测试结果:")
        print(f"   总时长: {elapsed:.1f}秒")
        print(f"   成功帧数: {frame_count}")
        print(f"   失败帧数: {failed_frames}")
        print(f"   实际FPS: {actual_fps:.1f}")
        print(f"   成功率: {(frame_count/(frame_count+failed_frames)*100):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ RTSP测试异常: {e}")
        return False
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("🧪 RTSP测试结束")

if __name__ == "__main__":
    # 示例：测试海康摄像头
    test_rtsp_connection("rtsp://admin:13221953816wjy!@192.168.1.64:554/Streaming/Channels/101")