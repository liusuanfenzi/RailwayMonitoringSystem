# system_controller.py
import threading
import time
import yaml
from .frame_buffer import ThreadSafeFrameBuffer
from .display_manager import ResultManager, UnifiedDisplayManager
from .video_capture import VideoCaptureThread
from .person_vehicle_detector import PersonVehicleDetectionThread
from .train_station_detector import TrainStationDetectionThread
from .foreign_object_thread import ForeignObjectThread


class MultiModuleSystemController:
    """多模块系统控制器 - 可选择性启用检测模块"""
    
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.stop_event = threading.Event()

        # Jetson 专用优化
        if self._is_jetson():
            self._apply_jetson_optimizations()
        
        # 创建共享资源
        self.frame_buffer = ThreadSafeFrameBuffer(max_size=self.config.get('buffer_size', 10))
        self.result_manager = ResultManager()
        
        # 线程列表
        self.threads = []
        
        # 显示管理器
        self.display_manager = None
        
        # 启用的模块配置
        self.enabled_modules = self.config.get('enabled_modules', ['person_vehicle', 'train_station', 'foreign_object'])
        
        # 模块映射表
        self.module_mapping = {
            'person_vehicle': {
                'class': PersonVehicleDetectionThread,
                'name': 'personvehicledetection',
                'display_name': 'person_vehicle',
                'config_prefix': 'person_vehicle'
            },
            'train_station': {
                'class': TrainStationDetectionThread,
                'name': 'trainstationdetection',
                'display_name': 'train_station',
                'config_prefix': 'train_station'
            },
            'foreign_object': {
                'class': ForeignObjectThread,
                'name': 'foreignobjectdetection',
                'display_name': 'foreign_object',
                'config_prefix': 'foreign_object'
            }
        }

    def _is_jetson(self):
        """检查是否在Jetson设备上运行"""
        import platform
        return 'jetson' in platform.machine().lower() or 'aarch64' in platform.machine()

    def _apply_jetson_optimizations(self):
        """应用Jetson优化"""
        print("🚀 检测到Jetson设备，应用优化配置")
        
        # 调整配置参数 - 使用正确的键名
        optimizations = {
            'buffer_size': 5,  # 减少缓冲区大小
            'target_fps': 15,  # 降低目标FPS
            
            # 注意：配置文件中是 person_vehicle_target_fps，不是 person_vehicle_target_fps
            'person_vehicle_target_fps': 10,
            'train_station_target_fps': 5,
            'foreign_object_target_fps': 8,
        }
        
        for key, value in optimizations.items():
            # 检查配置中是否有这个键，或者使用默认值
            if key not in self.config:
                self.config[key] = value
                print(f"  📊 {key}: {value} (默认)")
            elif self.config[key] > value:
                self.config[key] = value
                print(f"  📊 {key}: {value} (优化)")

    def load_config(self, config_path):
        """加载配置文件 - 支持嵌套结构"""
        default_config = {
            # 视频源配置
            'video_source': "data/test_videos/safe_gesture/gf1_new.mp4",
            'video_sources': ["data/test_videos/safe_gesture/gf1_new.mp4", "data/test_videos/trash_in_area/1.mp4"],
            'frame_width': 640,
            'frame_height': 480,
            'target_fps': 30,
            'loop_video': False,
            
            # 启用模块配置
            'enabled_modules': ['person_vehicle', 'train_station', 'foreign_object'],
            
            # 人车检测配置（扁平化）
            'person_vehicle_engine_path': 'yolov8n.engine',
            'person_vehicle_target_fps': 20,
            'person_vehicle_confidence': 0.6,
            
            # 列车检测配置（扁平化）
            'train_station_target_fps': 10,
            'train_station_bg_learning_rate': 0.01,
            'train_station_bg_history': 500,
            'train_station_bg_var_threshold': 16,
            'train_station_bg_detect_shadows': True,
            'train_station_spatial_threshold': 0.05,
            'train_station_temporal_frames': 50,
            'train_station_temporal_threshold': 45,
            'train_station_print_interval': 10,
            'train_station_warmup_frames': 15,
            
            # 跟踪器配置
            'person_vehicle_stay_threshold': 10.0,
            'person_vehicle_movement_threshold': 15.0,
            'person_vehicle_max_age': 50,
            'person_vehicle_min_hits': 2,
            'person_vehicle_iou_threshold': 0.3,
            
            # 显示配置
            'fullscreen': False,
            'buffer_size': 10,
            
            # ROI配置
            'person_vehicle_detection_roi': [[350, 340], [750, 580]],
            'train_station_roi': [[100, 100], [600, 400]],
            
            # 异物检测配置（扁平化）
            'foreign_object_roi': [[550, 400, 400, 300]],
            'foreign_object_min_static_duration': 2.0,
            'foreign_object_threshold': 200,
            'foreign_object_min_area': 100,
            'foreign_object_alert_dir': "alerts/foreign_object_detection",
            'foreign_object_motion_threshold': 800,
            'foreign_object_background_frames': 30,
            'foreign_object_difference_threshold': 50,
            'foreign_object_target_fps': 15
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    if user_config:
                        print(f"✅ 配置文件加载成功: {config_path}")
                        
                        # 将嵌套配置扁平化
                        flattened_config = self._flatten_config(user_config)
                        
                        # 递归合并配置
                        self._merge_configs(default_config, flattened_config)
                        
            except yaml.YAMLError as e:
                print(f"❌ YAML语法错误: {e}")
                print(f"⚠️ 使用默认配置")
            except Exception as e:
                print(f"⚠️ 配置文件加载失败: {e}，使用默认配置")
        
        # 打印最终配置（调试用）
        print("\n📋 最终配置摘要:")
        for key, value in list(default_config.items())[:10]:  # 只打印前10个
            print(f"  {key}: {value}")
        if len(default_config) > 10:
            print(f"  ... 还有 {len(default_config)-10} 个配置项")
        
        return default_config

    def _flatten_config(self, config, prefix=""):
        """将嵌套配置扁平化
        
        Args:
            config: 嵌套配置字典
            prefix: 键名前缀
            
        Returns:
            扁平化的配置字典
        """
        flattened = {}
        
        for key, value in config.items():
            full_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                # 递归扁平化嵌套字典
                flattened.update(self._flatten_config(value, full_key))
            elif isinstance(value, list) and key in ['enabled_modules', 'video_sources']:
                # 列表类型的配置直接保留
                flattened[full_key] = value
            elif key in ['detection_roi', 'train_roi', 'roi']:
                # ROI配置特殊处理
                if full_key == 'foreign_object_roi':
                    flattened['foreign_object_roi'] = value
                elif full_key == 'detection_roi':
                    flattened['person_vehicle_detection_roi'] = value
                elif full_key == 'train_roi':
                    flattened['train_station_roi'] = value
                else:
                    flattened[full_key] = value
            else:
                # 其他配置直接添加
                flattened[full_key] = value
        
        return flattened

    def _merge_configs(self, base, new):
        """递归合并配置字典 - 支持嵌套和扁平配置"""
        for key, value in new.items():
            if key in base:
                # 如果键已存在，根据类型处理
                if isinstance(value, dict) and isinstance(base[key], dict):
                    self._merge_configs(base[key], value)
                else:
                    base[key] = value
            else:
                # 新键直接添加
                base[key] = value

    def show_module_selection(self):
        """显示模块选择菜单"""
        print("\n" + "="*50)
        print("🎯 检测模块选择")
        print("="*50)
        
        available_modules = list(self.module_mapping.keys())
        
        print("请选择要运行的两个检测模块（输入对应数字）：")
        for i, module_key in enumerate(available_modules, 1):
            module_info = self.module_mapping[module_key]
            print(f"  {i}. {module_info['display_name']} ({module_key})")
        
        print(f"  0. 使用配置文件设置 ({', '.join(self.enabled_modules)})")
        print("-"*50)
        
        # 获取用户选择
        selected_modules = []
        while len(selected_modules) < 2:
            try:
                choice = input(f"请选择第{len(selected_modules)+1}个模块（输入数字，0使用配置）: ").strip()
                
                if choice == '0':
                    # 使用配置文件设置
                    if len(self.enabled_modules) >= 2:
                        selected_modules = self.enabled_modules[:2]
                        print(f"✅ 使用配置文件设置: {', '.join(selected_modules)}")
                        break
                    else:
                        print("❌ 配置文件中启用的模块少于2个，请手动选择")
                        continue
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_modules):
                    module_key = available_modules[choice_idx]
                    if module_key in selected_modules:
                        print(f"❌ 模块 {module_key} 已选择，请选择其他模块")
                    else:
                        selected_modules.append(module_key)
                        module_info = self.module_mapping[module_key]
                        print(f"✅ 选择: {module_info['display_name']}")
                else:
                    print(f"❌ 无效选择，请输入 1-{len(available_modules)} 或 0")
            except ValueError:
                print("❌ 请输入有效数字")
            except KeyboardInterrupt:
                print("\n⏹️ 用户中断选择")
                return None
        
        return selected_modules
    
    def initialize_system(self):
        """初始化系统 - 使用统一显示管理器"""
        print("🚀 初始化多模块检测系统")
        
        # 显示模块选择
        selected_modules = self.show_module_selection()
        if not selected_modules:
            print("❌ 模块选择失败，退出系统")
            return False
        
        self.enabled_modules = selected_modules
        
        try:
            self.threads = []
            
            # 获取视频源列表
            video_sources = self.config.get('video_sources')
            if not isinstance(video_sources, (list, tuple)):
                print(f"❌ 视频源配置错误，期望列表，实际: {type(video_sources)}")
                return False
            
            if len(video_sources) < len(self.enabled_modules):
                print(f"❌ 视频源数量不足: {len(video_sources)} 个视频源，但需要 {len(self.enabled_modules)} 个")
                return False
            
            print(f"📊 为 {len(self.enabled_modules)} 个模块分配视频源...")
            
            # 创建显示管理器
            self.display_manager = UnifiedDisplayManager(
                self.result_manager, 
                self.stop_event, 
                self.config
            )
            
            # 窗口位置配置（避免重叠）
            window_positions = [
                (100, 100),    # 窗口1位置
                (1000, 100),   # 窗口2位置
                (100, 650),    # 窗口3位置（如果有）
            ]
            
            # 为每个选择的模块创建独立的处理管道
            for idx, module_key in enumerate(self.enabled_modules):
                src = video_sources[idx]
                print(f"\n🔗 创建模块管道 {idx+1}: {module_key} -> {src}")
                
                # 1. 创建独立的帧缓冲区（带名称）
                buffer_name = f"Buffer_{module_key}_{idx}"
                fb = ThreadSafeFrameBuffer(max_size=self.config.get('buffer_size', 10), name=buffer_name)
                print(f"   ✅ 创建帧缓冲区: {buffer_name}")
                
                # 2. 创建视频捕获线程
                cap_thread = VideoCaptureThread(src, fb, self.result_manager, self.stop_event, self.config)
                self.threads.append(cap_thread)
                print(f"   ✅ 创建视频捕获线程: {src}")
                
                # 3. 创建检测线程
                if module_key in self.module_mapping:
                    module_info = self.module_mapping[module_key]
                    ThreadClass = module_info['class']
                    thread_name = module_info['name']
                    
                    print(f"   ✅ 创建检测线程: {module_info['display_name']}")
                    
                    thread_instance = ThreadClass(
                        name=thread_name,
                        frame_buffer=fb,  # 使用独立的缓冲区
                        result_manager=self.result_manager,  # 使用共享的结果管理器
                        stop_event=self.stop_event,
                        config=self.config
                    )
                    self.threads.append(thread_instance)
                    
                    # 4. 在显示管理器中注册窗口
                    if idx < len(window_positions):
                        window_pos = window_positions[idx]
                    else:
                        window_pos = (100 + idx * 50, 100 + idx * 50)
                    
                    window_name = f"{module_info['display_name']} - Source {idx+1}"
                    self.display_manager.add_window(
                        window_name=window_name,
                        module_key=thread_name,  # 使用检测线程的标准化名称
                        position=window_pos,
                        size=(800, 600)
                    )
                else:
                    print(f"❌ 未知模块: {module_key}")
                    return False
                
                print(f"   ✅ 模块 {module_key} 管道创建完成")
            
            print(f"\n✅ 成功创建 {len(self.threads)} 个线程和 {len(self.enabled_modules)} 个显示窗口")
            
            # 显示配置信息
            print("\n🔗 视频源 -> 模块 映射:")
            for idx, module_key in enumerate(self.enabled_modules):
                src = video_sources[idx]
                display_name = self.module_mapping.get(module_key, {}).get('display_name', module_key)
                print(f"  - [{idx}] {src} -> {display_name} ({module_key})")

            self.show_configuration_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ 线程初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def show_configuration_summary(self):
        """显示配置摘要"""
        print("\n" + "="*50)
        print("📋 系统配置摘要")
        print("="*50)
        # 显示单/多视频源信息
        if isinstance(self.config.get('video_sources'), (list, tuple)):
            print(f"📹 视频源列表: {self.config.get('video_sources')}")
        else:
            print(f"📹 视频源: {self.config.get('video_source')}")
        print(f"🖥️ 显示模式: {'全屏' if self.config['fullscreen'] else '窗口'}")
        print(f"📊 缓冲区大小: {self.config.get('buffer_size', 10)}")
        print(f"🎯 目标FPS: {self.config.get('target_fps', 30)}")
        print("-"*50)
        
        # 显示启用的模块
        enabled_display_names = []
        for module_key in self.enabled_modules:
            if module_key in self.module_mapping:
                enabled_display_names.append(self.module_mapping[module_key]['display_name'])
        
        print(f"🔍 启用检测模块: {', '.join(enabled_display_names)}")
        
        # 显示各模块关键配置
        for module_key in self.enabled_modules:
            if module_key in self.module_mapping:
                module_info = self.module_mapping[module_key]
                config_prefix = module_info['config_prefix']
                
                print(f"\n🔧 {module_info['display_name']} 配置:")
                
                if module_key == 'person_vehicle':
                    print(f"  模型路径: {self.config.get(f'{config_prefix}_engine_path', 'N/A')}")
                    print(f"  目标FPS: {self.config.get(f'{config_prefix}_target_fps', 'N/A')}")
                    print(f"  置信度: {self.config.get(f'{config_prefix}_confidence', 'N/A')}")
                
                elif module_key == 'train_station':
                    print(f"  目标FPS: {self.config.get(f'{config_prefix}_target_fps', 'N/A')}")
                    print(f"  学习率: {self.config.get('bg_learning_rate', 'N/A')}")
                    print(f"  历史帧数: {self.config.get('bg_history', 'N/A')}")
                
                elif module_key == 'foreign_object':
                    print(f"  ROI: {self.config.get(f'{config_prefix}_roi', 'N/A')}")
                    print(f"  最小静止时间: {self.config.get(f'{config_prefix}_min_static_duration', 'N/A')}秒")
                    print(f"  白色阈值: {self.config.get(f'{config_prefix}_threshold', 'N/A')}")
                    print(f"  警报目录: {self.config.get(f'{config_prefix}_alert_dir', 'N/A')}")
        
        print("="*50)
    
    def start_system(self):
        """启动系统 - 使用统一的显示管理器"""
        print("\n🎯 启动多模块检测系统")
        
        try:
            # 第一步：启动所有工作线程（视频捕获和检测）
            print("\n🚀 第一步：启动所有工作线程")
            
            for thread in self.threads:
                if thread is None:
                    continue
                    
                thread_name = thread.name if hasattr(thread, 'name') else thread.__class__.__name__
                print(f"  ▶️ 启动: {thread_name}")
                thread.start()
                time.sleep(0.3)  # 给线程时间初始化
            
            # 等待线程初始化
            print("\n⏳ 等待线程初始化...")
            init_timeout = 10
            start_time = time.time()
            
            while not self.stop_event.is_set() and (time.time() - start_time) < init_timeout:
                # 检查是否有结果产生
                all_results = self.result_manager.get_all_results()
                if len(all_results) >= len(self.enabled_modules):
                    print("✅ 所有检测线程已开始产生结果")
                    break
                
                # 打印当前状态
                print(f"  等待结果... ({len(all_results)}/{len(self.enabled_modules)} 个模块就绪)")
                time.sleep(1.0)
            
            # 第二步：在主线程中运行显示管理器
            print("\n🚀 第二步：启动显示管理器（在主线程中运行）")
            print("⚠️ 注意：显示管理器将在主线程中运行，不要在其他地方调用cv2.waitKey()")
            
            # 创建显示线程（实际上是运行显示循环）
            display_thread = threading.Thread(
                target=self.display_manager.run,
                name="UnifiedDisplayManager",
                daemon=True
            )
            display_thread.start()
            
            print("\n✅ 系统已启动，所有线程正在运行...")
            print("📊 性能监控每2秒更新一次")
            print("🎮 按窗口中的 'q' 键或 ESC 键停止系统")
            
        except Exception as e:
            print(f"❌ 启动系统失败: {e}")
            import traceback
            traceback.print_exc()
    
    def stop_system(self):
        """停止系统"""
        print("\n🛑 停止多模块检测系统...")
        self.stop_event.set()
        
        # 等待所有线程结束
        for thread in self.threads:
            if thread and thread.is_alive():
                thread_name = thread.__class__.__name__
                
                # 获取友好的线程名称
                for module_key, module_info in self.module_mapping.items():
                    if module_info['name'] in thread_name or isinstance(thread, module_info['class']):
                        display_name = module_info['display_name']
                        break
                else:
                    if 'VideoCapture' in thread_name:
                        display_name = '视频捕获'
                    elif 'Display' in thread_name:
                        display_name = '显示'
                    else:
                        display_name = thread_name
                
                thread.join(timeout=2.0)
                print(f"✅ 停止线程: {display_name}")
        
        print("✅ 系统已完全停止")
    
    def run(self):
        """运行系统 - 修改主循环"""
        try:
            # 初始化系统
            if not self.initialize_system():
                print("❌ 系统初始化失败，退出")
                return
            
            # 启动系统
            self.start_system()
            
            # 主线程现在等待停止事件
            print("\n" + "="*50)
            print("🎮 系统运行中...")
            print("="*50)
            
            # 定期更新性能统计
            last_perf_time = time.time()
            
            while not self.stop_event.is_set():
                try:
                    current_time = time.time()
                    
                    # 定期更新性能统计
                    if current_time - last_perf_time >= 2.0:
                        self.update_performance_stats()
                        last_perf_time = current_time
                    
                    # 每30秒检查一次线程状态
                    if current_time % 30 < 0.1:
                        self.check_thread_status()
                    
                    time.sleep(0.1)
                    
                except KeyboardInterrupt:
                    print("\n⏹️ 用户中断")
                    break
                except Exception as e:
                    print(f"⚠️ 主循环异常: {e}")
                    time.sleep(1.0)
                    
        except Exception as e:
            print(f"❌ 系统运行异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_system()
    
    def update_performance_stats(self):
        """更新性能统计"""
        try:
            stats_collected = {}
            
            for thread in self.threads:
                if hasattr(thread, 'get_performance_stats'):
                    try:
                        stats = thread.get_performance_stats()
                        if stats and 'module' in stats:
                            module_name = stats['module']
                            
                            # 确保模块名称统一
                            if 'person' in module_name.lower():
                                module_name = 'personvehicledetection'
                            elif 'foreign' in module_name.lower():
                                module_name = 'foreignobjectdetection'
                            elif 'video' in module_name.lower():
                                module_name = 'videocapture'
                            
                            # 更新到结果管理器
                            self.result_manager.update_performance(module_name, stats)
                            
                            # 收集性能统计用于显示
                            if 'fps' in stats:
                                stats_collected[module_name] = {
                                    'fps': stats['fps'],
                                    'processing_time': stats.get('avg_processing_time', 0) * 1000
                                }
                                
                        elif stats:
                            print(f"⚠️ 统计缺少模块名: {thread.__class__.__name__} - {stats.keys()}")
                            
                    except Exception as e:
                        print(f"⚠️ 获取性能统计失败 {thread.__class__.__name__}: {e}")
            
            # 显示性能摘要
            if stats_collected:
                print("\n📊 性能摘要:")
                for module, stats in stats_collected.items():
                    print(f"  {module}: FPS={stats['fps']:.1f}, 处理时间={stats['processing_time']:.1f}ms")
                
        except Exception as e:
            print(f"⚠️ 更新性能统计异常: {e}")
    
    def check_thread_status(self):
        """检查线程状态"""
        alive_count = sum(1 for thread in self.threads if thread and thread.is_alive())
        total_count = len(self.threads)
        
        if alive_count < total_count:
            print(f"\n⚠️ 线程状态: {alive_count}/{total_count} 个线程运行中")
            
            for i, thread in enumerate(self.threads):
                if thread:
                    status = "运行" if thread.is_alive() else "停止"
                    
                    # 获取友好的线程名称
                    thread_name = thread.__class__.__name__
                    for module_key, module_info in self.module_mapping.items():
                        if module_info['name'] in thread_name or isinstance(thread, module_info['class']):
                            display_name = module_info['display_name']
                            break
                    else:
                        if 'VideoCapture' in thread_name:
                            display_name = '视频捕获'
                        elif 'Display' in thread_name:
                            display_name = '显示'
                        else:
                            display_name = thread_name
                    
                    print(f"  - {display_name}: [{status}]")