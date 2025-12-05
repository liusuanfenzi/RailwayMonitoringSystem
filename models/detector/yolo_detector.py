# models/yolo_detector.py
import numpy as np
import os
import cv2
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 添加这一行，自动初始化CUDA上下文
from collections import deque

class YOLODetector:
    """真正的TensorRT YOLO目标检测器"""

    def __init__(self, model_path='yolov8n.engine', conf_threshold=0.5,
                 target_classes=None, use_gpu=True):
        """
        TensorRT YOLO检测器初始化

        Args:
            model_path: TensorRT引擎文件路径 (.engine)
            conf_threshold: 置信度阈值
            target_classes: 目标类别列表
            use_gpu: 是否使用GPU
        """

        print("🎯 使用pycuda.autoinit自动创建CUDA上下文")
        
        # 添加调试信息
        try:
            # 获取当前CUDA上下文（用于调试）
            ctx = cuda.Context.get_current()
            print(f"✅ 当前线程已获取CUDA上下文: {ctx}")
        except Exception as e:
            print(f"⚠️ 获取CUDA上下文时出错: {e}")
        # ------------------------------------------------
        
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes or ['person', 'car']
        self.use_gpu = use_gpu
        self.input_shape = (1, 3, 640, 640)
        self.input_size = 640
        
        # COCO类别名称 (保持不变)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # 创建目标类别的ID映射
        self.target_class_ids = []
        for class_name in self.target_classes:
            if class_name in self.class_names:
                self.target_class_ids.append(self.class_names.index(class_name))
        
        # 性能监控
        self.inference_times = deque(maxlen=30)
        self.frame_count = 0
        
        # ROI相关
        self.roi_points = None
        self.roi_active = False
        
        # 标记已清理状态
        self._cleaned = False
        self._context_created_by_autoinit = True  # 标记上下文创建方式
        
        # 加载TensorRT引擎
        self.engine = self._load_tensorrt_engine(model_path)
        
        print(f"✅ TensorRT YOLO检测器初始化完成")
        print(f"🎯 目标类别: {self.target_classes}")
        print(f"🎯 目标类别ID: {self.target_class_ids}")

        # if use_gpu:
        #     self.verify_cuda_context()

    def verify_cuda_context(self):
        """验证CUDA上下文是否正确设置"""
        try:
            import pycuda.driver as cuda
            import traceback
            
            print("\n🔍 验证CUDA上下文状态:")
            
            # 1. 检查当前上下文
            try:
                ctx = cuda.Context.get_current()
                print(f"  ✅ 当前CUDA上下文: {ctx}")
            except cuda.Error as e:
                print(f"  ❌ 无法获取当前CUDA上下文: {e}")
                return False
            
            # 2. 检查设备信息
            try:
                device = ctx.get_device()
                print(f"  ✅ 当前设备: {device.name()}")
                print(f"  ✅ 设备计算能力: {device.compute_capability()}")
            except Exception as e:
                print(f"  ⚠️ 无法获取设备信息: {e}")
            
            # 3. 检查GPU内存分配
            if hasattr(self, 'input_gpu') and self.input_gpu:
                print(f"  ✅ 输入GPU内存已分配: {int(self.input_gpu)}")
            else:
                print(f"  ❌ 输入GPU内存未分配")
                return False
                
            if hasattr(self, 'output_gpu') and self.output_gpu:
                print(f"  ✅ 输出GPU内存已分配: {int(self.output_gpu)}")
            else:
                print(f"  ❌ 输出GPU内存未分配")
                return False
            
            # 4. 检查CUDA流
            if hasattr(self, 'stream') and self.stream:
                print(f"  ✅ CUDA流已创建: {self.stream}")
            else:
                print(f"  ❌ CUDA流未创建")
                return False
            
            # 5. 检查TensorRT上下文
            if hasattr(self, 'context') and self.context:
                print(f"  ✅ TensorRT上下文已创建")
            else:
                print(f"  ❌ TensorRT上下文未创建")
                return False
            
            print("  🎉 所有CUDA和TensorRT资源验证通过！")
            return True
            
        except Exception as e:
            print(f"  ❌ 验证过程中发生错误: {e}")
            traceback.print_exc()
            return False

    def _load_tensorrt_engine(self, engine_path):
        """加载TensorRT引擎 - 确保在正确的CUDA上下文中执行"""
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT引擎文件不存在: {engine_path}")
        
        print(f"📁 加载TensorRT引擎: {engine_path}")
        
        # try:
        #     # 验证当前是否有CUDA上下文（用于调试）
        #     import pycuda.driver as cuda
        #     try:
        #         ctx = cuda.Context.get_current()
        #         print(f"🔍 TensorRT引擎加载时的CUDA上下文: {ctx}")
        #     except Exception as e:
        #         print(f"⚠️ 警告: 当前线程没有CUDA上下文: {e}")
        #         print("🔄 正在尝试通过pycuda操作自动创建上下文...")
        # except ImportError:
        #     pass
        
        # 初始化TensorRT
        logger = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        # 创建执行上下文
        self.context = engine.create_execution_context()
        
        # 分配输入输出内存 - 这会在当前CUDA上下文中分配内存
        self._allocate_buffers(engine)
        
        print("✅ TensorRT引擎加载成功")
        
        # 验证引擎绑定是否成功
        if hasattr(self, 'bindings') and self.bindings:
            print(f"🔗 引擎绑定完成，binding数量: {len(self.bindings)}")
        
        return engine

    def _allocate_buffers(self, engine):
        """分配GPU内存缓冲区 - 确保在正确的CUDA上下文中执行"""
        try:
            # 验证当前线程是否有CUDA上下文
            import pycuda.driver as cuda
            
            # 添加调试信息
            # try:
            #     ctx = cuda.Context.get_current()
            #     device = ctx.get_device()
            #     print(f"🔍 内存分配时CUDA上下文: {ctx}")
            #     print(f"🔍 当前设备: {device.name()}")
            # except Exception as e:
            #     print(f"⚠️ 无法获取CUDA上下文信息: {e}")
                # 这可能是正常的，因为有些系统可能不提供详细的上下文信息
                
        except ImportError:
            print("⚠️ 无法导入pycuda.driver，内存分配可能失败")
            return
        
        # 输入配置
        self.input_shape = (1, 3, 640, 640)
        self.input_size = int(np.prod(self.input_shape))
        
        # 输出配置
        self.output_shape = (84, 8400)
        self.output_size = int(np.prod(self.output_shape))
        
        # print(f"📊 内存分配信息:")
        # print(f"  输入形状: {self.input_shape}, 大小: {self.input_size} 元素")
        # print(f"  输出形状: {self.output_shape}, 大小: {self.output_size} 元素")
        
        try:
            # 分配GPU内存 - 这些操作必须在有效的CUDA上下文中执行
            print("🔄 分配输入GPU内存...")
            self.input_gpu = cuda.mem_alloc(self.input_size * 4)  # float32 (4字节)
            print(f"  输入内存地址: {int(self.input_gpu)}")
            
            print("🔄 分配输出GPU内存...")
            self.output_gpu = cuda.mem_alloc(self.output_size * 4)  # float32 (4字节)
            print(f"  输出内存地址: {int(self.output_gpu)}")
            
            # 创建bindings列表
            self.bindings = [int(self.input_gpu), int(self.output_gpu)]
            
            # 尝试设置张量地址（TensorRT 8.5+ API）
            try:
                print("🔄 设置TensorRT张量地址...")
                self.context.set_tensor_address("images", int(self.input_gpu))
                self.context.set_tensor_address("output0", int(self.output_gpu))
                print("✅ 使用TensorRT 8.5+ set_tensor_address API")
            except Exception as e:
                print(f"⚠️ 设置张量地址时出错，使用传统bindings方法: {e}")
                # 对于旧版本TensorRT，bindings已经足够
            
            # 创建CUDA流
            print("🔄 创建CUDA流...")
            self.stream = cuda.Stream()
            print(f"  CUDA流创建成功: {self.stream}")
            
            print("✅ GPU内存分配完成")
            
            # 验证内存分配
            total_memory = (self.input_size + self.output_size) * 4 / 1024 / 1024  # MB
            print(f"📊 分配的GPU内存总量: {total_memory:.2f} MB")
            
        except Exception as e:
            print(f"❌ GPU内存分配失败: {e}")
            print("⚠️ 可能的原因:")
            print("   1. 没有可用的GPU")
            print("   2. GPU内存不足")
            print("   3. CUDA上下文未正确初始化")
            print("   4. pycuda安装有问题")
            
            # 清理已分配的资源
            if hasattr(self, 'input_gpu') and self.input_gpu:
                try:
                    self.input_gpu.free()
                except:
                    pass
            
            if hasattr(self, 'output_gpu') and self.output_gpu:
                try:
                    self.output_gpu.free()
                except:
                    pass
            
            raise RuntimeError(f"GPU内存分配失败: {e}")

    def set_roi(self, points):
        """设置ROI区域"""
        if len(points) == 2:
            self.roi_points = points
            self.roi_active = True
            print(f"🎯 设置检测ROI: {points}")
        else:
            print("⚠️ ROI点必须是两个点 [(x1,y1), (x2,y2)]")

    def disable_roi(self):
        """禁用ROI检测"""
        self.roi_active = False
        print("🔓 禁用ROI检测")

    def detect(self, frame):
        """
        TensorRT目标检测

        Args:
            frame: 输入图像

        Returns:
            detections: 检测结果 [[x1, y1, x2, y2, confidence, class_id], ...]
        """
        start_time = time.time()
        self.frame_count += 1

        try:
            if self.roi_active and self.roi_points:
                detections = self._detect_in_roi(frame)
            else:
                detections = self._detect_full_frame(frame)

            # 性能统计
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            return detections

        except Exception as e:
            print(f"❌ TensorRT检测失败: {e}")
            return np.empty((0, 6), dtype=np.float32)

    def _detect_in_roi(self, frame):
        """ROI区域检测"""
        x1, y1 = self.roi_points[0]
        x2, y2 = self.roi_points[1]

        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 6), dtype=np.float32)

        roi_frame = frame[y1:y2, x1:x2]
        
        if roi_frame.size == 0:
            return np.empty((0, 6), dtype=np.float32)

        # 在ROI区域进行推理
        detections = self._inference(roi_frame)
        
        # 坐标映射回原图
        if len(detections) > 0:
            detections[:, 0] += x1  # x1
            detections[:, 1] += y1  # y1  
            detections[:, 2] += x1  # x2
            detections[:, 3] += y1  # y2
            
        return detections

    def _detect_full_frame(self, frame):
        """全图检测"""
        return self._inference(frame)

    def _inference(self, frame):
        """TensorRT推理核心函数"""
        # 使用优化版推理
        return self._inference_optimized(frame)

    def _preprocess(self, frame):
        """图像预处理 - 使用640x640"""
        # 调整大小到640x640
        img = cv2.resize(frame, (640, 640))  # 改回640x640
        
        # 归一化: 0-255 -> 0-1
        img = img.astype(np.float32) / 255.0
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        
        # 添加batch维度并确保内存连续
        blob = np.ascontiguousarray(img).reshape(1, 3, 640, 640)  # 改回640x640
        
        return blob

    def _postprocess(self, outputs, orig_shape):
        """完全向量化的后处理 - 参考手势检测代码"""
        start_time = time.time()
        
        # 1. 输出重塑 (0.1ms)
        predictions = outputs.transpose(1, 0)  # [8400, 84]
        
        # 2. 一次性提取所有分数和类别 (1ms)
        scores = predictions[:, 4:84]
        max_scores = np.max(scores, axis=1)
        max_class_ids = np.argmax(scores, axis=1)
        
        # 3. 向量化过滤 (1ms)
        conf_mask = max_scores > self.conf_threshold
        class_mask = np.isin(max_class_ids, self.target_class_ids)
        valid_mask = conf_mask & class_mask
        
        if not np.any(valid_mask):
            return np.empty((0, 6), dtype=np.float32)
        
        # 4. 提取有效检测
        valid_indices = np.where(valid_mask)[0]
        boxes = predictions[valid_indices, :4]
        scores = max_scores[valid_indices]
        class_ids = max_class_ids[valid_indices]
        
        # 5. 向量化坐标转换 (2ms)
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0
        
        # 中心坐标转角点坐标
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        x1 = (x_center - width / 2) * scale_x
        y1 = (y_center - height / 2) * scale_y
        x2 = (x_center + width / 2) * scale_x
        y2 = (y_center + height / 2) * scale_y
        
        # 边界检查
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        # 6. 组装框
        boxes_array = np.column_stack([x1, y1, x2, y2])
        
        # 7. 快速NMS - 使用OpenCV (2-5ms)
        indices = self._fast_nms_opencv(boxes_array, scores)
        
        # 8. 最终结果
        result = np.column_stack([
            boxes_array[indices, 0], boxes_array[indices, 1],
            boxes_array[indices, 2], boxes_array[indices, 3],
            scores[indices], class_ids[indices]
        ])
        # 每100帧输出一次，而不是每帧
        if self.frame_count % 100 == 0:
            print(f"🔧 向量化后处理耗时: {(time.time()-start_time)*1000:.1f}ms")
        return result.astype(np.float32)

    def _fast_nms_opencv(self, boxes, scores, iou_threshold=0.45):
        """使用OpenCV的快速NMS"""
        if len(boxes) == 0:
            return []
        
        # 转换为(x, y, w, h)格式
        boxes_wh = boxes.copy()
        boxes_wh[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
        boxes_wh[:, 3] = boxes[:, 3] - boxes[:, 1]  # h
        
        # 使用OpenCV的NMS (C++实现，很快)
        indices = cv2.dnn.NMSBoxes(
            boxes_wh.tolist(), 
            scores.tolist(), 
            self.conf_threshold, 
            iou_threshold
        )
        
        return indices.flatten() if len(indices) > 0 else []

    def get_performance_stats(self):
        """获取性能统计"""
        if not self.inference_times:
            return {"avg_inference_time": 0, "avg_fps": 0, "total_frames": 0}
            
        avg_time = sum(self.inference_times) / len(self.inference_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_inference_time': avg_time,
            'avg_fps': avg_fps,
            'total_frames': self.frame_count
        }

    def cleanup(self):
        """清理资源 - 改回autoinit方式"""
        if self._cleaned:
            return
        
        print("🧹 开始清理TensorRT资源（autoinit方式）...")
        
        try:
            # 1. 清理GPU内存和CUDA流
            if hasattr(self, 'input_gpu') and self.input_gpu:
                try:
                    self.input_gpu.free()
                    self.input_gpu = None
                    print("✅ input_gpu 已释放")
                except Exception as e:
                    print(f"⚠️ 释放input_gpu时出错: {e}")
            
            if hasattr(self, 'output_gpu') and self.output_gpu:
                try:
                    self.output_gpu.free()
                    self.output_gpu = None
                    print("✅ output_gpu 已释放")
                except Exception as e:
                    print(f"⚠️ 释放output_gpu时出错: {e}")
            
            if hasattr(self, 'stream') and self.stream:
                try:
                    self.stream.synchronize()
                    # 对于autoinit方式，我们不需要销毁流，但可以置为None
                    self.stream = None
                    print("✅ CUDA流 已同步")
                except Exception as e:
                    print(f"⚠️ 同步CUDA流时出错: {e}")
            
            # 2. 清理TensorRT资源
            if hasattr(self, 'context') and self.context:
                try:
                    del self.context
                    self.context = None
                    print("✅ TensorRT上下文已清理")
                except Exception as e:
                    print(f"⚠️ 清理TensorRT上下文时出错: {e}")
            
            if hasattr(self, 'engine') and self.engine:
                try:
                    del self.engine
                    self.engine = None
                    print("✅ TensorRT引擎已清理")
                except Exception as e:
                    print(f"⚠️ 清理TensorRT引擎时出错: {e}")
            
            # 3. 重要：对于autoinit方式，我们不需要手动弹出CUDA上下文
            # autoinit会自动管理上下文的生命周期
            # 但我们可以打印一些调试信息
            try:
                ctx = cuda.Context.get_current()
                print(f"🔍 清理后当前CUDA上下文: {ctx}")
            except:
                print("🔍 清理后无法获取CUDA上下文（可能已被释放）")
            
            self._cleaned = True
            print("✅ TensorRT资源已完全清理（autoinit方式）")
            
        except Exception as e:
            print(f"❌ TensorRT资源清理失败: {e}")
            import traceback
            traceback.print_exc()
            self._cleaned = True

    def _inference_optimized(self, frame):
        """优化版推理 - 添加性能监控"""
        import time
        
        # 性能计时
        preprocess_time = 0
        inference_time = 0
        postprocess_time = 0
        
        # 预处理
        preprocess_start = time.time()
        input_blob = self._preprocess(frame)
        preprocess_time = time.time() - preprocess_start
        
        # 执行推理
        inference_start = time.time()
        cuda.memcpy_htod_async(self.input_gpu, input_blob, self.stream)
        
        # 使用最快的执行方法
        try:
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        except AttributeError:
            try:
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            except AttributeError:
                self.context.execute_async(batch_size=1, bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 获取输出
        host_output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(host_output, self.output_gpu, self.stream)
        self.stream.synchronize()
        inference_time = time.time() - inference_start
        
        # 后处理
        postprocess_start = time.time()
        detections = self._postprocess(host_output, frame.shape)
        postprocess_time = time.time() - postprocess_start
        
        # 性能日志（每30帧输出一次）
        if self.frame_count % 30 == 0:
            print(f"⏱️ YOLO性能统计 (最近30帧):")
            print(f"  预处理: {preprocess_time*1000:.1f}ms")
            print(f"  推理: {inference_time*1000:.1f}ms") 
            print(f"  后处理: {postprocess_time*1000:.1f}ms")
            print(f"  总时间: {(preprocess_time+inference_time+postprocess_time)*1000:.1f}ms, "
                  f"FPS: {1/(preprocess_time+inference_time+postprocess_time):.1f}")
        
        return detections