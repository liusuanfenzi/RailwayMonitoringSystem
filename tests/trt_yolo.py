#!/usr/bin/env python3
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time

ENGINE_PATH = 'yolov8n.engine'   # ① 改成你的路径
INPUT_SHAPE = (1, 3, 640, 640)   # YOLOv8 输入
CLASS_NUM = 80
CONF_THRES = 0.4
NMS_THRES = 0.45

# ------------ TensorRT 引擎初始化 ------------
class TRTYOLO:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        # 分配 GPU/CPU 内存
        self.inp_gpu = cuda.mem_alloc(int(np.prod(INPUT_SHAPE) * 4))
        self.out_gpu = cuda.mem_alloc(int(CLASS_NUM * 8400 * 4))
        self.stream = cuda.Stream()
        self.context.set_tensor_address("images", int(self.inp_gpu))      # 输入 tensor 名
        self.context.set_tensor_address("output0", int(self.out_gpu))     # 输出 tensor 名

    def infer(self, img):
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (640,640), swapRB=True, crop=False)
        blob = np.ascontiguousarray(blob.astype(np.float32))
        cuda.memcpy_htod_async(self.inp_gpu, blob, self.stream)
        # 只需传 stream，绑定已写进 context
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        host_out = np.empty((CLASS_NUM, 8400), dtype=np.float32)
        cuda.memcpy_dtoh_async(host_out, self.out_gpu, self.stream)
        self.stream.synchronize()
        return host_out

# ------------ 后处理：提取框 + NMS ------------
def postprocess(out):
    # out: (84,8400) -> box(4)+obj(1)+cls(80)
    out = out.transpose(1, 0)          # 8400×84
    boxes, scores, cls_ids = [], [], []
    for row in out:
        obj_score = row[4]
        if obj_score < CONF_THRES: continue
        cls = row[5:].argmax()
        cls_score = row[5+cls]
        score = obj_score * cls_score
        if score < CONF_THRES: continue
        x, y, w, h = row[:4]
        left = int(x - w/2)
        top  = int(y - h/2)
        boxes.append([left, top, int(w), int(h)])
        scores.append(float(score))
        cls_ids.append(cls)
    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRES, NMS_THRES)
    return [(boxes[i], scores[i], cls_ids[i]) for i in indices]

# ------------ 主流程 ------------
def main():
    img = np.zeros((480,640,3), np.uint8)
    cv2.putText(img,'HELLO',(100,240),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
    cv2.imwrite('test.jpg', img)
    model = TRTYOLO(ENGINE_PATH)
    img = cv2.imread('test.jpg')               # ② 测试图
    assert img is not None, 'read image fail'
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    t0 = time.time()
    out = model.infer(img_rgb)
    boxes = postprocess(out)
    t1 = time.time()
    print(f'TensorRT inference  {(t1-t0)*1000:.2f} ms')
    print(f'detected objects: {len(boxes)}')

    # 画框
    for (x, y, w, h), score, cls in boxes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, f'{int(cls)} {score:.2f}', (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imwrite('result.jpg', img)
    print('saved -> result.jpg')

if __name__ == '__main__':
    main()