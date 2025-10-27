import cv2
import numpy as np
from pathlib import Path

# ---------- 参数 ----------
img_paths = ['assets/1.png', 'assets/2.png', 'assets/3.png', 'assets/4.png', 'assets/5.png',
             'assets/6.png', 'assets/7.png', 'assets/8.png', 'assets/9.png', 'assets/10.png']   # 你的原始图片路径
out_path = 'grid.jpg'                    # 输出路径
grid_cols = 3                             # 宫格列数
grid_rows = 4                             # 宫格行数
target_w = 400                           # 缩小后统一宽度
# ---------------------------


def resize_keep_aspect(img, target_width):
    """等比缩放，宽度=target_width，高度自动"""
    h, w = img.shape[:2]
    scale = target_width / w
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def build_grid(images, rows, cols):
    """将 images 列表按 rows×cols 拼成一张图，不足用黑图补齐"""
    if not images:
        raise ValueError("images 列表为空")

    # 统一尺寸：以第一张图为基准
    h, w = images[0].shape[:2]
    # 补齐缺失图
    while len(images) < rows * cols:
        images.append(np.zeros((h, w, 3), dtype=np.uint8))

    # 按行拼接
    grid = []
    for r in range(rows):
        start, end = r * cols, (r + 1) * cols
        row = np.hstack(images[start:end])
        grid.append(row)
    return np.vstack(grid)


def main():
    # 1. 读取 + 缩放
    resized = []
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"警告：无法读取 {p}，已跳过")
            continue
        resized.append(resize_keep_aspect(img, target_w))

    if not resized:
        print("没有成功读取任何图片")
        return

    # 2. 宫格拼接
    grid_img = build_grid(resized, grid_rows, grid_cols)

    # 3. 保存
    cv2.imwrite(out_path, grid_img)
    print(f"已保存宫格图：{Path(out_path).resolve()}")


if __name__ == '__main__':
    main()
