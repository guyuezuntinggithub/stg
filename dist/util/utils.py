import numpy as np
import math
import warnings
import colorsys
import time
import pyvista as pv
from scipy.spatial.distance import pdist


def packRGBA2u32(colors: np.ndarray) -> np.ndarray:
    shape0 = colors.shape[0]
    if colors.shape[1] != 4 or colors.dtype != np.float32:
        raise ValueError("输入数组的形状必须是 (num, 4) 且类型为 float32")

    colors = np.round(colors * 255.0).astype(np.uint8)
    packed_colors = colors.copy().view(dtype=np.uint32)
    return packed_colors.reshape([shape0, 1])

def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

def padBack(x: np.ndarray, num = 1):
    return np.pad(x, ((0, 0), (0, num)), mode='constant', constant_values=0)

def uint8Quantify(x: np.ndarray, min, max):
    return np.clip(np.round((x - min) / (max - min) * 255), 0, 255).astype(np.uint8)

def alignUp(x, alignment):
    return ((x + alignment - 1) // alignment) * alignment

def compute_tex_size(texel_num: int, chunkBased: bool) -> tuple:
    # we wanna pad as less as possible
    # for general usage, width and height are limited to 4096
    if texel_num <= 0:
        return 0, 0

    max_height = max_width = 4096 // 16 if chunkBased else 4096

    if texel_num > max_height * max_width:
        raise ValueError("point num is too large! Should be less or equal to 4096 * 4096!")
    
    # suppose we have i columns of max_height
    for i in range(1, max_width + 1):
        if texel_num <= i * max_height:
            return i, alignUp(texel_num, i) // i

    return 0, 0
    
def alignTo256(ply: np.ndarray, opacityIdx:int, alignment: int = 256) -> np.ndarray:
    num_vertices = ply.shape[0]
    num_to_pad = (alignment - (num_vertices % alignment)) % alignment
    if num_to_pad > 0:
        last_vertex = ply[-1].copy()
        last_vertex[opacityIdx] = -70  # 0-2 are position, -70 is for small opacity
        padding_array = np.tile(last_vertex, (num_to_pad, 1))
        return np.concatenate((ply, padding_array), axis=0)
    else:
        return ply

def create_block_colors_high_contrast(n_points: int, block_size: int = 256) -> np.ndarray:
    """
    使用黄金比例配色法为点云创建高对比度的分块颜色。

    Args:
        n_points: 点的总数。
        block_size: 每个颜色块的大小。

    Returns:
        一个形状为 (n_points, 3) 的 uint8 RGB 颜色数组。
    """
    
    # --- 步骤 1: 使用黄金比例生成高对比度的调色板 ---
    num_blocks = (n_points + block_size - 1) // block_size
    
    # 黄金比例的共轭数
    GOLDEN_RATIO_CONJUGATE = (np.sqrt(5) - 1) / 2
    
    # 初始化一个随机的起始色相
    hue = np.random.rand()
    
    palette_rgb_float = []
    for _ in range(num_blocks):
        # 递增色相
        hue = (hue + GOLDEN_RATIO_CONJUGATE) % 1.0
        
        # 将HSV颜色转换为RGB颜色。
        # 我们保持饱和度(S)和亮度(V)较高，以获得鲜艳的颜色
        saturation = 0.85
        value = 0.9
        rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)
        palette_rgb_float.append(rgb_float)
    
    # 将 [0, 1] 范围的浮点数颜色转换为 [0, 255] 的 uint8 格式
    palette_uint8 = (np.array(palette_rgb_float) * 255).astype(np.uint8)
    
    # --- 步骤 2: 将颜色分配给每个点 ---
    # 这部分逻辑和之前一样
    block_indices = np.arange(n_points) // block_size
    colors = palette_uint8[block_indices]

    return colors
