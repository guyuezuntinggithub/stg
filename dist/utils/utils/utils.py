import numpy as np
import math
import warnings

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

def compute_tex_size(texel_num: int) -> tuple:
    if texel_num <= 0:
        return 0, 0

    log2_texel_num = math.ceil(math.log2(texel_num))

    # Clamp to the maximum texture size (4096*4096 = 2^24)
    if log2_texel_num > 24:
        warnings.warn(f"texel_num {texel_num} exceeds maximum 4096*4096 and was clamped.")
        log2_texel_num = 24

    if log2_texel_num % 2 == 0:
        side_length = 2 ** (log2_texel_num / 2)
        return side_length, side_length
    else:
        height = 2 ** (log2_texel_num // 2)
        width = height * 2
        return width, height