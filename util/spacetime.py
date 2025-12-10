from enum import IntEnum, auto
import numpy as np
import struct
import utils as utils
import time
from hilbertcurve.hilbertcurve import HilbertCurve
from pygltflib import *
import math
import pyvista as pv
from scipy.spatial.distance import pdist
import keyboard
from threeD import Kernel_3dgs

P = {
    'x': -1,
    'y': -1,
    'z': -1,
    'trbf_center': -1,
    'trbf_scale': -1,
    'nx': -1,
    'ny': -1,
    'nz': -1,
    'motion_0': -1,
    'motion_1': -1,
    'motion_2': -1,
    'motion_3': -1,
    'motion_4': -1,
    'motion_5': -1,
    'motion_6': -1,
    'motion_7': -1,
    'motion_8': -1,
    'f_dc_0': -1,
    'f_dc_1': -1,
    'f_dc_2': -1,
    'opacity': -1,
    'scale_0': -1,
    'scale_1': -1,
    'scale_2': -1,
    'rot_0': -1,
    'rot_1': -1,
    'rot_2': -1,
    'rot_3': -1,
    'omega_0': -1,
    'omega_1': -1,
    'omega_2': -1,
    'omega_3': -1,
    'total': -1,
}

class Kernel_spacetime:

    @staticmethod
    def identify(headerLines: str):
        for key in P:
            P[key] = -1
        cnter = 0
        for line in headerLines:
            if line[0] == 'property' and line[1] == 'float':
                key = line[2]
                if key in P:
                    P[key] = cnter
                    cnter = cnter + 1
                else:
                    return False
        P['total'] = cnter
        return True
    
    @staticmethod
    def getParams(data: bytes):
        arr = np.frombuffer(data, dtype=np.float32)

        # 打印数组元素个数
        print("Number of float32 elements:", arr.size)
        print("total:", P['total'])
        ply = np.frombuffer(data, dtype=np.float32).reshape([-1, P['total']])
        ply = utils.alignTo256(ply, P['opacity'], 256)

        xyz = ply[:, [P['x'], P['y'], P['z']]]
        motion1 = ply[:, [P['motion_0'],  P['motion_1'], P['motion_2']]]
        motion2 = ply[:, [P['motion_3'],  P['motion_4'], P['motion_5']]]
        motion3 = ply[:, [P['motion_6'],  P['motion_7'], P['motion_8']]]
        tc = ply[:, P['trbf_center']:P['trbf_center'] + 1]
        s = ply[:, [P['scale_0'], P['scale_1'], P['scale_2']]]
        ts = ply[:, P['trbf_scale']:P['trbf_scale'] + 1]
        q = ply[:, [P['rot_1'], P['rot_2'], P['rot_3'], P['rot_0']]]
        omega = ply[:, [P['omega_1'], P['omega_2'], P['omega_3'], P['omega_0']]]
        color = ply[:, [P['f_dc_0'], P['f_dc_1'], P['f_dc_2'], P['opacity']]]
        
        color[:, 3] = utils.sigmoid(color[:, 3])
        # rgb value may exceed 1.0, hack: clamp to 6.0
        color = np.clip(color, 0.0, 6.0)
        s = np.exp(s)
        ts = np.exp(-ts)**2
        # even with spacetime gaussian, ignoring omega seems to be no harm for quality
        # so we directly normalize q
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        
        return xyz, motion1, motion2, motion3, tc, s, ts, q, color
                
    @staticmethod
    def calcCov(s: np.ndarray, q: np.ndarray):
        res = np.zeros([s.shape[0], 6], dtype=np.float32)

        x = q[:, 0:1]
        y = q[:, 1:2]
        z = q[:, 2:3]
        w = q[:, 3:4]

        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        rot = np.zeros((q.shape[0], 3, 3), dtype=np.float32)

        rot[:, 0, 0] = 1.0 - 2.0 * (yy + zz).flatten()
        rot[:, 0, 1] = 2.0 * (xy + wz).flatten()
        rot[:, 0, 2] = 2.0 * (xz - wy).flatten()
        rot[:, 1, 0] = 2.0 * (xy - wz).flatten()
        rot[:, 1, 1] = 1.0 - 2.0 * (xx + zz).flatten()
        rot[:, 1, 2] = 2.0 * (yz + wx).flatten()
        rot[:, 2, 0] = 2.0 * (xz + wy).flatten()
        rot[:, 2, 1] = 2.0 * (yz - wx).flatten()
        rot[:, 2, 2] = 1.0 - 2.0 * (xx + yy).flatten()

        ss = np.zeros((s.shape[0], 3, 3), dtype=np.float32)
        ss[:, 0, 0] = (s[:, 0]**2)
        ss[:, 1, 1] = (s[:, 1]**2)
        ss[:, 2, 2] = (s[:, 2]**2)

        cov3d = np.transpose(rot, (0, 2, 1)) @ ss @ rot

        res[:, 0] = cov3d[:, 0, 0]
        res[:, 1] = cov3d[:, 0, 1]
        res[:, 2] = cov3d[:, 0, 2]
        res[:, 3] = cov3d[:, 1, 1]
        res[:, 4] = cov3d[:, 1, 2]
        res[:, 5] = cov3d[:, 2, 2]
        return res

    @staticmethod
    def reorder(params, type):
        xyz, motion1, motion2, motion3, tc, s, ts, q, color = params

        xyzt = np.concatenate([xyz, tc], axis=1).copy()

        # use time as the fourth dimension may even make results worse
        # use 3dgs methods for now before more tests
        if type == 'Morton':
            sort_indices = Kernel_3dgs.z_order_sort(xyz)
        elif type == 'Hilbert':
            sort_indices = Kernel_3dgs.hilbert_curve_sort(xyz)
        
        xyz = xyz[sort_indices]
        motion1 = motion1[sort_indices]
        motion2 = motion2[sort_indices]
        motion3 = motion3[sort_indices]
        tc = tc[sort_indices]
        s = s[sort_indices]
        ts = ts[sort_indices]
        q = q[sort_indices]
        color = color[sort_indices]
        return xyz, motion1, motion2, motion3, tc, s, ts, q, color
    
    @staticmethod
    def z_order_sort(xyzt: np.ndarray, time_weight: float = 1.0) -> np.ndarray:
        """
        沿Z序（莫顿）曲线对4D点（xyzt）进行排序。

        此方法将4D坐标映射到1D莫顿码，并根据这些编码进行排序。
        它提供了良好的空间局部性，将邻近的点在排序后的列表中组合在一起。
        
        Args:
            xyzt: 一个形状为 (N, 4) 的numpy数组，代表xyzt坐标。
            time_weight: 一个浮点数，用于调整时间维度在排序中的权重。
                         大于1会增加时间的重要性，小于1会降低其重要性。

        Returns:
            一个形状为 (N,) 的numpy数组，包含可以对原始点数组进行排序的索引。
        """
        start_time = time.time()

        # --- 步骤 A: 加权、归一化和量化 ---
        # 【修改】创建一个点的副本以进行加权，避免修改原始数据
        xyzt_weighted = xyzt.copy()
        xyzt_weighted[:, 3] *= time_weight # 将时间维度乘以权重

        min_coords = xyzt_weighted.min(axis=0)
        max_coords = xyzt_weighted.max(axis=0)
        
        scale = (max_coords - min_coords).max()
        if scale == 0:
            return np.arange(len(xyzt))

        normalized_xyzt = (xyzt_weighted - min_coords) / scale

        p = 16
        max_int_val = (1 << p) - 1
        int_coords = (normalized_xyzt * max_int_val).astype(np.uint64)

        # --- 步骤 B: 计算莫顿码 ---
        def spread_bits(coord: np.ndarray) -> np.ndarray:
            """ 将16位整数的位扩展开，为四维交错做准备。 """
            x = coord & 0xFFFF
            x = (x | (x << 24)) & 0x000000FF000000FF
            x = (x | (x << 12)) & 0x000F000F000F000F
            x = (x | (x << 6))  & 0x0303030303030303
            x = (x | (x << 3))  & 0x1111111111111111
            return x
        
        morton_codes = (spread_bits(int_coords[:, 0])       |
                        (spread_bits(int_coords[:, 1]) << 1) |
                        (spread_bits(int_coords[:, 2]) << 2) |
                        (spread_bits(int_coords[:, 3]) << 3))
        
        # --- 步骤 C: 排序 ---
        sort_indices = np.argsort(morton_codes)
        
        end_time = time.time()
        print(f"Morton curve sort (4D, time_weight={time_weight}) done, using: {end_time - start_time:.4f}s")
        
        return sort_indices

    @staticmethod
    def hilbert_curve_sort(xyzt: np.ndarray, time_weight: float = 1.0) -> np.ndarray:
        """
        沿希尔伯特曲线对4D点（xyzt）进行排序。

        此方法将4D坐标映射到它们在希尔伯特曲线上的1D距离。
        希尔伯特曲线通常提供比Z序曲线更好的空间局部性。
        
        Args:
            xyzt: 一个形状为 (N, 4) 的numpy数组，代表xyzt坐标。
            time_weight: 一个浮点数，用于调整时间维度在排序中的权重。
                         大于1会增加时间的重要性，小于1会降低其重要性。

        Returns:
            一个形状为 (N,) 的numpy数组，包含可以对原始点数组进行排序的索引。
        """
        start_time = time.time()

        # --- 步骤 A: 加权、归一化和量化 ---
        # 【修改】创建一个点的副本以进行加权
        xyzt_weighted = xyzt.copy()
        xyzt_weighted[:, 3] *= time_weight # 将时间维度乘以权重

        min_coords = xyzt_weighted.min(axis=0)
        max_coords = xyzt_weighted.max(axis=0)
        scale = (max_coords - min_coords).max()
        if scale == 0:
            return np.arange(len(xyzt))

        normalized_xyzt = (xyzt_weighted - min_coords) / scale

        p = 16 
        n = 4 
        max_int_val = (1 << p) - 1
        int_coords = (normalized_xyzt * max_int_val).astype(np.uint64)

        # --- 步骤 B: 计算希尔伯特曲线距离 ---
        hilbert_curve = HilbertCurve(p, n)
        hilbert_distances = hilbert_curve.distances_from_points(int_coords)

        # --- 步骤 C: 排序 ---
        sort_indices = np.argsort(hilbert_distances)

        end_time = time.time()
        print(f"Hilbert curve sort (4D, time_weight={time_weight}) done, using: {end_time - start_time:.4f}s")

        return sort_indices
    
    @staticmethod
    def toGLB(params, pointCount, name):
        chunk_size = 256
        num_chunks = pointCount // chunk_size

        descriptors, metadata = Kernel_spacetime.prepareForGLB(params)
        texData_len = len(metadata)
        gltf = GLTF2()

        # 2. 创建一个 Buffer 和一个 Sampler
        gltf.buffers.append(Buffer(byteLength=texData_len))
        gltf.samplers.append(Sampler(magFilter=NEAREST, minFilter=NEAREST))

        # 3. 为每个数据块创建 Image 和 Texture
        texture_indices = {}

        for key, descriptor in descriptors.items():
            buffer_view = BufferView(buffer=0, byteOffset=descriptor['offset'], byteLength=descriptor['size'])
            buffer_view_index = len(gltf.bufferViews)
            gltf.bufferViews.append(buffer_view)

            image = Image(
                bufferView=buffer_view_index,
                mimeType="image/vnd.custom-raw",
                extras={
                    "name": key,
                    "format": descriptor['format'],
                    "width": descriptor['width'],
                    "height": descriptor['height']
                }
            )
            image_index = len(gltf.images)
            gltf.images.append(image)

            texture = Texture(sampler=0, source=image_index)
            texture_index = len(gltf.textures)
            gltf.textures.append(texture)
            texture_indices[key] = texture_index

        # 4. 创建一个虚拟材质，并在 extras 中存储纹理映射
        material = Material(
            pbrMetallicRoughness=PbrMetallicRoughness(baseColorFactor=[1.0, 1.0, 1.0, 1.0]),
            extras={"dataTextures": texture_indices}
        )
        gltf.materials.append(material)

        # 5. 创建一个占位符 Mesh, Node, 和 Scene
        placeholder_pos = np.array([[0,0,0]], dtype="float32").tobytes()
        metadata += placeholder_pos

        gltf.buffers[0].byteLength = len(metadata)

        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=texData_len, byteLength=len(placeholder_pos), target=ARRAY_BUFFER))
        gltf.accessors.append(Accessor(bufferView=len(gltf.bufferViews)-1, componentType=FLOAT, count=1, type=VEC3, max=[0,0,0], min=[0,0,0]))

        primitive = Primitive(attributes=Attributes(POSITION=0), material=0, mode=POINTS)
        gltf.meshes.append(Mesh(primitives=[primitive]))

        gltf.nodes.append(Node(
            mesh=0,
            matrix=[
              1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0,
              0, 0, 0, 1
            ],
            extras={
                "gsType": "SPACETIME",
                "name": name,
                "num": pointCount,
                "quality": "medium",
            }
        ))
        gltf.scenes.append(Scene(nodes=[0]))

        # 6. 附加最终的二进制数据
        gltf.set_binary_blob(metadata)

        return gltf

    @staticmethod
    def prepareForGLB(params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        xyz, motion1, motion2, motion3, tc, s, ts, q, color = params
        n = xyz.shape[0]
        chunk_size = 256
        num_chunks = n // chunk_size

        # xyz, Shape: uint32 (num_chunks, chunk_size, 1)
        xyz_chunks = xyz.reshape((num_chunks, chunk_size, 3))
        xyz_min = xyz_chunks.min(axis=1)  # Shape: (num_chunks, 3)
        xyz_max = xyz_chunks.max(axis=1)  # Shape: (num_chunks, 3)
        xyz_metadata = np.concatenate((xyz_min, xyz_max), axis=1) # Shape: (num_chunks, 6)

        xyz_range = xyz_max - xyz_min
        xyz_range[xyz_range == 0] = 1.0 
        normalized_xyz = (xyz_chunks - xyz_min[:, np.newaxis, :]) / xyz_range[:, np.newaxis, :]

        quant_x = np.around(normalized_xyz[..., 0:1] * ((1 << 11) - 1)).astype(np.uint32)
        quant_y = np.around(normalized_xyz[..., 1:2] * ((1 << 10) - 1)).astype(np.uint32)
        quant_z = np.around(normalized_xyz[..., 2:3] * ((1 << 11) - 1)).astype(np.uint32)

        quantized_xyz = (quant_z << 21) | (quant_y << 11) | quant_x

        # q, Shape: uint8 (num_chunks, chunk_size, 4)
        q_chunks = q.reshape((num_chunks, chunk_size, 4))
        q_min = np.array(-1, dtype=np.float32)
        q_max = np.array( 1, dtype=np.float32)

        normalized_q = (q_chunks - q_min) / (q_max - q_min)

        quantized_q = np.around(normalized_q * ((1 << 8) - 1)).astype(np.uint8)

        # color, Shape: uint8 (num_chunks, chunk_size, 4)
        color_chunks = color.reshape((num_chunks, chunk_size, 4))

        color_min = color_chunks.min(axis=1)  # Shape: (num_chunks, 4)
        color_max = color_chunks.max(axis=1)  # Shape: (num_chunks, 4)
        color_metadata = np.concatenate([
            np.concatenate([color_min[:, i:i+1], color_max[:, i:i+1]], axis=1) for i in range(4)
        ], axis=1) # Shape: (num_chunks, 8)

        color_range = color_max - color_min
        color_range[color_range == 0] = 1.0
        normalized_color = (color_chunks - color_min[:, np.newaxis, :]) / color_range[:, np.newaxis, :]

        quantized_color = np.around(normalized_color * ((1 << 8) - 1)).astype(np.uint8)

        # s, Shape: uint8 (num_chunks, chunk_size, 3)
        s_chunks = np.sqrt(s.reshape((num_chunks, chunk_size, 3)))
        s_min = s_chunks.min(axis=(1, 2)) # Shape: (num_chunks,)
        s_max = s_chunks.max(axis=(1, 2)) # Shape: (num_chunks,)
        s_metadata = np.stack((s_min, s_max), axis=1) # Shape: (num_chunks, 2)

        s_range = s_max - s_min
        s_range[s_range == 0] = 1.0
        normalized_s = (s_chunks - s_min[:, np.newaxis, np.newaxis]) / s_range[:, np.newaxis, np.newaxis]

        quantized_s = np.around(normalized_s * ((1 << 8) - 1)).astype(np.uint8)

        # motion1, Shape: uint8 (num_chunks, chunk_size, 3)
        motion1_chunks = motion1.reshape((num_chunks, chunk_size, 3))
        motion1_min = motion1_chunks.min(axis=(1, 2)) # Shape: (num_chunks,)
        motion1_max = motion1_chunks.max(axis=(1, 2)) # Shape: (num_chunks,)
        motion1_metadata = np.stack((motion1_min, motion1_max), axis=1) # Shape: (num_chunks, 2)

        motion1_range = motion1_max - motion1_min
        motion1_range[motion1_range == 0] = 1.0
        normalized_motion1 = (motion1_chunks - motion1_min[:, np.newaxis, np.newaxis]) / motion1_range[:, np.newaxis, np.newaxis]

        quantized_motion1 = np.around(normalized_motion1 * ((1 << 8) - 1)).astype(np.uint8)

        # motion2, Shape: uint8 (num_chunks, chunk_size, 3)
        motion2_chunks = motion2.reshape((num_chunks, chunk_size, 3))
        motion2_min = motion2_chunks.min(axis=(1, 2)) # Shape: (num_chunks,)
        motion2_max = motion2_chunks.max(axis=(1, 2)) # Shape: (num_chunks,)
        motion2_metadata = np.stack((motion2_min, motion2_max), axis=1) # Shape: (num_chunks, 2)

        motion2_range = motion2_max - motion2_min
        motion2_range[motion2_range == 0] = 1.0
        normalized_motion2 = (motion2_chunks - motion2_min[:, np.newaxis, np.newaxis]) / motion2_range[:, np.newaxis, np.newaxis]

        quantized_motion2 = np.around(normalized_motion2 * ((1 << 8) - 1)).astype(np.uint8)

        # motion3, Shape: uint8 (num_chunks, chunk_size, 3)
        motion3_chunks = motion3.reshape((num_chunks, chunk_size, 3))
        motion3_min = motion3_chunks.min(axis=(1, 2)) # Shape: (num_chunks,)
        motion3_max = motion3_chunks.max(axis=(1, 2)) # Shape: (num_chunks,)
        motion3_metadata = np.stack((motion3_min, motion3_max), axis=1) # Shape: (num_chunks, 2)

        motion3_range = motion3_max - motion3_min
        motion3_range[motion3_range == 0] = 1.0
        normalized_motion3 = (motion3_chunks - motion3_min[:, np.newaxis, np.newaxis]) / motion3_range[:, np.newaxis, np.newaxis]

        quantized_motion3 = np.around(normalized_motion3 * ((1 << 8) - 1)).astype(np.uint8)

        # tc_ts
        tc_ts = np.concatenate([tc, ts], axis=1).reshape((num_chunks, chunk_size, 2))
        quantized_tc_ts = tc_ts.astype(np.float16).view(np.uint8)

        # other
        quantized_other = np.concatenate([quantized_motion1, quantized_s[:, :, 0:1], 
                                          quantized_motion2, quantized_s[:, :, 1:2], 
                                          quantized_motion3, quantized_s[:, :, 2:3], 
                                          quantized_tc_ts], axis=-1).view(np.uint32)

        # range, Shape: uint32 (num_chunks, 1, 12)
        quantized_range = np.concatenate((xyz_metadata, s_metadata, 
                                          motion1_metadata, motion2_metadata, motion3_metadata,
                                          np.zeros_like(motion1_metadata),
                                          color_metadata[:, :6],
                                          np.zeros_like(motion1_metadata)), axis=-1).astype(np.float16).view(np.uint32)
        quantized_range = quantized_range.reshape([num_chunks, 1, -1])

        # declare the textures
        quantized_params = {
            'xyz': quantized_xyz,
            'q': quantized_q,
            'color': quantized_color,
            'other': quantized_other,
            'range': quantized_range,
        }

        texture_formats = {
            'xyz': 'R32UI', 
            'q': 'RGBA8', 
            'color': 'RGBA8',
            'other': 'RGBA32UI', 
            'range': 'RGBA32UI'
        }

        # hilbert reorder for 16*16 texel region
        hilbert_order = Kernel_spacetime.generate_hilbert_array(16).flatten()
        for key in quantized_params.keys():
            quantized_param = quantized_params[key]
            if quantized_param.shape[1] == 256:
                quantized_params[key] = quantized_param[:, hilbert_order, :]

        # pad for textures
        chunkWidth, chunkHeight = utils.compute_tex_size(num_chunks, True)
        num_pad = chunkHeight * chunkWidth - num_chunks
        if num_pad > 0:
            for key in quantized_params.keys():
                quantized_param = quantized_params[key]
                shape = quantized_param.shape
                zeros = np.zeros([num_pad, shape[1], shape[2]], dtype=quantized_param.dtype)
                quantized_params[key] = np.concatenate([quantized_param, zeros], axis=0)

        # memory reorder for 16*16 texel region
        for key in quantized_params.keys():
            quantized_param = quantized_params[key]
            localHeight = localWidth = math.isqrt(quantized_param.shape[1])
            quantized_param = quantized_param.reshape([chunkHeight, chunkWidth, localHeight, localWidth, -1])
            quantized_param = quantized_param.transpose(0, 2, 1, 3, 4)
            quantized_param = quantized_param.reshape([chunkHeight * localHeight, chunkWidth * localWidth, -1])
            quantized_params[key] = quantized_param

        # create descriptors and metadata
        # special for u_range
        quantized_params['range'] = quantized_params['range'].reshape((chunkHeight, -1, 4))
        metadata = b""
        descriptors = {}
        offset = 0
        bind = 0
        for key, quantized_param in quantized_params.items():
            size = quantized_param.nbytes
            texture_name = "u_" + key
            descriptor = {
                "offset": offset,
                "size": size,
                "width": quantized_param.shape[1],
                "height": quantized_param.shape[0],
                "format": texture_formats[key],
                "bind": bind,
            }
            bind += 1
            offset += size
            metadata += quantized_param.tobytes()
            descriptors[texture_name] = descriptor
        
        return descriptors, metadata

    @staticmethod
    def generate_hilbert_array(size: int) -> np.ndarray:
        """
        生成一个按希尔伯特曲线顺序填充的2维 NumPy 数组。

        Args:
            size (int): 数组的边长，必须是2的幂 (e.g., 4, 8, 16, 32)。

        Returns:
            np.ndarray: 一个 (size, size) 的数组，其值表示希尔伯特曲线的访问顺序。
        """
        if size <= 0 or (size & (size - 1)) != 0:
            raise ValueError("Size 必须是 2 的正整数次幂。")

        hilbert_array = np.zeros((size, size), dtype=np.int32)

        def _d2xy(d: int, n: int) -> tuple[int, int]:
            """
            将一维希尔伯特距离 d 转换为二维坐标 (x, y)。
            n 是网格的阶数 (size = 2**n)。
            """
            x, y = 0, 0
            s = 1
            while s < n:
                rx = 1 & (d >> 1)
                ry = 1 & (d ^ rx)

                # 旋转和翻转子方块
                if ry == 0:
                    if rx == 1:
                        x = s - 1 - x
                        y = s - 1 - y
                    x, y = y, x

                x += s * rx
                y += s * ry
                d >>= 2
                s <<= 1
            return x, y

        total_points = size * size
        for i in range(total_points):
            # 注意：这里 n 是 size，不是阶数
            x, y = _d2xy(i, size)
            hilbert_array[y, x] = i

        return hilbert_array
    
    @staticmethod
    def analyze_point_blocks(points: np.ndarray, block_size: int = 256):
        """
        分析已分组的点云，统计组内的距离特性。

        Args:
            points: 已排序和分组的点云，形状为 (N, 3)。
            block_size: 每个分组的大小。
        """
        start_time = time.time()

        n_points = len(points)
        num_blocks = (n_points + block_size - 1) // block_size

        threshold_distance = np.sqrt(3)

        # 用于存储每个块的最大内部距离
        all_max_distances = []

        # 遍历每个块
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, n_points)

            current_block = points[start_idx:end_idx]

            # 如果块内少于2个点，则无法计算距离
            if len(current_block) < 2:
                all_max_distances.append(0)
                continue
            
            # 使用 pdist 高效计算块内所有点对的距离
            # pdist 返回一个压缩后的一维距离矩阵
            pairwise_distances = pdist(current_block, metric='euclidean')

            # 找到这个块内的最大距离
            max_dist_in_block = np.max(pairwise_distances)
            all_max_distances.append(max_dist_in_block)

        all_max_distances = np.array(all_max_distances)

        # --- 开始统计 ---
        # 1. 有多少个组
        total_blocks_found = len(all_max_distances)

        # 2. 组内最大距离小于 sqrt(3) 的组的数量和百分比
        compact_blocks_count = np.sum(all_max_distances < threshold_distance)
        percentage_compact = (compact_blocks_count / total_blocks_found) * 100 if total_blocks_found > 0 else 0

        # 3. 组内最大距离的最大值
        overall_max_distance = np.max(all_max_distances) if total_blocks_found > 0 else 0

        analysis_time = time.time() - start_time
        print(f"Analysis done, using {analysis_time:.2f}s")
        print("-" * 25)
        print(f"point num: {n_points:,}")
        print(f"chunk size: {block_size}")
        print(f"chunk num: {total_blocks_found}")
        print("-" * 25)
        print(f"compact threshold: sqrt(3) ≈ {threshold_distance:.4f}")
        print(f"compact chunk num: {compact_blocks_count} / {total_blocks_found}")
        print(f"compact chunk percentage: {percentage_compact:.2f}%")
        print("-" * 25)
        print(f"max radius of chunk: {overall_max_distance:.4f}\n")

    @staticmethod
    def visualize_with_pyvista(params: tuple):
        """
        Visualizes the point cloud in a native PyVista window,
        with support for custom colors and a specific camera view.

        Args:
            points: The (N, 3) NumPy array of XYZ coordinates.
            colors: (Optional) The (N, 3) NumPy array of RGB colors (uint8, 0-255).
        """
        xyz, motion1, motion2, motion3, tc, s, ts, q, color = params
        xyz_ = xyz.copy()
        motion1_ = motion1.copy()
        motion2_ = motion2.copy()
        motion3_ = motion3.copy()
        tc_ = tc.copy()
        ts_ = ts.copy()
        alpha_ = color[:, 3:4].copy()
        currentXYZ = xyz_.copy()
        deltaT = tc_.copy()
        alphaThreshold = 5
        def calcCurrentXYZ(t):
            nonlocal currentXYZ, deltaT, alphaThreshold
            deltaT = tc_ - t
            currentXYZ = xyz_ + (motion1_ + (motion2_ + motion3_ * deltaT) * deltaT) * deltaT
            currentXYZ[(alpha_ * np.exp(-ts_ * deltaT * deltaT) < alphaThreshold / 255).flatten()] = -99999

        print("Creating visualization with PyVista...")
        colors = utils.create_block_colors_high_contrast(xyz_.shape[0], 256)

        plotter = pv.Plotter(window_size=[1280, 720])
        pv.set_plot_theme("dark")

        print("Using custom RGB colors for visualization.")
        plotter.add_mesh(
            pv.PolyData(xyz_),
            scalars=colors,  # Pass the (N, 3) color array here
            rgb=True,        # IMPORTANT: Tell PyVista these are RGB colors
            point_size=5,
            render_points_as_spheres=True
        )

        plotter.show_axes()

        camera_position = [(0, 0, -10), (0, 0, 0), (0, -1, 0)]
        plotter.camera_position = camera_position

        print("\nDisplaying plot window. Press 'q' to close.")
        plotter.show(interactive=True, interactive_update=True)

        frame_index = 0
        num_frames = 21
        pause = False

        while True:
            try:
                if keyboard.is_pressed('q'):
                    print("'q' key pressed, exiting loop.")
                    break
                elif keyboard.is_pressed('space'):
                    pause = not pause
                elif keyboard.is_pressed('a'):
                    frame_index = (frame_index - 1 + num_frames) % num_frames
                    print(f"frame: {frame_index}/{num_frames}")
                elif keyboard.is_pressed('d'):
                    frame_index = (frame_index + 1) % num_frames
                    print(f"frame: {frame_index}/{num_frames}")
                elif keyboard.is_pressed('z'):
                    alphaThreshold = max(0, alphaThreshold - 1)
                    print(f"alphaThreshold: {alphaThreshold}")
                elif keyboard.is_pressed('c'):
                    alphaThreshold = min(120, alphaThreshold + 1)
                    print(f"alphaThreshold: {alphaThreshold}")
            except:
                break

            calcCurrentXYZ(frame_index * 0.05)
            plotter.meshes[0].points = currentXYZ
            plotter.update()
            
            if not pause:
                frame_index = (frame_index + 1) % num_frames

            #time.sleep(0.05) # 约等于20 FPS
        plotter.close()
        print("Window closed. Exiting.")