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

P = {
    'x': -1,
    'y': -1,
    'z': -1,
    'nx': -1,
    'ny': -1,
    'nz': -1,
    'f_dc_0': -1,
    'f_dc_1': -1,
    'f_dc_2': -1,
    'f_rest_0': -1,
    'f_rest_1': -1,
    'f_rest_2': -1,
    'f_rest_3': -1,
    'f_rest_4': -1,
    'f_rest_5': -1,
    'f_rest_6': -1,
    'f_rest_7': -1,
    'f_rest_8': -1,
    'f_rest_9': -1,
    'f_rest_10': -1,
    'f_rest_11': -1,
    'f_rest_12': -1,
    'f_rest_13': -1,
    'f_rest_14': -1,
    'f_rest_15': -1,
    'f_rest_16': -1,
    'f_rest_17': -1,
    'f_rest_18': -1,
    'f_rest_19': -1,
    'f_rest_20': -1,
    'f_rest_21': -1,
    'f_rest_22': -1,
    'f_rest_23': -1,
    'f_rest_24': -1,
    'f_rest_25': -1,
    'f_rest_26': -1,
    'f_rest_27': -1,
    'f_rest_28': -1,
    'f_rest_29': -1,
    'f_rest_30': -1,
    'f_rest_31': -1,
    'f_rest_32': -1,
    'f_rest_33': -1,
    'f_rest_34': -1,
    'f_rest_35': -1,
    'f_rest_36': -1,
    'f_rest_37': -1,
    'f_rest_38': -1,
    'f_rest_39': -1,
    'f_rest_40': -1,
    'f_rest_41': -1,
    'f_rest_42': -1,
    'f_rest_43': -1,
    'f_rest_44': -1,
    'opacity': -1,
    'scale_0': -1,
    'scale_1': -1,
    'scale_2': -1,
    'rot_0': -1,
    'rot_1': -1,
    'rot_2': -1,
    'rot_3': -1,
    'total': -1,
}

SH_C0 = 0.28209479177387814

class Kernel_3dgs:

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
        ply = np.frombuffer(data, dtype=np.float32).reshape([-1, P['total']])
        ply = utils.alignTo256(ply, P['opacity'], 256)

        xyz = ply[:, P['x']:P['z'] + 1]
        s = ply[:, P['scale_0']:P['scale_2'] + 1]
        q = ply[:, [P['rot_1'], P['rot_2'], P['rot_3'], P['rot_0']]]
        color = ply[:, [P['f_dc_0'], P['f_dc_1'], P['f_dc_2'], P['opacity']]]
        d1 = ply[:, [P['f_rest_0'],  P['f_rest_15'], P['f_rest_30'], 
                     P['f_rest_1'],  P['f_rest_16'], P['f_rest_31'], 
                     P['f_rest_2'],  P['f_rest_17'], P['f_rest_32']]]
        d2 = ply[:, [P['f_rest_3'],  P['f_rest_18'], P['f_rest_33'], 
                     P['f_rest_4'],  P['f_rest_19'], P['f_rest_34'], 
                     P['f_rest_5'],  P['f_rest_20'], P['f_rest_35'],
                     P['f_rest_6'],  P['f_rest_21'], P['f_rest_36'], 
                     P['f_rest_7'],  P['f_rest_22'], P['f_rest_37']]]
        d3 = ply[:, [P['f_rest_8'],  P['f_rest_23'], P['f_rest_38'], 
                     P['f_rest_9'],  P['f_rest_24'], P['f_rest_39'], 
                     P['f_rest_10'], P['f_rest_25'], P['f_rest_40'],
                     P['f_rest_11'], P['f_rest_26'], P['f_rest_41'],
                     P['f_rest_12'], P['f_rest_27'], P['f_rest_42'],
                     P['f_rest_13'], P['f_rest_28'], P['f_rest_43'], 
                     P['f_rest_14'], P['f_rest_29'], P['f_rest_44']]]
        
        # rgb value may exceed 1.0, hack: clamp to 6.0
        color[:, 0:3] = np.clip(0.5 + SH_C0 * color[:, 0:3], 0.0, 6.0)
        color[:, 3] = utils.sigmoid(color[:, 3])
        s = np.exp(s)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        
        return xyz, s, q, color, d1, d2, d3
                
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
        xyz, s, q, color, d1, d2, d3 = params

        if type == 'Morton':
            sort_indices = Kernel_3dgs.z_order_sort(xyz)
        elif type == 'Hilbert':
            sort_indices = Kernel_3dgs.hilbert_curve_sort(xyz)
        
        xyz = xyz[sort_indices]
        s = s[sort_indices]
        q = q[sort_indices]
        color = color[sort_indices]
        d1 = d1[sort_indices]
        d2 = d2[sort_indices]
        d3 = d3[sort_indices]
        return xyz, s, q, color, d1, d2, d3
    
    @staticmethod
    def z_order_sort(points: np.ndarray) -> np.ndarray:
        start_time = time.time()

        # --- 步骤 A: 归一化和量化 ---
        # 莫顿编码作用于非负整数，所以我们首先要将浮点坐标映射到整数空间。
        # 我们将点云归一化到一个单位立方体 [0, 1]^3 中。
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        scale = (max_coords - min_coords).max()

        normalized_points = (points - min_coords) / scale

        # 将归一化的坐标量化为21位整数。
        # 21位 * 3轴 = 63位，可以安全地存放在一个64位整数中。
        # 2**21 - 1 提供了足够高的精度。
        max_int_val = (1 << 21) - 1
        int_coords = (normalized_points * max_int_val).astype(np.uint64)

        # --- 步骤 B: 计算莫顿码 (使用"Magic Bits"高效算法) ---
        def spread_bits(coord: np.ndarray) -> np.ndarray:
            """将一个21位整数的位扩展开，用于交错。"""
            x = coord
            x = (x | (x << 32)) & 0x001f00000000ffff
            x = (x | (x << 16)) & 0x001f0000ff0000ff
            x = (x | (x << 8))  & 0x100f00f00f00f00f
            x = (x | (x << 4))  & 0x10c30c30c30c30c3
            x = (x | (x << 2))  & 0x1249249249249249
            return x
        # 对每个轴的坐标进行位扩展，然后交错合并
        morton_codes = (spread_bits(int_coords[:, 0]) |
                        (spread_bits(int_coords[:, 1]) << 1) |
                        (spread_bits(int_coords[:, 2]) << 2))
        
        # --- 步骤 C: 排序 ---
        # 获取根据莫顿码排序的索引
        sort_indices = np.argsort(morton_codes)
        
        end_time = time.time()
        print(f"Morton curve sort done, using: {end_time - start_time:.2f}s\n")
        
        return sort_indices

    @staticmethod
    def hilbert_curve_sort(points: np.ndarray) -> np.ndarray:
        start_time = time.time()

        # --- 步骤 A: 归一化和量化 (与Z曲线版本相同) ---
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        scale = (max_coords - min_coords).max()

        normalized_points = (points - min_coords) / scale

        # 定义希尔伯特曲线的精度（每个维度上的比特数）。
        # p=16 意味着每个坐标将被映射到 [0, 2^16 - 1] 的整数范围内。
        p = 16 
        n = 3 # 3个维度
        max_int_val = (1 << p) - 1
        int_coords = (normalized_points * max_int_val).astype(np.uint64)

        # --- 步骤 B: 计算希尔伯特曲线距离 (一维索引) ---
        # 1. 创建一个3D、16位精度的希尔伯特曲线对象
        hilbert_curve = HilbertCurve(p, n)

        # 2. 将所有整数坐标点转换为它们在曲线上的距离（一维索引）
        # 这是一个高效的、向量化的操作
        hilbert_distances = hilbert_curve.distances_from_points(int_coords)

        # --- 步骤 C: 排序 ---
        sort_indices = np.argsort(hilbert_distances)

        end_time = time.time()
        print(f"Hilbert curve sort done, using: {end_time - start_time:.2f}s\n")

        return sort_indices
    
    @staticmethod
    def toGLB(params, pointCount, name):
        chunk_size = 256
        num_chunks = pointCount // chunk_size

        descriptors, metadata = Kernel_3dgs.prepareForGLB(params)
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
                "gsType": "ThreeD",
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
        xyz, s, q, color, d1, d2, d3 = params
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

        # range, Shape: uint32 (num_chunks, 1, 8)
        quantized_range = np.concatenate((xyz_metadata, s_metadata,
                                          color_metadata[:, :6],
                                          np.zeros_like(s_metadata)), axis=1).astype(np.float16).view(np.uint32)
        quantized_range = quantized_range.reshape([num_chunks, 1, -1])

        # declare the textures
        quantized_params = {
            'xyz': quantized_xyz,
            'q': quantized_q,
            'color': quantized_color,
            's': quantized_s,
            'range': quantized_range,
        }

        texture_formats = {
            'xyz': 'R32UI', 
            'q': 'RGBA8', 
            'color': 'RGBA8',
            's': 'RGB8', 
            'range': 'RGBA32UI'
        }

        # hilbert reorder for 16*16 texel region
        hilbert_order = Kernel_3dgs.generate_hilbert_array(16).flatten()
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

    def visualize_with_pyvista(params: tuple):
        """
        Visualizes the point cloud in a native PyVista window,
        with support for custom colors and a specific camera view.

        Args:
            points: The (N, 3) NumPy array of XYZ coordinates.
            colors: (Optional) The (N, 3) NumPy array of RGB colors (uint8, 0-255).
        """
        xyz, s, q, color, d1, d2, d3 = params
        points = xyz.copy()
        print("Creating visualization with PyVista...")
        colors = utils.create_block_colors_high_contrast(points.shape[0], 256)

        # 3. Create a plotter object.
        plotter = pv.Plotter(window_size=[1280, 720])
        pv.set_plot_theme("dark")

        # 4. Add the point cloud to the plotter.
        # We check if custom colors were provided.
        # --- Using Custom RGB Colors ---
        print("Using custom RGB colors for visualization.")
        plotter.add_mesh(
            pv.PolyData(points),
            scalars=colors,  # Pass the (N, 3) color array here
            rgb=True,        # IMPORTANT: Tell PyVista these are RGB colors
            point_size=5,
            render_points_as_spheres=True
        )

        # 5. Customize the scene and controls.
        plotter.show_axes()

        # 6. ✅ Set the camera position BEFORE showing the plot.
        # The format is: [(position), (focal_point), (view_up)]
        camera_position = [(0, 0, -10), (0, 0, 0), (0, -1, 0)]
        plotter.camera_position = camera_position

        # 7. Display the interactive rendering window.
        print("\nDisplaying plot window. Press 'q' to close.")
        plotter.show()