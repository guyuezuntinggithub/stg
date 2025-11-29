from enum import IntEnum, auto
import numpy as np
import struct
import utils.utils as utils
from spb import SPB

class P(IntEnum):
        x = 0
        y = auto()
        z = auto()
        nx = auto()
        ny = auto()
        nz = auto()
        f_dc_0 = auto()
        f_dc_1 = auto()
        f_dc_2 = auto()
        f_rest_0 = auto()
        f_rest_1 = auto()
        f_rest_2 = auto()
        f_rest_3 = auto()
        f_rest_4 = auto()
        f_rest_5 = auto()
        f_rest_6 = auto()
        f_rest_7 = auto()
        f_rest_8 = auto()
        f_rest_9 = auto()
        f_rest_10 = auto()
        f_rest_11 = auto()
        f_rest_12 = auto()
        f_rest_13 = auto()
        f_rest_14 = auto()
        f_rest_15 = auto()
        f_rest_16 = auto()
        f_rest_17 = auto()
        f_rest_18 = auto()
        f_rest_19 = auto()
        f_rest_20 = auto()
        f_rest_21 = auto()
        f_rest_22 = auto()
        f_rest_23 = auto()
        f_rest_24 = auto()
        f_rest_25 = auto()
        f_rest_26 = auto()
        f_rest_27 = auto()
        f_rest_28 = auto()
        f_rest_29 = auto()
        f_rest_30 = auto()
        f_rest_31 = auto()
        f_rest_32 = auto()
        f_rest_33 = auto()
        f_rest_34 = auto()
        f_rest_35 = auto()
        f_rest_36 = auto()
        f_rest_37 = auto()
        f_rest_38 = auto()
        f_rest_39 = auto()
        f_rest_40 = auto()
        f_rest_41 = auto()
        f_rest_42 = auto()
        f_rest_43 = auto()
        f_rest_44 = auto()
        opacity = auto()
        scale_0 = auto()
        scale_1 = auto()
        scale_2 = auto()
        rot_0 = auto()
        rot_1 = auto()
        rot_2 = auto()
        rot_3 = auto()
        total = auto()

SH_C0 = 0.28209479177387814

class Kernel_3dgs:

    @staticmethod
    def identify(header: str):
        return 'f_rest_0' in header
    
    @staticmethod
    def getParams(ply: np.ndarray):
        xyz = ply[:, P.x:P.z + 1]
        s = ply[:, P.scale_0:P.scale_2 + 1]
        q = ply[:, [P.rot_1, P.rot_2, P.rot_3, P.rot_0]]
        color = ply[:, [P.f_dc_0, P.f_dc_1, P.f_dc_2, P.opacity]]
        d1 = ply[:, [P.f_rest_0,  P.f_rest_15, P.f_rest_30, 
                     P.f_rest_1,  P.f_rest_16, P.f_rest_31, 
                     P.f_rest_2,  P.f_rest_17, P.f_rest_32]]
        d2 = ply[:, [P.f_rest_3,  P.f_rest_18, P.f_rest_33, 
                     P.f_rest_4,  P.f_rest_19, P.f_rest_34, 
                     P.f_rest_5,  P.f_rest_20, P.f_rest_35,
                     P.f_rest_6,  P.f_rest_21, P.f_rest_36, 
                     P.f_rest_7,  P.f_rest_22, P.f_rest_37]]
        d3 = ply[:, [P.f_rest_8,  P.f_rest_23, P.f_rest_38, 
                     P.f_rest_9,  P.f_rest_24, P.f_rest_39, 
                     P.f_rest_10, P.f_rest_25, P.f_rest_40,
                     P.f_rest_11, P.f_rest_26, P.f_rest_41,
                     P.f_rest_12, P.f_rest_27, P.f_rest_42,
                     P.f_rest_13, P.f_rest_28, P.f_rest_43, 
                     P.f_rest_14, P.f_rest_29, P.f_rest_44]]
        
        color[:, 0:3] = np.clip(0.5 + SH_C0 * color[:, 0:3], 0.0, 1.0)
        color[:, 3] = utils.sigmoid(color[:, 3])
        s = np.exp(s)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        
        return xyz, s, q, color, d1, d2, d3
    
    @staticmethod
    def ply2spb(data: bytes, outputPath: bytes, level: int, pad: bool = False):
        pointCount = len(data) // 4 // P.total
        ply = np.array(struct.unpack(f'{len(data) // 4}f', data), dtype=np.float32).reshape(-1, P.total)

        xyz, s, q, color, d1, d2, _ = Kernel_3dgs.getParams(ply)
        cov = Kernel_3dgs.calcCov(s, q).astype(np.float16).view(np.uint32)
        rgba = utils.packRGBA2u32(color)

        with open(outputPath, 'wb') as file:
            file.write(SPB.header("ThreeD", level, pointCount, pad))
            if level == 2:
                file.write(SPB.buffer("Pos6Pad2", 8 * pointCount))
                file.write(SPB.buffer("Cov12Col4", 16 * pointCount))
                file.write(SPB.buffer("SH0", 0))
            elif level == 1:
                file.write(SPB.buffer("Pos6Pad2", 8 * pointCount))
                file.write(SPB.buffer("Cov12Col4", 16 * pointCount))
                file.write(SPB.buffer("SH9Pad3", 12 * pointCount))
            elif level == 0:
                file.write(SPB.buffer("Pos12Pad4", 16 * pointCount))
                file.write(SPB.buffer("Cov12Col4", 16 * pointCount))
                file.write(SPB.buffer("SH24", 24 * pointCount))
            file.write(SPB.endHeader())

            if level == 2:
                file.write(utils.padBack(xyz.astype(np.float16)).flatten().tobytes())
                file.write(np.concatenate([cov, rgba], axis=1).flatten().tobytes())
                if pad:
                    width, height = utils.compute_tex_size(pointCount)
                    zeros = np.zeros([(width * height - pointCount) * 4], dtype=np.uint32)
                    file.write(zeros.flatten().tobytes())
            elif level == 1:
                file.write(utils.padBack(xyz.astype(np.float16)).flatten().tobytes())
                file.write(np.concatenate([cov, rgba], axis=1).flatten().tobytes())
                file.write(utils.padBack(utils.uint8Quantify(d1, -1, 1), 3).flatten().tobytes())
                if pad:
                    width, height = utils.compute_tex_size(pointCount)
                    zeros = np.zeros([(width * height - pointCount) * 3], dtype=np.uint32)
                    file.write(zeros.flatten().tobytes())
            elif level == 0:
                file.write(utils.padBack(xyz).flatten().tobytes())
                file.write(np.concatenate([cov, rgba], axis=1).flatten().tobytes())
                file.write(utils.uint8Quantify(np.concatenate([d1, d2], axis=1), -1, 1).flatten().tobytes())
                if pad:
                    width, height = utils.compute_tex_size(pointCount * 2)    # 2 texel per splat
                    zeros = np.zeros([(width * height - pointCount) * 3], dtype=np.uint32)
                    file.write(zeros.flatten().tobytes())
                
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
