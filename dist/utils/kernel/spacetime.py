from enum import IntEnum, auto
import numpy as np
import struct
import utils.utils as utils
from spb import SPB

class P(IntEnum):
    x = 0
    y = auto()
    z = auto()
    trbf_center = auto()
    trbf_scale = auto()
    nx = auto()
    ny = auto()
    nz = auto()
    motion_0 = auto()
    motion_1 = auto()
    motion_2 = auto()
    motion_3 = auto()
    motion_4 = auto()
    motion_5 = auto()
    motion_6 = auto()
    motion_7 = auto()
    motion_8 = auto()
    f_dc_0 = auto()
    f_dc_1 = auto()
    f_dc_2 = auto()
    opacity = auto()
    scale_0 = auto()
    scale_1 = auto()
    scale_2 = auto()
    rot_0 = auto()
    rot_1 = auto()
    rot_2 = auto()
    rot_3 = auto()
    omega_0 = auto()
    omega_1 = auto()
    omega_2 = auto()
    omega_3 = auto()
    total = auto()

class Kernel_spacetime:
    rotRange = [-1.0, 1.3]
    omegaRange = [-0.17, 0.17]
    motionRange = [-5.0, 5.0]
    tcRange = [-0.05, 1.05]
    @staticmethod
    def identify(header: str):
        return 'trbf_center' in header
    
    @staticmethod
    def getParams(ply: np.ndarray):
        xyz = ply[:, P.x:P.z + 1]
        s = ply[:, P.scale_0:P.scale_2 + 1]
        q = ply[:, [P.rot_1, P.rot_2, P.rot_3, P.rot_0]].copy()
        omega = ply[:, [P.omega_1, P.omega_2, P.omega_3, P.omega_0]].copy()
        color = ply[:, [P.f_dc_0, P.f_dc_1, P.f_dc_2, P.opacity]]
        motion1 = ply[:, [P.motion_0,  P.motion_1, P.motion_2]]
        motion2 = ply[:, [P.motion_3,  P.motion_4, P.motion_5]]
        motion3 = ply[:, [P.motion_6,  P.motion_7, P.motion_8]]
        tc = ply[:, P.trbf_center:P.trbf_center + 1]
        ts = ply[:, P.trbf_scale:P.trbf_scale + 1]
        
        color[:, 3] = utils.sigmoid(color[:, 3])
        color = np.clip(color, 0.0, 1.0)
        s = np.exp(s)
        ts = np.exp(-ts)**2
        return xyz, s, q, omega, color, motion1, motion2, motion3, tc, ts
    
    @staticmethod
    def ply2spb(data: bytes, outputPath: bytes, level: int, pad: bool = False):
        K = Kernel_spacetime
        pointCount = len(data) // 4 // P.total
        ply = np.array(struct.unpack(f'{len(data) // 4}f', data), dtype=np.float32).reshape(-1, P.total)

        xyz, s, q, omega, color, motion1, motion2, motion3, tc, ts = Kernel_spacetime.getParams(ply)
        s_ts = np.concatenate([s, ts], axis=1).astype(np.float16).view(np.uint32)
        rgba = utils.packRGBA2u32(color)
        with open(outputPath, 'wb') as file:
            file.write(SPB.header("SPACETIME", level, pointCount, pad))
            if level == 2:
                file.write(SPB.buffer("Pos6Pad2", 8 * pointCount))
                file.write(SPB.buffer("Rot4Omega4Scale6Ts2", 16 * pointCount))
                file.write(SPB.buffer("Motion11Tc1Col4", 0))
            elif level == 1:
                file.write(SPB.buffer("Pos6Pad2", 8 * pointCount))
                file.write(SPB.buffer("Rot8Omega4Col4", 16 * pointCount))
                file.write(SPB.buffer("Motion15Tc1Scale6Ts2", 12 * pointCount))
            elif level == 0:
                file.write(SPB.buffer("Pos12Pad4", 16 * pointCount))
                file.write(SPB.buffer("Rot8Omega8", 16 * pointCount))
                file.write(SPB.buffer("Motion18Scale6Tc2Ts2Col4", 24 * pointCount))
            file.write(SPB.endHeader())

            if level == 2:
                file.write(utils.padBack(xyz.astype(np.float16)).flatten().tobytes())
                q_u32 = utils.uint8Quantify(q, K.rotRange[0], K.rotRange[1]).view(np.uint32)
                omega_u32 = utils.uint8Quantify(omega, K.omegaRange[0], K.omegaRange[1]).view(np.uint32)
                file.write(np.concatenate([q_u32, omega_u32, s_ts], axis=1).flatten().tobytes())
                m1_u32 = utils.padBack(utils.uint8Quantify(motion1, K.motionRange[0], K.motionRange[1]), 1).copy().view(np.uint32)
                m2_u32 = utils.padBack(utils.uint8Quantify(motion2, K.motionRange[0], K.motionRange[1]), 1).copy().view(np.uint32)
                min = np.array([K.motionRange[0], K.motionRange[0], K.motionRange[0], K.tcRange[0]], dtype=np.float32)
                max = np.array([K.motionRange[1], K.motionRange[1], K.motionRange[1], K.tcRange[1]], dtype=np.float32)
                m3_tc = utils.uint8Quantify(np.concatenate([motion3, tc], axis=1), min, max).copy().view(np.uint32)
                file.write(np.concatenate([m1_u32, m2_u32, m3_tc, rgba], axis=1).flatten().tobytes())
                if pad:
                    width, height = utils.compute_tex_size(pointCount)
                    zeros = np.zeros([(width * height - pointCount) * 4], dtype=np.uint32)
                    file.write(zeros.flatten().tobytes())
            elif level == 1:
                file.write(utils.padBack(xyz.astype(np.float16)).flatten().tobytes())
                q_u32 = q.astype(np.float16).view(np.uint32)
                omega_u32 = utils.uint8Quantify(omega, K.omegaRange[0], K.omegaRange[1]).view(np.uint32)
                file.write(np.concatenate([q_u32, omega_u32, rgba], axis=1).flatten().tobytes())
                m12 = np.concatenate([motion1, motion2], axis=1).astype(np.float16).copy().view(np.uint32)
                min = np.array([K.motionRange[0], K.motionRange[0], K.motionRange[0], K.tcRange[0]], dtype=np.float32)
                max = np.array([K.motionRange[1], K.motionRange[1], K.motionRange[1], K.tcRange[1]], dtype=np.float32)
                m3_tc = utils.uint8Quantify(np.concatenate([motion3, tc], axis=1), min, max).copy().view(np.uint32)
                file.write(np.concatenate([m12, m3_tc, s_ts], axis=1).flatten().tobytes())
                if pad:
                    width, height = utils.compute_tex_size(pointCount)
                    zeros = np.zeros([(width * height - pointCount) * 3], dtype=np.uint32)
                    file.write(zeros.flatten().tobytes())
            elif level == 0:
                file.write(utils.padBack(xyz).flatten().tobytes())
                file.write(np.concatenate([q, omega], axis=1).astype(np.float16).view(np.uint32).flatten().tobytes())
                m_tc = np.concatenate([motion1, motion2, motion3, tc], axis=1).astype(np.float16).copy().view(np.uint32)
                file.write(np.concatenate([m_tc, s_ts, rgba], axis=1).flatten().tobytes())
                if pad:
                    width, height = utils.compute_tex_size(pointCount * 2)    # 2 texel per splat
                    zeros = np.zeros([(width * height - pointCount) * 3], dtype=np.uint32)
                    file.write(zeros.flatten().tobytes())

