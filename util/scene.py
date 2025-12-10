import numpy as np
import utils as utils
from threeD import Kernel_3dgs
from spacetime import Kernel_spacetime
from spacetimeqing import Kernel_spacetime_qing
from pathlib import Path
from plyfile import PlyData
import zipfile
import os

class Scene:
    def __init__(self, inputPath: str = '', name: str = ''):
        if not os.path.exists(inputPath):
            raise FileNotFoundError(f"输入路径不存在: {inputPath}")
            
        self.inputPath = inputPath
        self.header: str | None = None
        self.data: bytes | None = None
        self.Kernel = None
        self.params = None
        self.name = name

        if inputPath != '':
            self.load(inputPath)

    def load(self, inputPath):
        self.inputPath = inputPath
        print(inputPath)
        try:
            with open(self.inputPath, 'rb') as file:
                header_str = ''
                while 'end_header\n' not in header_str:
                    byte = file.read(1)
                    if not byte:
                        raise ValueError("文件中未找到 'end_header'。")
                    header_str += byte.decode('utf-8')
                
                self.header = header_str
                self.data = file.read()

        except Exception as e:
            print(f"Load ply file error: {e}")
            return

        known_kernels = [Kernel_3dgs, Kernel_spacetime, Kernel_spacetime_qing]
        
        IdentifiedKernel = None
        for kernel_class in known_kernels:
            if kernel_class.identify([line.split() for line in self.header.splitlines()]):
                IdentifiedKernel = kernel_class
                break
        
        if IdentifiedKernel:
            print(f"gaussian type: {IdentifiedKernel.__name__}")
            self.Kernel = IdentifiedKernel
        else:
            raise ValueError(f"Unknown gaussian type")
        plydata = PlyData.read(inputPath)
        # 假设点元素叫 'vertex'
        vertex_data = plydata['vertex'].data

        # 转 numpy array
        ply_array = np.array(vertex_data.tolist(), dtype=np.float32)
        print("shape:")
        print(ply_array.shape)
        self.params = self.Kernel.getParams(self.data)
        self.data = None
        self.pointCount = self.params[0].shape[0]

    def reorder(self, type):
        self.params = self.Kernel.reorder(self.params, type)

        self.Kernel.analyze_point_blocks(self.params[0])

    def visualize(self):
        self.Kernel.visualize_with_pyvista(self.params)

    def toGLB(self, outputPath, saveJson):
        gltf = self.Kernel.toGLB(self.params, self.pointCount, self.name)
        gltf.save(outputPath)
        xyz, motion1, motion2, motion3, tc, s, ts, q, color, color_rest = self.params
        p = Path(self.inputPath)

        # 改成 txt 后缀
        txt_path = p.with_suffix(".txt")
        np.savetxt(txt_path, color_rest.astype(np.float32), fmt="%.8f", delimiter=" ")
        zip_file = p.with_suffix(".zip")
        with zipfile.ZipFile(zip_file, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(txt_path, arcname=txt_path)  # arcname 保持在 zip 内的文件名
        if saveJson:
            gltf.save_json(outputPath + ".json")
        