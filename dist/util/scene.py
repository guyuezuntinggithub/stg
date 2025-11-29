import numpy as np
import utils as utils
from threeD import Kernel_3dgs
from spacetime import Kernel_spacetime
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

        known_kernels = [Kernel_3dgs, Kernel_spacetime]
        
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
        if saveJson:
            gltf.save_json(outputPath + ".json")
        