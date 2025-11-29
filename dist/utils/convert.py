import argparse
import os
from kernel.threeD import Kernel_3dgs
from kernel.spacetime import Kernel_spacetime
import time
import numpy as np
import struct

def convert(inputPath: str, outputPath: str, level: int = 0):
    if not (0 <= level <= 2):
        print(f"Error: compression level must be  0, 1 or 2")
        exit(1)
    if not os.path.exists(inputPath):
        print(f"Error: input file does not exist")
        exit(1)
    if outputPath is None:
        base_name, _ = os.path.splitext(inputPath)
        outputPath = base_name + ".spb"

    with open(inputPath, 'rb') as file:
        header = ''
        while True:
            byte = file.read(1)
            char = byte.decode('utf8')
            header += char
            if 'end_header\n' in header:
                break
        rest = file.read()

    Kernel = None
    if Kernel_3dgs.identify(header):
        Kernel = Kernel_3dgs
    elif Kernel_spacetime.identify(header):
        Kernel = Kernel_spacetime
    else:
        print(f"Error: unknown gaussian type")
        exit(1)
    Kernel.ply2spb(rest, outputPath, level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert between gs file formats",
        formatter_class=argparse.RawTextHelpFormatter # 保持帮助信息中的换行格式
    )

    parser.add_argument(
        "-i", "--input",
        dest="input",
        type=str,
        help="input file path"
    )
    
    parser.add_argument(
        "-o", "--output",
        dest="output",
        type=str,
        default=None,
        help="output file path"
    )

    parser.add_argument(
        "-l", "--level",
        dest="level",
        type=int,
        choices=range(0, 4),
        default=0,
        help="Compression Level:\n0: high quality\n1: medium quality\n2: low quality\n"
    )

    # TODO: directly load to texImage2D with offset parameter
    parser.add_argument(
        '-p', '--pad',
        action='store_true',
        help='add padding to the output file to avoid data copying during loading\n' \
        'This will increases the file size but decreases the time needed to load'
    )

    args = parser.parse_args()

    convert(args.input, args.output, args.level)