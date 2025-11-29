from scene import Scene
import argparse
import os

def convert(args):
    level, inputPath, outputPath, name = args.level, args.input, args.output, args.name
    quiet, visualize, reorder, saveJson = args.quiet, args.visualize, args.reorder, args.json

    has_name = True
    if name == "":
        has_name = False
    if not (0 <= level <= 2):
        print(f"Error: compression level must be  0, 1 or 2")
        exit(1)
    if not os.path.exists(inputPath):
        print(f"Error: input file/directory does not exist")
        exit(1)

    first_level_files = []
    if os.path.isdir(inputPath):    # handle files in the directory
        first_level_files = []
        if outputPath is None:
            outputPath = inputPath
        else:
            if not os.path.exists(outputPath):
                print(f"Error: output directory does not exist")
                exit(1)
            elif not os.path.isdir(outputPath):
                print(f"Error: output path should be directory")
                exit(1)
        try:
            for entry_name in os.listdir(inputPath):
                full_path = os.path.join(inputPath, entry_name)
                full_out_path = os.path.join(outputPath, entry_name)
                if os.path.isfile(full_path) and entry_name.lower().endswith('.ply'):
                    first_level_files.append((full_path, full_out_path.replace('.ply', '.glb')))
        except OSError as e:
            print(f"do not have access to {inputPath}: {e}")
    elif os.path.isfile(inputPath):
        if outputPath is None:
            base_name, _ = os.path.splitext(inputPath)
            outputPath = base_name + ".glb"
        else:
            if not os.path.exists(os.path.dirname(outputPath)):
                print(f"Error: output directory '{os.path.dirname(outputPath)}' does not exist")
                exit(1)
            elif not outputPath.lower().endswith('.glb'):
                print(f"Error: output file '{outputPath}' should ends with .glb")
                exit(1)
        if inputPath.lower().endswith('.ply'):
            first_level_files.append((inputPath, outputPath))
    else:
        print("Invalid input path")
        exit(1)

    for file_path, out_path in first_level_files:
        if not has_name:
            name, _ = os.path.splitext(os.path.basename(file_path))
        print(f"\n\n============================================")
        print(f"converting {name} from {file_path} to {out_path}")
        scene = Scene(file_path, name)
        scene.reorder(reorder)
        if visualize:
            scene.visualize()
        if not quiet:
            scene.toGLB(out_path, saveJson)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert between gs file formats",
        formatter_class=argparse.RawTextHelpFormatter # 保持帮助信息中的换行格式
    )

    parser.add_argument(
        "-i", "--input",
        dest="input",
        type=str,
        help="input file path or directory"
    )
    
    parser.add_argument(
        "-o", "--output",
        dest="output",
        type=str,
        default=None,
        help="output file path or directory"
    )

    parser.add_argument(
        "-n", "--name",
        dest="name",
        type=str,
        default="",
        help="scene name. \n\
            Default: file name from input path"
    )

    parser.add_argument(
        "-r", "--reorder",
        dest="reorder",
        type=str,
        default="Morton",
        help="reorder using 'Morton' or 'Hilbert' curve. \n\
            'Morton' is quick while 'Hilbert' might take a while but brings better quality\n\
            Default: Morton"
    )

    parser.add_argument(
        "-l", "--level",
        dest="level",
        type=int,
        choices=range(0, 4),
        default=0,
        help="[deprecated]\n\
            Compression Level:\n0: high quality\n1: medium quality\n2: low quality\n"
    )

    parser.add_argument(
        '-q', 
        '--quiet', 
        action='store_true', 
        help="do not output file"
    )

    parser.add_argument(
        '-v', 
        '--visualize', 
        action='store_true', 
        help="visualize point cloud"
    )

    parser.add_argument(
        '-j', 
        '--json', 
        action='store_true', 
        help="save json file about the gltf"
    )

    args = parser.parse_args()

    convert(args)