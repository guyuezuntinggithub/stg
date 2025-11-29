
class SPB:
    @staticmethod
    def header(type: str, level: int, pointCount: int, pad: bool):
        use_pad = '1' if pad else '0'
        return bytes(f'SPB {type} {level} {pointCount} {use_pad}\n', 'ascii')
    
    @staticmethod
    def buffer(name: str, size: int):
        return bytes(f'Buffer {name} {size}\n', 'ascii')
    
    @staticmethod
    def endHeader():
        return bytes(f'end_header\n', 'ascii')