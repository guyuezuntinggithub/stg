import { ParserType, LoadType, FileType } from "../Global.js";
import { Utils } from "../Utils.js";
import { PlyLoader } from "./TypeLoader/PlyLoader.js";
import { SpbLoader } from "./TypeLoader/SpbLoader.js";
import { GlbLoader } from "./TypeLoader/GlbLoader.js";

console.log('Worker: Parser.js module loaded successfully');
/*return = {
    valid: Boolean,
    error: String,
    data: {
        xxx: {
            bytesPertexel: Number,
            buffer: ArrayBuffer,
        }
    },
};*/
const loadFromNative = function() {
    const map2FileType = {
        'ply': FileType.PLY,
        'spb': FileType.SPB,
        'glb': FileType.GLB,
    }
    return function(file, quality) {
        const extension = Utils.extractFileExtension(file.name);
        const fileType = map2FileType[extension] || FileType.NONE;
        file.type = fileType;
        switch (fileType) {
            case FileType.PLY:
                return PlyLoader.loadFromNative(file, quality);
            case FileType.SPB:
                return SpbLoader.loadFromNative(file);
            case FileType.GLB:
                return GlbLoader.loadFromNative(file);
            default:
                return {
                    'valid': false,
                    'error': 'Unknown file extension: ' + extension,
                };
        }
            
    };
}();

const loadFromURL = async function(file, quality) {
    const filePath = file.name;
    const response = await fetch(filePath);
    if (!response.ok) {
        throw new Error(`无法找到文件: ${filePath} - 状态: ${response.status} ${response.statusText}`);
    }
    file.data = await response.arrayBuffer();

    return loadFromNative(file, quality);
}

self.onmessage = async (event) => {
    const message = event.data;
    let error = '';
    const file = {
        name: message.name,
        data: message.data,
        from: message.from,
    }

    switch (message.parser) {
        case ParserType.CPU:
            console.log(`worker: handle ${message.name} using cpu`);
            let results;
            if (LoadType.NATIVE == message.type) {
                results = loadFromNative(file, message.quality);
            } else if(LoadType.URL == message.type) {
                results = await loadFromURL(file, message.quality);
            } else {
                results = {
                    valid: false,
                    error: "Unknown load type",
                }
            }
            
            if (results.valid) {
                const transferables = Object.values(results.data.buffers).map(value => value.buffer);
                transferables.push(results.data.file.data);
                transferables.push(results.data.sortBuffer);
                if (results.data.chunkBased) {
                    transferables.push(results.data.chunkBuffer);
                }
                // property
                results.data.sequential = message.sequential;
                results.data.frameIdx = message.frameIdx;
                self.postMessage({
                    'valid': results.valid,
                    'data': results.data,
                }, transferables);
                return;
            }
            error = results.error;
            break;
        default:
            error = 'Unknown parser type: ' + message.parser;
            break;
    }
    self.postMessage({
        'valid': false,
        'error': error,
    });

};