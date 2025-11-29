import { GSType } from "../../Global.js";
import { GSKernel_3DGS } from "../../GSKernal/3dgs.js";
import { GSKernel_SPACETIME } from "../../GSKernal/spacetime.js";
import { Utils } from "../../Utils.js";

export class GlbLoader {
    static splitHeaderAndData(file) {
        const dataView = new DataView(file.data);

        const magic = dataView.getUint32(0, true);
        const version = dataView.getUint32(4, true);
        const length = dataView.getUint32(8, true);

        if (magic !== 0x46546C67) {
            throw new Error('invalid GLB file');
        }
        if (version !== 2) {
            throw new Error('glTF 2.0 supported only。');
        }

        let chunkOffset = 12;

        // 'JSON'
        const jsonChunkLength = dataView.getUint32(chunkOffset, true);
        const jsonChunkType = dataView.getUint32(chunkOffset + 4, true);
        if (jsonChunkType !== 0x4E4F534A) {
            throw new Error('json not found in glb file');
        }
        chunkOffset += 8
        const jsonChunkData = new Uint8Array(file.data, chunkOffset, jsonChunkLength);
        const jsonString = new TextDecoder('utf-8').decode(jsonChunkData);
        const json = JSON.parse(jsonString);
        file.json = json;
        
        chunkOffset += jsonChunkLength;

        // BIN
        if (chunkOffset < length) {
            const binaryChunkLength = dataView.getUint32(chunkOffset, true);
            const binaryChunkType = dataView.getUint32(chunkOffset + 4, true);
            if (binaryChunkType !== 0x004E4942) { // 'BIN'
                throw new Error('BIN chunk not found in glb file');
            }
            chunkOffset += 8;
            file.headerEnd = chunkOffset;
        }
    }

    static loadFromNative(file) {
        try {
            GlbLoader.splitHeaderAndData(file);
            const scene = GlbLoader.parseJson(file);
            GlbLoader.createSortBufferAndChunkBuffer(scene);

            return {
                valid: true,
                data: scene,
            };

        } catch (error) {
            return {
                valid: false,
                error: error.message,
            };
        }
    }

    static parseJson(file) {
        const json = file.json;
        const scene = {};
        scene.buffers = {};

        for (const image of json.images) {
            const bufferView = json.bufferViews[image.bufferView];
            const name = image.extras.name;
            const arrayBuffer = file.data;
            scene.buffers[name] = {...image.extras};
            const offset = bufferView.byteOffset + file.headerEnd;
            scene.buffers[name].offset = offset;
            scene.buffers[name].size = bufferView.byteLength;
            scene.buffers[name].buffer = arrayBuffer.slice(offset, offset + bufferView.byteLength);
        }

        scene.gsType = json.nodes[0].extras.gsType;
        scene.name = json.nodes[0].extras.name;
        scene.quality = json.nodes[0].extras.quality;
        scene.num = json.nodes[0].extras.num;

        scene.appliedTransform = json.nodes[0].matrix;
        scene.file = file;
        scene.chunkBased = 'chunkBased';
        scene.chunkNum = scene.num / 256;
        scene.chunkResolution = {
            width: scene.buffers.u_xyz.width / 16,
            height: scene.buffers.u_xyz.height / 16,
        }
        
        for (const image of json.images) {
            const name = image.extras.name;
            if (name === 'u_range') {
                scene.buffers[name].texelPerSplat = scene.buffers[name].width / scene.chunkResolution.width;
            } else {
                scene.buffers[name].texelPerSplat = 1;
            }
        }
        return scene;
    }

    static createSortBufferAndChunkBuffer(scene) {
        switch (GSType[scene.gsType]) {
            case GSType.ThreeD:
                GSKernel_3DGS.createSortBufferAndChunkBuffer(scene);
                break;
            case GSType.SPACETIME:
                GSKernel_SPACETIME.createSortBufferAndChunkBuffer(scene);
                break;
            default:
                throw new Error(`Unknown gsType ${scene.gsType}`);
        }
    }
}