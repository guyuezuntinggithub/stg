import { GSType } from "../../Global.js";
import { GSKernel_3DGS } from "../../GSKernal/3dgs.js";
import { GSKernel_SPACETIME } from "../../GSKernal/spacetime.js";

export class PlyLoader {
    static splitHeaderAndData(arrayBuffer) {
        const contentStart = new TextDecoder('utf-8').decode(new Uint8Array(arrayBuffer, 0, Math.min(1600, arrayBuffer.byteLength)));
        const headerEnd = contentStart.indexOf('end_header') + 'end_header'.length + 1;
        const [header] = contentStart.split('end_header');
        return { header, headerEnd };
    }

    static loadFromNative(file, quality) {
        const { header, headerEnd } = PlyLoader.splitHeaderAndData(file.data);
        file.headerEnd = headerEnd;
        const { offsets, pointCount } = PlyLoader.parseHeader(header);
        const gsType = PlyLoader.identifyGSType(offsets);
        let res;
        switch (gsType) {
            case GSType.ThreeD:
                res = GSKernel_3DGS.parsePlyData2Buffers(pointCount, file, quality);
                break;
            case GSType.SPACETIME:
                res = GSKernel_SPACETIME.parsePlyData2Buffers(pointCount, file, quality);
                break;
            default:
                res = {
                    valid: false,
                    error: 'Unknown GSType',
                };
                break;
        };
        return res;
    }

    static parseHeader(text) {
        const lines = text.split('\n');
        const offsets = new Map();
        let offset = 0;
        let pointCount = 0;

        for (const line of lines) {
            if (line.trim() === "end_header") {
                break;
            }

            const words = line.split(/\s+/);
            const word = words[0];

            if (word === "property") {
                const type = words[1];
                const property = words[2];
                let size = 0;
                if (type === "float") {
                    size = 4;
                }
                offsets.set(property, offset);
                offset += size;
            } else if (word === "element") {
                const type = words[1];
                const count = parseInt(words[2], 10);

                if (type === "vertex") {
                    pointCount = count;
                }
            } else if (word === "format") {
                if (words[1] !== "binary_little_endian") {
                    throw new Error("ply file only supports binary_little_endian");
                }
            }
        }
        offsets.set("total", offset);
        return { offsets, pointCount };
    }

    static identifyGSType(offsets) {
        if (GSKernel_3DGS.identifyGSType(offsets)) {
            return GSType.ThreeD;
        } else if (GSKernel_SPACETIME.identifyGSType(offsets)) {
            return GSType.SPACETIME;
        } else {
            return GSType.NONE;
        }
    }
}