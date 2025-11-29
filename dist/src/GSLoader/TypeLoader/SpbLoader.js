import { GSType } from "../../Global.js";
import { GSKernel_3DGS } from "../../GSKernal/3dgs.js";
import { GSKernel_SPACETIME } from "../../GSKernal/spacetime.js";

export class SpbLoader {
    static splitHeaderAndData(arrayBuffer) {
        const contentStart = new TextDecoder('utf-8').decode(new Uint8Array(arrayBuffer, 0, Math.min(1600, arrayBuffer.byteLength)));
        const headerEnd = contentStart.indexOf('end_header') + 'end_header'.length + 1;
        const [header] = contentStart.split('end_header');
        return { header, headerEnd };
    }

    static loadFromNative(file) {
        const { header, headerEnd } = SpbLoader.splitHeaderAndData(file.data);
        file.headerEnd = headerEnd;
        const descriptor = SpbLoader.parseHeader(header, headerEnd);
        file.pad = descriptor.pad;
        let res;
        switch (GSType[descriptor.gsType]) {
            case GSType.ThreeD:
                res = GSKernel_3DGS.parseSpbData2Buffers(descriptor, file);
                break;
            case GSType.SPACETIME:
                res = GSKernel_SPACETIME.parseSpbData2Buffers(descriptor, file);
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

    static parseHeader(header, headerEnd) {
        const lines = header.split('\n');
        const descriptor = { buffers: {} };
        let bindIndex = 0;
        let offset = headerEnd;

        for (const line of lines) {
            if (line.trim() === "end_header") {
                break;
            }

            const words = line.split(/\s+/);
            const word = words[0];

            if (word === "SPB") {
                descriptor.gsType = words[1];
                descriptor.quality = words[2] === '0' ? 'high' :
                                     words[2] === '1' ? 'medium' : 'low';
                descriptor.num = parseInt(words[3], 10);
                descriptor.pad = words[4] == "0" ? false : true;
            } else if (word === "Buffer") {
                const alias = `bind${bindIndex}`;
                descriptor.buffers[alias] = {}
                descriptor.buffers[alias].name = words[1];
                descriptor.buffers[alias].offset = offset;
                offset += parseInt(words[2], 10);
                descriptor.buffers[alias].bind = bindIndex;
                bindIndex++;
            }
        }
        return descriptor;
    }
}