import { Utils } from "../Utils.js";

export class GSKernel_3DGS {
    static offsets = [];
    static params = {
        x: 0, y: 1, z: 2,
        sx: 3, sy: 4, sz: 5,
        rx: 6, ry: 7, rz: 8, rw: 9,
        cr: 10, cg: 11, cb: 12, ca: 13,
        d1r0: 14, d1r1: 15, d1r2: 16,
        d1g0: 17, d1g1: 18, d1g2: 19,
        d1b0: 20, d1b1: 21, d1b2: 22,
        d2r0: 23, d2r1: 24, d2r2: 25, d2r3: 26, d2r4: 27,
        d2g0: 28, d2g1: 29, d2g2: 30, d2g3: 31, d2g4: 32,
        d2b0: 33, d2b1: 34, d2b2: 35, d2b3: 36, d2b4: 37,
        d3r0: 38, d3r1: 39, d3r2: 40, d3r3: 41, d3r4: 42, d3r5: 43, d3r6: 44, 
        d3g0: 45, d3g1: 46, d3g2: 47, d3g3: 48, d3g4: 49, d3g5: 50, d3g6: 51, 
        d3b0: 52, d3b1: 53, d3b2: 54, d3b3: 55, d3b4: 56, d3b5: 57, d3b6: 58,
        total: 59, 
    };
    static unit8PackRangeMin = -1.0;
    static unit8PackRangeMax = 1.0;
    static identifyGSType(offsets) {
        // a little hack, we only check f_rest_x to differ from 3dgs and spacetime
        const keysToCheck = [
            'f_rest_0',
        ];
        const res = keysToCheck.every(key => offsets.has(key));
        if (res) {
            GSKernel_3DGS.updateOffsets(offsets);
        }
        return res;
    }
    static config = {
        low : {
            pospad: {
                name: 'Pos6Pad2',
                bytesPerTexel: 3 * 2 + 2,
                texelPerSplat: 1,
                format: "RGBA16F",
                array: 1,
            },
            covcol: {
                name: 'Cov12Col4',
                bytesPerTexel: 2 * 6 + 4,
                texelPerSplat: 1,
                format: "RGBA32UI",
                array: 1,
            },
            sh: {  // deg 0
                name: 'SH0',
                bytesPerTexel: 0,
                texelPerSplat: 0,
                format: "",
                array: 0,
                deg: 0,
            },
        },
        medium: {
            pospad: {
                name: 'Pos6Pad2',
                bytesPerTexel: 3 * 2 + 2,
                texelPerSplat: 1,
                format: "RGBA16F",
                array: 1,
            },
            covcol: {
                name: 'Cov12Col4',
                bytesPerTexel: 2 * 6 + 4,
                texelPerSplat: 1,
                format: "RGBA32UI",
                array: 1,
            },
            sh: {   // deg 1
                name: 'SH9Pad3',
                bytesPerTexel: 12,
                texelPerSplat: 1,
                format: "RGB32UI",
                array: 1,
                deg: 1,
            },
        },
        high: {
            pospad: {
                name: 'Pos12Pad4',
                bytesPerTexel: 3 * 4 + 4,
                texelPerSplat: 1,
                format: "RGBA32F",
                array: 1,
            },
            covcol: {
                name: 'Cov12Col4',
                bytesPerTexel: 2 * 6 + 4,
                texelPerSplat: 1,
                format: "RGBA32UI",
                array: 1,
            },
            sh: { // deg 2
                name: 'SH24',
                bytesPerTexel: 12,
                texelPerSplat: 2,
                format: "RGB32UI",
                array: 1,
                deg: 2,
            },
        }
    }

    static updateOffsets = function() {
        
        return function(offsets) {
            GSKernel_3DGS.offsets[GSKernel_3DGS.params.x] = (offsets.get("x") / 4)>>>0;
            GSKernel_3DGS.offsets[GSKernel_3DGS.params.y] = (offsets.get("y") / 4)>>>0;
            GSKernel_3DGS.offsets[GSKernel_3DGS.params.z] = (offsets.get("z") / 4)>>>0;
            GSKernel_3DGS.offsets[GSKernel_3DGS.params.sx] = (offsets.get("scale_0") / 4)>>>0;
            GSKernel_3DGS.offsets[GSKernel_3DGS.params.sy] = (offsets.get("scale_1") / 4)>>>0;
            GSKernel_3DGS.offsets[GSKernel_3DGS.params.sz] = (offsets.get("scale_2") / 4)>>>0;
            GSKernel_3DGS.offsets[GSKernel_3DGS.params.rx] = (offsets.get("rot_1") / 4)>>>0;
            GSKernel_3DGS.offsets[GSKernel_3DGS.params.ry] = (offsets.get("rot_2") / 4)>>>0;
            GSKernel_3DGS.offsets[GSKernel_3DGS.params.rz] = (offsets.get("rot_3") / 4)>>>0;
            GSKernel_3DGS.offsets[GSKernel_3DGS.params.rw] = (offsets.get("rot_0") / 4)>>>0;
            GSKernel_3DGS.offsets[GSKernel_3DGS.params.cr] = (offsets.get("f_dc_0") / 4)>>>0;
            GSKernel_3DGS.offsets[GSKernel_3DGS.params.cg] = (offsets.get("f_dc_1") / 4)>>>0;
            GSKernel_3DGS.offsets[GSKernel_3DGS.params.cb] = (offsets.get("f_dc_2") / 4)>>>0;
            GSKernel_3DGS.offsets[GSKernel_3DGS.params.ca] = (offsets.get("opacity") / 4)>>>0;
            for (let i = 0; i < 3; ++i) {
                GSKernel_3DGS.offsets[GSKernel_3DGS.params.d1r0 + i] = (offsets.get("f_rest_" + ( 0 + i)) / 4)>>>0;
                GSKernel_3DGS.offsets[GSKernel_3DGS.params.d1g0 + i] = (offsets.get("f_rest_" + (15 + i)) / 4)>>>0;
                GSKernel_3DGS.offsets[GSKernel_3DGS.params.d1b0 + i] = (offsets.get("f_rest_" + (30 + i)) / 4)>>>0;
            }
            for (let i = 0; i < 5; ++i) {
                GSKernel_3DGS.offsets[GSKernel_3DGS.params.d2r0 + i] = (offsets.get("f_rest_" + ( 0 + 3 + i)) / 4)>>>0;
                GSKernel_3DGS.offsets[GSKernel_3DGS.params.d2g0 + i] = (offsets.get("f_rest_" + (15 + 3 + i)) / 4)>>>0;
                GSKernel_3DGS.offsets[GSKernel_3DGS.params.d2b0 + i] = (offsets.get("f_rest_" + (30 + 3 + i)) / 4)>>>0;
            }
            for (let i = 0; i < 7; ++i) {
                GSKernel_3DGS.offsets[GSKernel_3DGS.params.d3r0 + i] = (offsets.get("f_rest_" + ( 0 + 8 + i)) / 4)>>>0;
                GSKernel_3DGS.offsets[GSKernel_3DGS.params.d3g0 + i] = (offsets.get("f_rest_" + (15 + 8 + i)) / 4)>>>0;
                GSKernel_3DGS.offsets[GSKernel_3DGS.params.d3b0 + i] = (offsets.get("f_rest_" + (30 + 8 + i)) / 4)>>>0;
            }
            GSKernel_3DGS.offsets[GSKernel_3DGS.params.total] = (offsets.get("total") / 4)>>>0;
        }
    }();

    static parseSplatFromData = function() {
        const SH_C0 = 0.28209479177387814;

        return function(idx, splat, dataview) {
            const splatBytesOffset = idx * GSKernel_3DGS.offsets[GSKernel_3DGS.params.total] * 4;
            Object.keys(splat).forEach(key => {
                if(key === 'total') return;
                splat[key] = dataview.getFloat32(splatBytesOffset + GSKernel_3DGS.offsets[GSKernel_3DGS.params[key]] * 4, true);
            });
            splat.sx = Math.exp(splat.sx);
            splat.sy = Math.exp(splat.sy);
            splat.sz = Math.exp(splat.sz);
            splat.cr = Utils.clamp(0.5 + SH_C0 * splat.cr, 0, 1);
            splat.cg = Utils.clamp(0.5 + SH_C0 * splat.cg, 0, 1);
            splat.cb = Utils.clamp(0.5 + SH_C0 * splat.cb, 0, 1);
            splat.ca = Utils.sigmoid(splat.ca);
        }
    }();

    static parsePlyData2Buffers = function() {

        return function(pointCount, file, quality = 'medium') {
            // a little hack, pointCount shouldn't be too large (<= 8,388,608)
            if (pointCount > 4096 * 2048) {
                console.warn(`pointCount ${pointCount} is too large and is clamped to 8,388,608`);
                pointCount = 4096 * 2048;
            }
            const dataview = new DataView(file.data, file.headerEnd);

            const buffers = GSKernel_3DGS.config[quality];
            const pospad = buffers.pospad;
            const covcol = buffers.covcol;
            const sh     = buffers.sh;

            Object.assign(pospad, Utils.computeTexSize(pospad.texelPerSplat * pointCount));
            Object.assign(covcol, Utils.computeTexSize(covcol.texelPerSplat * pointCount));
            Object.assign(sh, Utils.computeTexSize(sh.texelPerSplat * pointCount));

            // TODO: deg 0 and deg1 and deg3
            pospad.buffer = new ArrayBuffer(pospad.width * pospad.height * pospad.bytesPerTexel);
            covcol.buffer = new ArrayBuffer(covcol.width * covcol.height * covcol.bytesPerTexel);
            sh.buffer = new ArrayBuffer(sh.width * sh.height * sh.bytesPerTexel);
            const sortBuffer = new Float32Array(pointCount * 4);

            const pospadView = new DataView(pospad.buffer);
            const covcolView = new DataView(covcol.buffer);
            const shView = new DataView(sh.buffer);
            const splat = {...GSKernel_3DGS.params};
            let pospadOffset = 0, covcolOffset = 0, shOffset = 0, sortOffset = 0;
            for (let i = 0;i < pointCount; ++i) {
                GSKernel_3DGS.parseSplatFromData(i, splat, dataview);

                if (pospad.bytesPerTexel == 8) {
                    // TODO: test if using fp16 affects the quality, for now we dont use it
                    pospadView.setUint16(pospadOffset + 0, Utils.f2fp162uint16(splat.x), true);
                    pospadView.setUint16(pospadOffset + 2, Utils.f2fp162uint16(splat.y), true);
                    pospadView.setUint16(pospadOffset + 4, Utils.f2fp162uint16(splat.z), true);
                } else {
                    pospadView.setFloat32(pospadOffset + 0, splat.x, true);
                    pospadView.setFloat32(pospadOffset + 4, splat.y, true);
                    pospadView.setFloat32(pospadOffset + 8, splat.z, true);
                }
                
                Utils.computeCov3dPack2fp16(
                    splat.sx, splat.sy, splat.sz, splat.rx, splat.ry, splat.rz, splat.rw, 
                    covcolView, covcolOffset
                );
                Utils.packFloat2rgba(
                    splat.cr, splat.cg, splat.cb, splat.ca,
                    covcolView, covcolOffset + 12
                );
                if (sh.deg >= 1) {
                    for(let j = 0;j < 3;++j) {
                        shView.setUint8(shOffset + 3 * j + 0, Utils.float2uint8(splat['d1r' + j], GSKernel_3DGS.unit8PackRangeMin, GSKernel_3DGS.unit8PackRangeMax));
                        shView.setUint8(shOffset + 3 * j + 1, Utils.float2uint8(splat['d1g' + j], GSKernel_3DGS.unit8PackRangeMin, GSKernel_3DGS.unit8PackRangeMax));
                        shView.setUint8(shOffset + 3 * j + 2, Utils.float2uint8(splat['d1b' + j], GSKernel_3DGS.unit8PackRangeMin, GSKernel_3DGS.unit8PackRangeMax));
                    }
                    if (sh.deg >= 2) {
                        for(let j = 0;j < 5;++j) {
                            shView.setUint8(shOffset + 9 + 3 * j + 0, Utils.float2uint8(splat['d2r' + j], GSKernel_3DGS.unit8PackRangeMin, GSKernel_3DGS.unit8PackRangeMax));
                            shView.setUint8(shOffset + 9 + 3 * j + 1, Utils.float2uint8(splat['d2g' + j], GSKernel_3DGS.unit8PackRangeMin, GSKernel_3DGS.unit8PackRangeMax));
                            shView.setUint8(shOffset + 9 + 3 * j + 2, Utils.float2uint8(splat['d2b' + j], GSKernel_3DGS.unit8PackRangeMin, GSKernel_3DGS.unit8PackRangeMax));
                        }
                        // we don't use deg3 for now
                        /*if (sh.deg >= 3) {
                            for(let j = 0;j < 7;++j) {
                                shView.setUint8(shOffset + 24 + 3 * j + 0, Utils.float2uint8(splat['d3r' + j], unit8PackRangeMin, unit8PackRangeMax));
                                shView.setUint8(shOffset + 24 + 3 * j + 1, Utils.float2uint8(splat['d3g' + j], unit8PackRangeMin, unit8PackRangeMax));
                                shView.setUint8(shOffset + 24 + 3 * j + 2, Utils.float2uint8(splat['d3b' + j], unit8PackRangeMin, unit8PackRangeMax));
                            }
                        }*/
                    }
                }

                sortBuffer[sortOffset + 0] = splat.x;
                sortBuffer[sortOffset + 1] = splat.y;
                sortBuffer[sortOffset + 2] = splat.z;
                sortBuffer[sortOffset + 3] = 1;
                
                pospadOffset += pospad.bytesPerTexel * pospad.texelPerSplat;
                covcolOffset += covcol.bytesPerTexel * covcol.texelPerSplat;
                shOffset += sh.bytesPerTexel * sh.texelPerSplat;
                sortOffset += 4;
            }

            if (sh.deg == 0) {
                delete buffers.sh;
            }

            return {
                valid: true,
                data: {
                    buffers: buffers,
                    file: file,
                    gsType: 'ThreeD',
                    num: pointCount,
                    sortBuffer: sortBuffer.buffer,
                    quality: quality,
                },
            }
        }
    }();

    static parseSpbData2Buffers(descriptor, file) {
        const arrayBuffer = file.data;
        const quality = descriptor.quality;
        const pointCount = descriptor.num;
        const buffers = {...GSKernel_3DGS.config[quality]};
        const pospad = buffers.pospad;
        const covcol = buffers.covcol;
        const sh = buffers.sh;

        pospad.offset = descriptor.buffers.bind0.offset;
        covcol.offset = descriptor.buffers.bind1.offset;
        sh.offset = descriptor.buffers.bind2.offset;

        Object.assign(pospad, Utils.computeTexSize(pospad.texelPerSplat * pointCount));
        Object.assign(covcol, Utils.computeTexSize(covcol.texelPerSplat * pointCount));
        Object.assign(sh, Utils.computeTexSize(sh.texelPerSplat * pointCount));

        // TODO: deg 0 and deg1 and deg3
        if (descriptor.pad) {
            pospad.buffer = arrayBuffer.slice(pospad.offset, pospad.offset + pospad.width * pospad.height * pospad.bytesPerTexel);
            covcol.buffer = arrayBuffer.slice(covcol.offset, covcol.offset + covcol.width * covcol.height * covcol.bytesPerTexel);
            sh.buffer = arrayBuffer.slice(sh.offset, sh.offset + sh.width * sh.height * sh.bytesPerTexel);
        } else {
            let sliceEnd = pospad.offset + pospad.width * pospad.height * pospad.bytesPerTexel;
            if (sliceEnd <= arrayBuffer.byteLength) {
                pospad.buffer = arrayBuffer.slice(pospad.offset, sliceEnd);
            } else {
                pospad.buffer = new ArrayBuffer(pospad.width * pospad.height * pospad.bytesPerTexel);
                new Uint8Array(pospad.buffer).set(new Uint8Array(arrayBuffer, pospad.offset));
            }
            sliceEnd = covcol.offset + covcol.width * covcol.height * covcol.bytesPerTexel;
            if (sliceEnd <= arrayBuffer.byteLength) {
                covcol.buffer = arrayBuffer.slice(covcol.offset, sliceEnd);
            } else {
                covcol.buffer = new ArrayBuffer(covcol.width * covcol.height * covcol.bytesPerTexel);
                new Uint8Array(covcol.buffer).set(new Uint8Array(arrayBuffer, covcol.offset));
            }
            sliceEnd = sh.offset + sh.width * sh.height * sh.bytesPerTexel;
            if (sliceEnd <= arrayBuffer.byteLength) {
                sh.buffer = arrayBuffer.slice(sh.offset, sliceEnd);
            } else {
                sh.buffer = new ArrayBuffer(sh.width * sh.height * sh.bytesPerTexel);
                new Uint8Array(sh.buffer).set(new Uint8Array(arrayBuffer, sh.offset));
            }
        }
        
        const sortBuffer = new Int32Array(pointCount * 4);
        const dataview = new DataView(pospad.buffer);
        let offset = 0, sortOffset = 0;
        if (quality == "high") {
            for (let i = 0;i < pointCount; ++i) {
                sortBuffer[sortOffset + 0] = Math.round(dataview.getFloat32(offset + 0, true) * 1000.0);
                sortBuffer[sortOffset + 1] = Math.round(dataview.getFloat32(offset + 4, true) * 1000.0);
                sortBuffer[sortOffset + 2] = Math.round(dataview.getFloat32(offset + 8, true) * 1000.0);
                sortBuffer[sortOffset + 3] = 1000;
                offset += 16;
                sortOffset += 4;
            }
        } else {
            for (let i = 0;i < pointCount; ++i) {
                sortBuffer[sortOffset + 0] = Math.round(Utils.readFp16(dataview, offset + 0, true) * 1000.0);
                sortBuffer[sortOffset + 1] = Math.round(Utils.readFp16(dataview, offset + 2, true) * 1000.0);
                sortBuffer[sortOffset + 2] = Math.round(Utils.readFp16(dataview, offset + 4, true) * 1000.0);
                sortBuffer[sortOffset + 3] = 1000;
                offset += 8;
                sortOffset += 4;
            }
        }

        if (sh.deg == 0) {
            delete buffers.sh;
        }
        
        return {
            valid: true,
            data: {
                buffers: buffers,
                file: file,
                gsType: 'ThreeD',
                num: pointCount,
                sortBuffer: sortBuffer.buffer,
                quality: quality,
            },
        }
    }

    static getUniformDefines() {
        return `
            // 3dgs specific uniforms
            // nothing
        `;
    }

    static getFetchFunc(buffers, chunkBased) {
        let res = ``;

        if (chunkBased) {
            res += `
                void splatIndex2RangeUV(in uint splatIndex, inout ivec2 rangeUV, inout ivec2 uv) {
                    ivec2 texSize = textureSize(u_xyz, 0);

                    int chunkWidth = texSize.x >> 4;;
                    int current = int(splatIndex) >> 4;
                    rangeUV.x = current % chunkWidth;
                    rangeUV.y = (current / chunkWidth) >> 4;

                    uv = index2uv(splatIndex, 1u, 0u, texSize);
                }

                mat3 fetchVrk(in vec3 s, in vec4 q)
                {
                    q = normalize(q);

                    float xx = q.x * q.x;
                    float yy = q.y * q.y;
                    float zz = q.z * q.z;
                    float xy = q.x * q.y;
                    float xz = q.x * q.z;
                    float yz = q.y * q.z;
                    float wx = q.w * q.x;
                    float wy = q.w * q.y;
                    float wz = q.w * q.z;
                    mat3 rot = mat3(
                        1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy),
                        2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx),
                        2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)
                    );

                    mat3 ss = mat3(
                        s.x * s.x, 0.0, 0.0,
                        0.0, s.y * s.y, 0.0,
                        0.0, 0.0, s.z * s.z
                    );
                    return rot * ss * transpose(rot);
                }
            `
            return res;
        }

        const pospad = buffers.pospad;
        if (pospad) {
            res += `
                void fetchCenter(in uint splatIndex, inout vec3 center)
                {
                    center = vec3(texelFetch(${pospad.name}, index2uv(splatIndex, ${pospad.texelPerSplat}u, 0u, textureSize(${pospad.name}, 0)), 0));
                }
            `
        }
        const covcol = buffers.covcol;
        if (covcol) {
            res += `
                void fetchCovCol(in uint splatIndex, inout mat3 cov3d, inout vec4 color)
                {
                    uvec4 texel = texelFetch(${covcol.name}, index2uv(splatIndex, ${covcol.texelPerSplat}u, 0u, textureSize(${covcol.name}, 0)), 0);
                    vec2 cov01 = uint2fp16x2(texel.x);
                    vec2 cov24 = uint2fp16x2(texel.y);
                    vec2 cov58 = uint2fp16x2(texel.z);
                    cov3d = mat3(
                        cov01.x, cov01.y, cov24.x,
                        cov01.y, cov24.y, cov58.x,
                        cov24.x, cov58.x, cov58.y
                    );
                    color = uint2rgba(texel.w);
                }
            `
        }
        const sh = buffers.sh;
        if (sh) {
            const deg2 = sh.deg === 2;
            const range = (GSKernel_3DGS.unit8PackRangeMax - GSKernel_3DGS.unit8PackRangeMin).toFixed(5);
            const min = GSKernel_3DGS.unit8PackRangeMin.toFixed(5);
            res += `
                void fetchSH(in uint splatIndex, inout vec3 shd1[3]${deg2 ? `, inout vec3 shd2[5]` : ``})
                {
                    uvec4 texel = texelFetch(${sh.name}, index2uv(splatIndex, ${sh.texelPerSplat}u, 0u, textureSize(${sh.name}, 0)), 0);
                    vec4 sh00_03 = uint2rgba(texel.x);
                    vec4 sh04_07 = uint2rgba(texel.y);
                    vec4 sh08_11 = uint2rgba(texel.z);
                    shd1[0] = vec3(sh00_03.x, sh00_03.y, sh00_03.z) * ${range} + (${min});
                    shd1[1] = vec3(sh00_03.w, sh04_07.x, sh04_07.y) * ${range} + (${min});
                    shd1[2] = vec3(sh04_07.z, sh04_07.w, sh08_11.x) * ${range} + (${min});
                    ${deg2 ? `
                    texel = texelFetch(${sh.name}, index2uv(splatIndex, ${sh.texelPerSplat}u, 1u, textureSize(${sh.name}, 0)), 0);
                    vec4 sh12_15 = uint2rgba(texel.x);
                    vec4 sh16_19 = uint2rgba(texel.y);
                    vec4 sh20_23 = uint2rgba(texel.z);
                    shd2[0] = vec3(sh08_11.y, sh08_11.z, sh08_11.w) * ${range} + (${min});
                    shd2[1] = vec3(sh12_15.x, sh12_15.y, sh12_15.z) * ${range} + (${min});
                    shd2[2] = vec3(sh12_15.w, sh16_19.x, sh16_19.y) * ${range} + (${min});
                    shd2[3] = vec3(sh16_19.z, sh16_19.w, sh20_23.x) * ${range} + (${min});
                    shd2[4] = vec3(sh20_23.y, sh20_23.z, sh20_23.w) * ${range} + (${min});
                    ` : ``}
                }
            `
        }

        return res;
    }

    static getFetchParams(chunkBased) {
        let res = ``;
        if (chunkBased) {
            res += `{
                ivec2 rangeUV, uv;
                splatIndex2RangeUV(splatIndex, rangeUV, uv);
                
                rangeUV.x *= 2;
                uvec4 range = texelFetch(u_range, rangeUV, 0);
                vec2 xmin_ymin = unpackHalf2x16(range.x);
                vec2 zmin_xmax = unpackHalf2x16(range.y);
                vec2 ymax_zmax = unpackHalf2x16(range.z);
                vec2 smin_smax = unpackHalf2x16(range.w);

                rangeUV.x += 1;
                range = texelFetch(u_range, rangeUV, 0);
                vec2 rmin_rmax = unpackHalf2x16(range.x);
                vec2 gmin_gmax = unpackHalf2x16(range.y);
                vec2 bmin_bmax = unpackHalf2x16(range.z);

                uint x11y10z11 = texelFetch(u_xyz, uv, 0).r;
                const float inv1023 = 0.0009775171;
                const float inv2047 = 0.0004885198;
                splatCenter.x = (float((x11y10z11 >> 0) & 0x7FFu) * inv2047) * (zmin_xmax.y - xmin_ymin.x) + xmin_ymin.x;
                splatCenter.y = (float((x11y10z11 >>11) & 0x3FFu) * inv1023) * (ymax_zmax.x - xmin_ymin.y) + xmin_ymin.y;
                splatCenter.z = (float((x11y10z11 >>21) & 0x7FFu) * inv2047) * (ymax_zmax.y - zmin_xmax.x) + zmin_xmax.x;

                vec3 s = texelFetch(u_s, uv, 0).rgb;
                s = s * (smin_smax.y - smin_smax.x) + smin_smax.x;
                s *= s;
                vec4 q = texelFetch(u_q, uv, 0);
                q = q * 2.0 - 1.0;
                Vrk = fetchVrk(s, q);

                splatColor = texelFetch(u_color, uv, 0);
                splatColor.r = splatColor.r * (rmin_rmax.y - rmin_rmax.x) + rmin_rmax.x;
                splatColor.g = splatColor.g * (gmin_gmax.y - gmin_gmax.x) + gmin_gmax.x;
                splatColor.b = splatColor.b * (bmin_bmax.y - bmin_bmax.x) + bmin_bmax.x;
            }`;
            return res;
        }

        res = `
            fetchCenter(splatIndex, splatCenter);
            fetchCovCol(splatIndex, Vrk, splatColor);
        `

        return res;
    }

    static getSpecificCode(buffers) {
        const sh = buffers.sh;
        let res = ``;
        if (sh) {
            const deg2 = sh.deg === 2;
            res += `
            {
                vec3 shd1[3];
                ${deg2 ? `vec3 shd2[5];` : ``}
                fetchSH(splatIndex, shd1${deg2 ? `, shd2` : ``});

                vec3 worldViewDir = normalize(splatCenter - cameraPosition);
                float x = worldViewDir.x;
                float y = worldViewDir.y;
                float z = worldViewDir.z;
                splatColor.rgb += SH_C1 * (-shd1[0] * y + shd1[1] * z - shd1[2] * x);

                ${deg2 ? `
                float xx = x * x;
                float yy = y * y;
                float zz = z * z;
                float xy = x * y;
                float yz = y * z;
                float xz = x * z;

                splatColor.rgb += (SH_C2[0] * xy) * shd2[0] + (SH_C2[1] * yz) * shd2[1] + (SH_C2[2] * (2.0 * zz - xx - yy)) * shd2[2]
                        + (SH_C2[3] * xz) * shd2[3] + (SH_C2[4] * (xx - yy)) * shd2[4];
                ` : ``}
            }`
        }
        return res;
    }

    static createSortBufferAndChunkBuffer(scene) {
        // splats may not fill the entire texture
        // so that the indices of valid splats are likely of incontiuity.
        // therefore we need all splats on texture
        const allSplatsOnTexture = scene.chunkResolution.width * scene.chunkResolution.height * 256;
        const sortBuffer = new Float32Array(allSplatsOnTexture * 4);
        const chunkBuffer = new Float32Array(scene.chunkNum * 6);
        const xyz = new Uint32Array(scene.buffers.u_xyz.buffer);
        const range = new DataView(scene.buffers.u_range.buffer);
        const chunkWidth = scene.buffers.u_xyz.width / 16, chunkHeight = scene.buffers.u_xyz.height / 16;
        const chunkNum = scene.chunkNum;
        const bit11Mask = 0x7FF;
        const bit10Mask = 0x3FF;
        // [chunkHeight, 16, chunkWidth, 16, 1]
        // note that: chunkNum <= chunkWidth * chunkHeight
        for (let i = 0; i < chunkNum; ++i) {
            const rangeOffset = i * 4 * 4 * scene.buffers.u_range.texelPerSplat;
            const xmin = Utils.readFp16(range, rangeOffset + 0, true);
            const ymin = Utils.readFp16(range, rangeOffset + 2, true);
            const zmin = Utils.readFp16(range, rangeOffset + 4, true);
            const xmax = Utils.readFp16(range, rangeOffset + 6, true);
            const ymax = Utils.readFp16(range, rangeOffset + 8, true);
            const zmax = Utils.readFp16(range, rangeOffset + 10, true);
            const chunk_w = i % chunkWidth;
            const chunk_h = Math.floor(i / chunkWidth);
            for (let local_h = 0; local_h < 16; ++local_h) {
                for (let local_w = 0; local_w < 16; ++local_w) {
                    const splatIndex = 1 * local_w + 16 * chunk_w + 16 * chunkWidth * local_h + 16 * chunkWidth * 16 * chunk_h;
                    const x11y10z11 = xyz[splatIndex];
                    const offset = splatIndex * 4;
                    if (offset >= sortBuffer.length) {
                        console.log('aaa', sortBuffer[offset])
                    }
                    sortBuffer[offset + 0] = Utils.uintX2float(bit11Mask&(x11y10z11>> 0), 11, xmin, xmax);
                    sortBuffer[offset + 1] = Utils.uintX2float(bit10Mask&(x11y10z11>>11), 10, ymin, ymax);
                    sortBuffer[offset + 2] = Utils.uintX2float(bit11Mask&(x11y10z11>>21), 11, zmin, zmax);
                }
            }

            const chunkOffset = 6 * i;
            chunkBuffer[chunkOffset + 0] = xmin;
            chunkBuffer[chunkOffset + 1] = ymin;
            chunkBuffer[chunkOffset + 2] = zmin;
            chunkBuffer[chunkOffset + 3] = xmax;
            chunkBuffer[chunkOffset + 4] = ymax;
            chunkBuffer[chunkOffset + 5] = zmax;
        }

        scene.sortBuffer = sortBuffer.buffer;
        scene.chunkBuffer = chunkBuffer.buffer;
    }
}