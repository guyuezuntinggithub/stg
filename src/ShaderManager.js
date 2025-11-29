import { GSType } from "./Global.js";
import { GSKernel_3DGS } from "./GSKernal/3dgs.js";
import { GSKernel_SPACETIME } from "./GSKernal/spacetime.js";
import { Utils } from "./Utils.js";

export class ShaderManager {
    static shaderHelperFunc = `
        ivec2 index2uv(in uint index, in uint stride, in uint offset, in ivec2 size) {
            int linearIndex = int(index * stride + offset);
            return ivec2(linearIndex % size.x, linearIndex / size.x);
        }

        vec2 uint2fp16x2(in uint packedData) {
            return unpackHalf2x16(packedData);
        }

        const float inv255 = 1.0 / 255.0;
        vec4 uint2rgba(in uint packedData) {
            float a = float((packedData >> 24) & 0xFFu) * inv255;
            float b = float((packedData >> 16) & 0xFFu) * inv255;
            float g = float((packedData >> 8)  & 0xFFu) * inv255;
            float r = float( packedData        & 0xFFu) * inv255;
            return vec4(r, g, b, a);
        }

        vec4 uint2vec4(in uint packedData, vec4 min, vec4 max) {
            return uint2rgba(packedData) * (max - min) + min;
        }

        vec3 sRGBToLinear(vec3 srgb)
        {
          return mix(srgb / 12.92, pow((srgb + 0.055) / 1.055, vec3(2.2)), step(0.04045, srgb));
        }

        const float sqrt8 = sqrt(8.0);
        const float SH_C1 = 0.4886025119029199f;
        const float[5] SH_C2 = float[](1.0925484, -1.0925484, 0.3153916, -1.0925484, 0.5462742);
    `

    constructor(options, eventBus, graphicsAPI) {
        this.cacheShaders = options.cacheShaders;

        this.debug = options.debug;
        this.debugTF = {
            outName: 'debugOutput',
            tf: null,
            buffer: null,
            size: 0,
        };

        this.eventBus = eventBus;
        this.eventBus.on('buffersReady', this.onBuffersReady.bind(this));
        this.eventBus.on('sortDone', this.onSortDone.bind(this));
        this.graphicsAPI = graphicsAPI;
        this.programs = {};
        this.uniforms = {};
        this.attributes = {};
        this.vaos = {};
        this.vars = {
            'projectionMatrix': {
                'value': new Float32Array(16),
                'type': 'Matrix4fv',
                'transpose': false,
                'update': true,
            },
            'viewMatrix': {
                'value': new Float32Array(16),
                'type': 'Matrix4fv',
                'transpose': false,
                'update': true,
            },
            'cameraPosition': {
                'value': [0, 0, 0],
                'type': '3fv',
                'update': true,
            },
            'inverseFocalAdjustment': {
                'value': 1.0,
                'type': '1f',
                'update': true,
            },
            'focal': {
                'value': [0, 0],
                'type': '2fv',
                'update': true,
            },
            'invViewport': {
                'value': [0, 0],
                'type': '2fv',
                'update': true,
            },
            'orthoZoom': {
                'value': 1.0,
                'type': '1f',
                'update': true,
            },
            'orthographicMode': {
                'value': 0,
                'type': '1i',
                'update': true,
            },
            'splatCount': {
                'value': 0,
                'type': '1i',
                'update': true,
            },
            'splatScale': {
                'value': 1.0,
                'type': '1f',
                'update': true,
            },
            'frustumDilation': {
                'value': 0.1,
                'type': '1f',
                'update': true,
            },
            'alphaCullThreshold': {
                'value': 3 / 255,
                'type': '1f',
                'update': true,
            },
            'timestamp': {
                'value': 0.0,
                'type': '1f',
                'update': true,
            },
            'renderMode': {
                'value': 1,
                'type': '1i',
                'update': true,
            }
        };
        this.vbo = null;    // shared vertexBuffer

        // state
        this.key = '';
        this.ready = false;
    }

    updateUniform(name, value, check = false) {
        if (this.vars[name]) {
            if (!check || Utils.valueChanged(this.vars[name].value, value)) {
                this.vars[name].value = value;
                this.vars[name].update = true;
            }
        } else {
            console.warn('ShaderManager: No such vars: ', name);
        }
    }

    updateUniformTextures(buffers) {
        Object.values(buffers).forEach(value => {
            this.graphicsAPI.updateUniform(this.uniforms[this.key][value.name], '1i', value.bind);
        });
    }

    updateUniforms(force = false) {
        for (const [key, value] of Object.entries(this.vars)) {
            if (force || value.update) {
                this.graphicsAPI.updateUniform(this.uniforms[this.key][key], value.type, value.value, value.transpose);
                value.update = false;
            }
        }
    }

    debugLog() {
        const capturedData = this.graphicsAPI.getBufferData(this.debugTF);

        console.log("--- Captured Vertex Positions (from GPU) ---");
        for (let i = 0;i<1;++i) {
            const base = i * 16;
            console.log(`debugOutput 0: ${capturedData[base+0].toFixed(3)}, ${capturedData[base+1].toFixed(3)}, ${capturedData[base+2].toFixed(3)}, ${capturedData[base+3].toFixed(3)}`);
        }

    }

    onSortDone(indexArray) {
        this.graphicsAPI.updateBuffer(this.vaos[this.key].instanceIndexBuffer, indexArray);
    }

    async onBuffersReady({ data, sceneName }) {
        const sceneType = data.sceneType;
        if (data.ready || sceneType.virtualSequentialThreeD) {
            return;
        }
        this.ready = false;
        // if we do not use cache, keep old key to delete later
        const oldKey = this.key;

        let gsKernel;
        switch (GSType[data.gsType]) {
            case GSType.ThreeD:
                gsKernel = GSKernel_3DGS;
                break;
            case GSType.SPACETIME:
                gsKernel = GSKernel_SPACETIME;
                break;
            default:
                break;
        }
        const key = data.gsType + '/' + data.quality + '/' + data.chunkBased;

        let allSplatsOnTexture = data.num;
        if (data.chunkResolution) {
            allSplatsOnTexture = data.chunkResolution.width * data.chunkResolution.height * 256;
        }
        // if we have no cache for this program, build one
        if (!this.programs[key]) {
            const vs = this.createVS(data.buffers, gsKernel, data.chunkBased);
            const fs = this.createFS();
            console.log('build program for ', key)
            //console.log(vs);
            //console.log(fs);
            this.createProgram(key, vs, fs, this.debug ? [this.debugTF.outName] : null);
            this.vaos[key] = this.graphicsAPI.setupVAO(this.getAttribLoc(key, 'inPosition'), this.getAttribLoc(key, 'splatIndex'), allSplatsOnTexture, this.vbo);
            this.vbo = this.vaos[key].vertexBuffer;
        } else {
            this.graphicsAPI.rebuildInstanceBuffer2VAO(this.vaos[key], this.getAttribLoc(key, 'splatIndex'), allSplatsOnTexture, key);
        }
        
        if (this.debug) {
            const size = data.num * 4 * 4 * 4;
            const { tf, buffer } = this.graphicsAPI.setupTransformFeedback(size);
            this.debugTF.tf = tf;
            this.debugTF.buffer = buffer;
            this.debugTF.size = size;
        }

        this.key = key;
        this.ready = true;
        if (!this.cacheShaders && this.key != oldKey) {
            this.deleteProgram(oldKey);
        }
    }

    createVS(buffers, gsKernel, chunkBased) {
        let vs = `#version 300 es 
            precision highp float;
        `

        vs += `
            in uint splatIndex;
            in vec3 inPosition;
        `

        Object.values(buffers).forEach(value => {
            vs += `uniform highp ${value.format.includes('UI') ? 'u' : ''}sampler2D ${value.name};\n`
        });
        vs += `
            uniform mat4 projectionMatrix;
            uniform mat4 viewMatrix;
            uniform vec3 cameraPosition;
            uniform float inverseFocalAdjustment;
            uniform vec2 focal;
            uniform vec2 invViewport;
            uniform float orthoZoom;
            uniform int orthographicMode;

            uniform int splatCount;
            uniform float splatScale;
            uniform float frustumDilation;
            uniform float alphaCullThreshold;
            uniform int renderMode;
        `;
        vs += gsKernel.getUniformDefines();

        vs += `
            out vec4 v_fragCol;
            out vec2 v_fragPos;
            ${this.debug ? `out vec4 debugOutput;` : ``}
        `;

        vs += ShaderManager.shaderHelperFunc;

        vs += gsKernel.getFetchFunc(buffers, chunkBased);

        vs += `
            void main()
            {
                vec3 splatCenter;
                vec4 splatColor;
                mat3 Vrk;
                ${gsKernel.getFetchParams(chunkBased)}

                vec4 viewCenter = viewMatrix * vec4(splatCenter, 1.0);
                vec4 clipCenter = projectionMatrix * viewCenter;
                vec2 fragPos = inPosition.xy;
                v_fragPos = fragPos * sqrt8;

                ${gsKernel.getSpecificCode(buffers)}

                // culling
                float clip = (1.0 + frustumDilation) * clipCenter.w;
                if(abs(clipCenter.x) > clip || abs(clipCenter.y) > clip
                    || clipCenter.z < -clipCenter.w || clipCenter.z > clipCenter.w
                    || splatColor.a < alphaCullThreshold)
                {
                    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                    return;
                }

                v_fragCol = splatColor;

                float s = 1.0 / (viewCenter.z * viewCenter.z);
                mat3 J = mat3(focal.x / viewCenter.z, 0., -(focal.x * viewCenter.x) * s, 0.,
                    focal.y / viewCenter.z, -(focal.y * viewCenter.y) * s, 0., 0., 0.);
                mat3 W = transpose(mat3(viewMatrix));
                mat3 T = W * J;

                mat3 cov2Dm = transpose(T) * Vrk * T;
                cov2Dm[0][0] += 0.3;
                cov2Dm[1][1] += 0.3;
                vec3 cov2Dv = vec3(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);
                vec3 ndcCenter = clipCenter.xyz / clipCenter.w;

                float a           = cov2Dv.x;
                float d           = cov2Dv.z;
                float b           = cov2Dv.y;
                float D           = a * d - b * b;
                float trace       = a + d;
                float traceOver2  = 0.5 * trace;
                float term2       = sqrt(max(0.1f, traceOver2 * traceOver2 - D));
                float       eigenValue1 = traceOver2 + term2;
                float       eigenValue2 = traceOver2 - term2;

                if(eigenValue2 <= 0.0)
                {
                    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                    return;
                }

                if (renderMode == 2) {
                    eigenValue1 = eigenValue2 = 0.4; // point cloud
                }

                vec2 eigenVector1 = normalize(vec2(b, eigenValue1 - a));
                vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);
                vec2 basisVector1 = eigenVector1 * splatScale * min(sqrt8 * sqrt(eigenValue1), 2048.0);
                vec2 basisVector2 = eigenVector2 * splatScale * min(sqrt8 * sqrt(eigenValue2), 2048.0);

                vec2 ndcOffset = vec2(fragPos.x * basisVector1 + fragPos.y * basisVector2) * invViewport * 2.0 * inverseFocalAdjustment;
                vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);
                gl_Position        = quadPos;

                ${this.debug ? `
                //debugOutput = vec4(0, 1, 2, 3);
                ` : ``};
            }
        `;

        return vs;
    }

    createFS() {
        return `#version 300 es
            precision highp float;
            in vec4 v_fragCol;
            in vec2 v_fragPos;

            out vec4 out_FragColor;
        
            void main () {
                float A = dot(v_fragPos, v_fragPos);
                if (A > 8.0) discard;
                float opacity = exp(-0.5 * A) * v_fragCol.a;
                out_FragColor = vec4(v_fragCol.rgb * opacity, opacity);
            }
        `
    }

    createProgram(key, vsSrc, fsSrc, capture = null) {
        const program = this.graphicsAPI.setupProgram(vsSrc, fsSrc, capture);

        if (!program) {
            console.error(`Fail to create program for ${key}`);
        }
        this.programs[key] = program;
        this.saveLocations(key);
    }

    setPipeline(key = null) {
        key = key || this.key;
        const program = this.programs[key];
        if (!program) {
            console.warn(`Program "${key}" not found.`);
            return;
        }
        this.graphicsAPI.updateProgram(program);
        this.graphicsAPI.updateVertexInput(this.vaos[key].vao);
        this.graphicsAPI.disableCull();
        this.graphicsAPI.disableDepth();
    }

    getUniformLoc(key, name) {
        const locMap = this.uniforms[key];
        if (!locMap) return null;
        return locMap[name] || null;
    }

    getAttribLoc(key, name) {
        const locMap = this.attributes[key];
        if (!locMap) return -1;
        return locMap[name] !== undefined ? locMap[name] : -1;
    }

    getUniformMap(key) {
        const locMap = this.uniforms[key];
        if (!locMap) return null;
        return locMap || null;
    }

    getAttribMap(key) {
        const locMap = this.attributes[key];
        if (!locMap) return -1;
        return locMap !== undefined ? locMap : -1;
    }

    async deleteProgram(key) {
        const program = this.programs[key];
        if (program) {
            this.graphicsAPI.deleteProgram(program);
            this.programs[key] = undefined;
            this.uniforms[key] = undefined;
            this.attributes[key] = undefined;
            this.graphicsAPI.deleteVAO(this.vaos[key].vao);
            this.graphicsAPI.deleteBuffer(this.vaos[key].instanceIndexBuffer);
            this.vaos[key] = undefined;
        }
    }

    async deleteAllProgram() {
        const keys = Object.keys(this.programs);

        for (const key of keys) {
            await this.deleteProgram(key);
        }
    }

    saveLocations(key) {
        const program = this.programs[key];
        this.uniforms[key] = this.graphicsAPI.getUniform(program);
        this.attributes[key] = this.graphicsAPI.getAttrib(program);
    }
}