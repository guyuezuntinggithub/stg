import { Utils } from "./Utils.js";
import * as THREE from "three"

export class GSScene {
    constructor(options, eventBus, graphicsAPI) {
        this.destroyOnLoad = options.destroyOnLoad;
        this.enableDebugOutput = options.enableDebugOutput;

        this.eventBus = eventBus;
        this.eventBus.on('buffersReady', this.onBuffersReady.bind(this));
        this.scenes = {};
        this.graphicsAPI = graphicsAPI;

        // state
        this.currentUID = '';
        this.ready = false;
    }

    /**
     * Scene Descriptor
     * @param {Array<Object>} buffers descriptors of gaussian property buffers
     * @param {string} chunkBased if the scene is chunk-based
     * @param {ArrayBuffer} chunkBuffer use to generate BVH
     * @param {number} chunkNum count of chunks
     * @param {Object} chunkResolution { width, height }
     * @param {Object} file descriptor of input file
     * @param {string} gsType gaussian type
     * @param {Object} modelMatrix
     * @param {} appliedTransform
     * @param {string} name
     * @param {number} num
     * @param {string} quality
     * @param {ArrayBuffer} sortBuffer
     * @param {Object} transform
     * @param {string} uid
     */

    get currentScene() { return this.scenes[this.currentUID]; }
    get actualCurrentScene() {
        let currentScene = this.currentScene;

        if (currentScene && currentScene.virtual) {
            if (currentScene.sequential) {
                const frameUID = currentScene.frames[currentScene.currentFrame];
                currentScene = this.scenes[frameUID];
            }
        }
        return currentScene;
    }
    // default to return actual current scene property
    get splatNum() { return this.getSceneProperty(this.actualCurrentScene, 'num', 0); }
    get buffers() { return this.getSceneProperty(this.actualCurrentScene, 'buffers', null); }
    get name() { return this.getSceneProperty(this.actualCurrentScene, 'name', ''); }
    get uid() { return this.getSceneProperty(this.actualCurrentScene, 'uid', ''); }
    get transform() { return this.getSceneProperty(this.actualCurrentScene, 'transform', null); }
    get gsType() { return this.getSceneProperty(this.actualCurrentScene, 'gsType', 'none'); }
    get modelMatrix() { return this.getSceneProperty(this.actualCurrentScene, 'modelMatrix', null); }

    getSceneProperty(scene, property, defaultVal) {
        if (scene) {
            return scene[property] || defaultVal;
        }
        return defaultVal;
    }

    updateTextureBindings() {
        let uid = this.currentUID;
        const currentScene = this.currentScene;
        if (currentScene.virtual && currentScene.sequential) {
            uid = currentScene.frames[currentScene.currentFrame];
        }

        this.setupTex(uid);
    }

    updateVirtualSequentialThreeDFrame(loopedTime) {
        const scene = this.currentScene;
        if (scene && scene.sceneType.virtualSequentialThreeD) {
            const lastFrameIdx = scene.currentFrame;
            const currentFrameIdx = Math.floor(loopedTime) % scene.frameNum;
            
            if (currentFrameIdx !== lastFrameIdx && scene.prepared && this.ready) {
                scene.currentFrame = currentFrameIdx;
                const frameScene = this.scenes[scene.frames[scene.currentFrame]];
                this.eventBus.emit('buffersReady', {
                    data: frameScene,
                    sceneName: frameScene.name,
                });
            }
        }
    }

    setSceneReady() {
        // prepared: declare whether the scene is prepared by GSScene, GSSorter, ShaderManager through 'buffersReady' event.
        const currentScene = this.currentScene;
        const sceneType = currentScene.sceneType;
        if (sceneType.ThreeD || sceneType.STG) {
            currentScene.prepared = true;
        } else if (sceneType.virtualSequentialThreeD) {
            currentScene.prepared = true;
            for (const subScene of currentScene.frames) {
                this.scenes[subScene].prepared = true;
            }
        }
    }

    async onBuffersReady({ data, sceneName }) {
        const uid = data.uid;
        this.scenes[uid] = data;

        const sceneType = data.sceneType;
        if (sceneType.ThreeD || sceneType.STG) {
            this.ready = false;
            this.setupTex(uid);

            const oldUID = this.currentUID;
            this.currentUID = uid;
            this.ready = true;
            if (this.destroyOnLoad && (oldUID !== this.currentUID)) {
                this.removeScene(oldUID);
            }
        } else if (sceneType.virtualSequentialThreeD) {
            if (!data.prepared) {
                this.ready = false;
            }

            // set state: new scene is ready
            const oldUID = this.currentUID;
            this.currentUID = uid;
            if (this.destroyOnLoad && (oldUID !== this.currentUID)) {
                this.removeScene(oldUID);
            }
        } else if (sceneType.generalSequentialThreeD) {
            const loaded = Boolean(this.currentScene.frames[data.frameIdx - this.currentScene.startFrameIdx]);
            if (loaded) {
                this.setupTex(uid);
            } else {
                this.ready = false;
                this.setupTex(uid);
                this.currentScene.frames[data.frameIdx - this.currentScene.startFrameIdx] = uid;
                // 'cause we load the sequence reversely
                if (data.frameIdx == this.currentScene.startFrameIdx) {
                    this.currentScene.appliedTransform.copy(data.appliedTransform);
                    this.currentScene.modelMatrix.copy(data.modelMatrix);
                    this.ready = true;
                }
            }
        }
    }

    setupTex(sceneName) {
        let bindIndex = 0;
        Object.values(this.scenes[sceneName].buffers).forEach(value => {
            value.bind = value.bind || bindIndex;
            this.graphicsAPI.setupTexture(value);
            value.buffer = null;
            ++bindIndex;
        });
        if (this.enableDebugOutput) {
            console.log(this.scenes[sceneName]);
        }
    }

    async destoryOldScene(oldScene) {
        if (oldScene) {
            const scene = this.scenes[oldScene];
            const sceneType = scene.sceneType;
            if (sceneType.virtualSequentialThreeD) {
                for (const oldGeneralSequentialThreeD of scene.frames) {
                    this.destoryOldScene(oldGeneralSequentialThreeD);
                }
            } else {
                Object.values(scene.buffers).forEach(value => {
                    this.graphicsAPI.deleteTexture(value.texture);
                    value.buffer = null;
                    value.texture = null;
                });
                scene.file.data = null;
            }
            delete this.scenes[oldScene];
        }
        if (this.enableDebugOutput) {
            console.log('destory', this.scenes);
        }
    }

    removeScene(uid) {
        if (uid === this.currentUID) {
            this.ready = false;
        }
        this.destoryOldScene(uid);
    }

    switchToScene(uid) {
        const scene = this.scenes[uid];
        
        if (scene) {
            this.eventBus.emit('buffersReady', {
                data: scene,
                sceneName: scene.name,
            });
        }
        
    }

    updateTransform = function() {
        const scaleMat = new THREE.Matrix4();
        const tmpMat = new THREE.Matrix4();
        const euler = new THREE.Euler(0, 0, 0, 'ZXY');
        const deg2rad = Math.PI / 180;
        return function() {
            const transform = this.scenes[this.currentUID].transform;
            euler.set(transform.rotation.x * deg2rad, transform.rotation.y * deg2rad, transform.rotation.z * deg2rad);
            scaleMat.makeScale(transform.scale.x, transform.scale.x, transform.scale.x);
            tmpMat.makeRotationFromEuler(euler);
            tmpMat.multiply(scaleMat);
            tmpMat.setPosition(transform.position.x, transform.position.y, transform.position.z);

            const modelMatrix = this.scenes[this.currentUID].modelMatrix;
            modelMatrix.copy(this.scenes[this.currentUID].appliedTransform);
            modelMatrix.premultiply(tmpMat);
        }
    }();

    applyTransform() {
        this.scenes[this.currentUID].appliedTransform.copy(this.scenes[this.currentUID].modelMatrix);

        const transform = this.scenes[this.currentUID].transform;
        transform.position.x = 0;
        transform.position.y = 0;
        transform.position.z = 0;
        transform.rotation.x = 0;
        transform.rotation.y = 0;
        transform.rotation.z = 0;
        transform.scale.x = 1;
        transform.scale.y = 1;
        transform.scale.z = 1;
    }

    resetTransform() {
        this.scenes[this.currentUID].appliedTransform.fromArray(this.scenes[this.currentUID].file.json.nodes[0].matrix);
        this.scenes[this.currentUID].modelMatrix.copy(this.scenes[this.currentUID].appliedTransform);

        const transform = this.scenes[this.currentUID].transform;
        transform.position.x = 0;
        transform.position.y = 0;
        transform.position.z = 0;
        transform.rotation.x = 0;
        transform.rotation.y = 0;
        transform.rotation.z = 0;
        transform.scale.x = 1;
        transform.scale.y = 1;
        transform.scale.z = 1;
    }

    forceSort() {
        const currentScene = this.currentScene;
        if (this.ready && currentScene) {
            const virtualSequentialThreeD = currentScene.virtual && currentScene.sequential;
            return currentScene.gsType === 'SPACETIME' || virtualSequentialThreeD;
        }
        return false;
    }

    export() {
        const scene = this.currentScene;
        const sceneType = scene.sceneType;
        if (sceneType.ThreeD || sceneType.STG) {
            GSScene.exportGlbFile(this.modifyGlbJson(scene), this.name + '.glb');
        } else if (sceneType.virtualSequentialThreeD) {
            for (const subSceneUID of scene.frames) {
                const subScene = this.scenes[subSceneUID];
                GSScene.exportGlbFile(this.modifyGlbJson(subScene, scene.modelMatrix), Utils.extractFileName(subScene.file.name) + '.glb');
            }
        }
    }

    static debugUnpackBuffer(buffers, idx = 0) {
        const pospad = new DataView(buffers.pospad.buffer, buffers.pospad.bytesPerTexel * idx);
        console.log('pos', pospad.getFloat32(0, true), pospad.getFloat32(4, true), pospad.getFloat32(8, true))
        const covcol = new DataView(buffers.covcol.buffer, buffers.covcol.bytesPerTexel * idx);
        console.log('cov', Utils.uint162fp162f(covcol.getUint16(0, true)),
            Utils.uint162fp162f(covcol.getUint16(2, true)),
            Utils.uint162fp162f(covcol.getUint16(4, true)),
            Utils.uint162fp162f(covcol.getUint16(6, true)),
            Utils.uint162fp162f(covcol.getUint16(8, true)),
            Utils.uint162fp162f(covcol.getUint16(10, true))
        )
        console.log('col', 
            Utils.uint82float(covcol.getUint8(12, true)),
            Utils.uint82float(covcol.getUint8(13, true)),
            Utils.uint82float(covcol.getUint8(14, true)),
            Utils.uint82float(covcol.getUint8(15, true))
        )
    }

    static exportGlbFile(buffer, fileName) {
        // 1. 从 ArrayBuffer 创建一个 Blob
        const blob = new Blob([buffer], { type: 'model/gltf-binary' });

        // 2. 为 Blob 创建一个临时的 URL
        const url = URL.createObjectURL(blob);

        // 3. 创建一个隐藏的下载链接并配置它
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = fileName; // 设置下载文件名
        
        // 4. 将链接添加到文档中，模拟点击，然后移除
        document.body.appendChild(a);
        a.click();
        
        // 5. 清理：等待片刻后移除链接并释放 URL
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 100);
    }

    modifyGlbJson(scene, modelMatrix = null) {
        const originalGlbBuffer = scene.file.data;

        const dataView = new DataView(originalGlbBuffer);
        const textDecoder = new TextDecoder('utf-8');
        const textEncoder = new TextEncoder();

        // --- 步骤 1: 解析头部和旧的 JSON 块 ---

        // 检查 "glTF" 魔术字
        const magic = dataView.getUint32(0, true);
        if (magic !== 0x46546C67) {
            console.error("提供的文件不是有效的 GLB 文件。");
            return originalGlbBuffer;
        }

        const version = dataView.getUint32(4, true);
        const fileLength = dataView.getUint32(8, true);

        // 第一个块总是 JSON 块
        let byteOffset = 12;
        const jsonChunkLength = dataView.getUint32(byteOffset, true);
        byteOffset += 4;
        const jsonChunkType = dataView.getUint32(byteOffset, true);
        byteOffset += 4;

        if (jsonChunkType !== 0x4E4F534A) { // 'JSON'
            console.error("找不到 GLB 的 JSON 块。");
            return originalGlbBuffer;
        }

        const jsonChunkData = new Uint8Array(originalGlbBuffer, byteOffset, jsonChunkLength);
        // 计算旧 JSON 块对齐后的长度，以便找到 BIN 块的起始位置
        const oldPaddedJsonLength = (jsonChunkLength + 3) & ~3; 

        // --- 步骤 2: 解码并修改 JSON 对象 ---

        const jsonString = textDecoder.decode(jsonChunkData);
        let json = JSON.parse(jsonString);

        // 调用用户提供的函数来修改 JSON
        json.nodes[0].matrix = (modelMatrix || scene.modelMatrix).toArray();

        // --- 步骤 3: 重新编码新的 JSON 数据 ---

        const newJsonString = JSON.stringify(json);
        const newJsonChunkData = textEncoder.encode(newJsonString);
        const newJsonChunkLength = newJsonChunkData.length;

        // GLB 块必须是4字节对齐的。计算需要填充的空格数。
        const padding = (4 - (newJsonChunkLength % 4)) % 4;
        const newPaddedJsonLength = newJsonChunkLength + padding;

        // 创建一个包含新 JSON 数据和填充的 Uint8Array
        const paddedNewJsonData = new Uint8Array(newPaddedJsonLength);
        paddedNewJsonData.set(newJsonChunkData);
        // 用空格（0x20）填充
        for (let i = 0; i < padding; i++) {
            paddedNewJsonData[newJsonChunkLength + i] = 0x20;
        }

        // --- 步骤 4: 计算新文件总长度 ---

        const originalBinChunkAndHeader = originalGlbBuffer.slice(12 + 8 + oldPaddedJsonLength);
        const newFileLength = 12 + 8 + newPaddedJsonLength + originalBinChunkAndHeader.byteLength;

        // --- 步骤 5: 重新组装新的 GLB 文件 ---

        const newGlbBuffer = new ArrayBuffer(newFileLength);
        const newGlbData = new Uint8Array(newGlbBuffer);
        const newGlbDataView = new DataView(newGlbBuffer);

        // 写入新的 GLB 头部
        newGlbData.set(new Uint8Array(originalGlbBuffer, 0, 12)); // 复制旧头部
        newGlbDataView.setUint32(8, newFileLength, true); // 更新总长度

        // 写入新的 JSON 块
        let newByteOffset = 12;
        newGlbDataView.setUint32(newByteOffset, newPaddedJsonLength, true); // 新的 JSON 长度
        newByteOffset += 4;
        newGlbDataView.setUint32(newByteOffset, 0x4E4F534A, true); // 'JSON' 类型
        newByteOffset += 4;
        newGlbData.set(paddedNewJsonData, newByteOffset); // 新的、已填充的 JSON 数据
        newByteOffset += newPaddedJsonLength;

        // 写入原始的 BIN 块（包括其头部）
        newGlbData.set(new Uint8Array(originalBinChunkAndHeader), newByteOffset);

        return newGlbBuffer;
    }
}