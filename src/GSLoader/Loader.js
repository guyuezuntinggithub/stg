import { ParserType, LoadType } from "../Global.js";
import { Utils } from "../Utils.js";
import { FileType } from "../Global.js";
import * as THREE from "three";

export class GSLoader {
    constructor(eventBus) {
        this.eventBus = eventBus;
        {   // drag to load files
            const dropZone = document.getElementById('drop-zone');
		    dropZone.addEventListener('dragover', (event) => {
		    	event.preventDefault();
		    });
		    dropZone.addEventListener('drop', async (event) => {
		    	event.preventDefault();

                const items = event.dataTransfer.items;
                if (this.currentFile || !items || items.length === 0) {
                    return;
                }
            
                const areAllFiles = Array.from(items).every(item => item.kind === 'file' && item.webkitGetAsEntry() && item.webkitGetAsEntry().isFile);
                if (areAllFiles) {
                    const files = Array.from(event.dataTransfer.files);
                    await this.handleFiles(files);
                } else {
                    const firstItem = items[0];
                    const entry = firstItem.webkitGetAsEntry();
                
                    if (entry) {
                        if (entry.isFile) {
                            entry.file(async file => {
                                await this.handleFiles([file]);
                            });
                        } else if (entry.isDirectory) {
                            const subEntries = await Utils.readFirstLevelDirectory(entry);
                            const firstLevelFiles = [];
                            for (const subEntry of subEntries) {
                                if (subEntry.isFile) {
                                    const file = await new Promise(resolve => subEntry.file(resolve));
                                    firstLevelFiles.push(file);
                                }
                            }
                            await this.handleFiles(firstLevelFiles, entry.name);
                        }
                    }
                }
		    });
        }

        // we wanna use worker as a module so that we can import
        this.worker = new Worker(new URL('Parser.js', import.meta.url), { type: 'module' });
        this.worker.onmessage = (event) => {
            const message = event.data;
            this.recvTime = performance.now();
            if (message.valid) {
                const data = message.data;
                data.name = data.name || Utils.extractFileName(data.file.name);
                data.uid = Utils.getRandomUID();
                data.transform = {
                    position: { x: 0, y: 0, z: 0 },
                    scale: { x: 1, y: 1, z: 1 },
                    rotation: { x: 0, y: 0, z: 0 },
                };
                data.appliedTransform = data.appliedTransform ? new THREE.Matrix4().fromArray(data.appliedTransform) : new THREE.Matrix4();
                data.modelMatrix = data.appliedTransform.clone();
                data.chunkBased = data.chunkBased || '';
                data.sceneType = Utils.getSceneType(data);
                console.log(`[${(this.recvTime - this.sendTime)}ms]`);
                this.eventBus.emit('buffersReady', {
                    data: data,
                    sceneName: data.name,
                });
            } else {
                console.log(`[${(this.recvTime - this.sendTime)}ms] GSLoader ERROR: ${message.error}`);
                this.eventBus.emit('noteExternalListener', {
                    failLoad: true,
                    error: message.error,
                });
            }
        };
        this.worker.onerror = (event) => {
            console.error('Worker error:', event.message);
            console.error('Filename:', event.filename);
            console.error('Line:', event.lineno);
            console.error('Error object:', event.error);
        };

        this.reader = new FileReader();
        this.reader.onload = (e) => {
            const content = e.target.result;
            console.log(`send file ${this.currentFile} to worker`);
            this.sendTime = performance.now();
            this.worker.postMessage({
                'type': LoadType.NATIVE,
                'parser': ParserType.CPU,
                'name': this.currentFile,
                'data': content,
                'quality': 'medium',
                'from': 'drag',
            }, [content]);
            this.noteExternalListener();
            this.reset();
        };

        this.currentFile = '';  // not blank => is loading
        this.sendTime = 0;
        this.recvTime = 0;
    }

    /**
     * 从本地服务器异步读取文件。
     * @param {string} filePath - 相对于HTML文件的文件路径，例如 './shaders/vertex.glsl'。
     * @param {string} [type='text'] - 您期望的文件格式。可选值: 'text', 'json', 'blob', 'arrayBuffer'。
     * @returns {Promise<string|object|Blob|ArrayBuffer|null>} - 返回一个包含文件内容的Promise，如果失败则返回null。
     */
    async readFileFromServer(filePath) {
        if (this.currentFile) {
            return;
        }
        this.currentFile = filePath;
        this.sendTime = performance.now();
        this.worker.postMessage({
            'type': LoadType.URL,
            'parser': ParserType.CPU,
            'name': this.currentFile,
            'data': null,
            'quality': 'medium',
            'from': 'url',
        });
        this.noteExternalListener();
        this.reset();
    }

    async readFileFromNative(file) {
        if (!this.currentFile) {
            this.currentFile = file.name;
			this.reader.readAsArrayBuffer(file);
		}
    }

    async handleFiles(files, directoryName = '') {
        files = files.filter(file => FileType[Utils.extractFileExtension(file.name).toUpperCase()] > 0);
        if (files.length === 0) {
            return;
        } else if (files.length === 1) {
            this.currentFile = files[0].name;
		    this.reader.readAsArrayBuffer(files[0]);
            return;
        }
        // else they might be sequential 3dgs files
        let allFileHasNumber = true;
        for (const file of files) {
            const number = Utils.extractFileNameIdx(Utils.extractFileName(file.name));
            if (number < 0) {
                allFileHasNumber = false;
                break;
            } else {
                file.frameIdx = number;
            }
        }
        if (!allFileHasNumber) {
            this.currentFile = files[0].name;
		    this.reader.readAsArrayBuffer(files[0]);
            return;
        }
        // else we have sequential 3dgs files
        files.sort((a, b) => a.frameIdx - b.frameIdx);
        // check to ensure sequential 3dgs files are consistent from (start + 0) to (start + n)
        const { isSequential, startFrame } = Utils.isFrameIdxSequential(files);
        if (!isSequential) {

            return;
        }
        // reverse to load the first frame at the end
        files.sort((a, b) => b.frameIdx - a.frameIdx);
        // emit virtual sequential 3dgs scene
        this.currentFile = directoryName;
        const data = {
            chunkBased: '',
            gsType: 'ThreeD',
            name: this.currentFile,
            uid: Utils.getRandomUID(),
            transform: {
                position: { x: 0, y: 0, z: 0 },
                scale: { x: 1, y: 1, z: 1 },
                rotation: { x: 0, y: 0, z: 0 },
            },
            appliedTransform: new THREE.Matrix4(),
            modelMatrix:  new THREE.Matrix4(),
            virtual: true,
            sequential: true,
            frameNum: files.length,
            startFrameIdx: startFrame,
            frames: new Array(files.length).fill(''),
            currentFrame: startFrame,
        };
        data.sceneType = Utils.getSceneType(data);
        this.eventBus.emit('buffersReady', {
            data: data,
            sceneName: data.name
        });
        // then emit real scene sequence
        const reader = new FileReader();
        for (const file of files) {
            await new Promise((resolve, reject) => {
                reader.onload = (e) => {
                    const content = e.target.result;
                    this.worker.postMessage({
                        'type': LoadType.NATIVE,
                        'parser': ParserType.CPU,
                        'name': file.name,
                        'data': content,
                        'quality': 'medium',
                        'from': 'drag',
                        'sequential': true,
                        'frameIdx': file.frameIdx,
                    }, [content]);
                    this.noteExternalListener(file.name);
                    if (file.frameIdx === startFrame + files.length - 1) {
                        // we have read all sequential scenes
                        this.reset();
                    }
                    resolve();
                };
                reader.onerror = reject;
                reader.readAsArrayBuffer(file);
            });
        }
    }

    noteExternalListener(name) {
        this.eventBus.emit('noteExternalListener', {
            startLoad: true,
            name: name || this.currentFile,
        });
    }

    reset() {
        this.currentFile = '';
    }
}