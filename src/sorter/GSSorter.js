import { GSType } from "../Global.js";

export class GSSorter {
    constructor(options, eventBus) {
        this.sharedMemoryForWorkers = options.sharedMemoryForWorkers;
        this.enableSIMDInSort = options.enableSIMDInSort;
        this.enableDebugOutput = options.enableDebugOutput;

        this.worker = new Worker(new URL('./SortWorker.js', import.meta.url), { type: 'module' });
        this.sourceWasm = '';
        this.ready = false;
        this.sortRunning = false;
        this.chunkBased = false;
        this.sortForSkipFrame = false;

        this.splatSortCount = 0;
        this.splatCount = 0;    // max capacity
        this.currentSplatCount = 0;     // splat count of current scene (might be less than this.splatCount)

        this.lastSortTime = 0;
        this.lastCullTime = 0;

        this.sortWorkerSortedIndexes = null;

        this.eventBus = eventBus;
        this.eventBus.on('buffersReady', this.onBuffersReady.bind(this));
    }

    getLastSortTime() {
        return this.lastSortTime;
    }

    getLastCullTime() {
        return this.lastCullTime;
    }

    getSplatSortCount() {
        return this.splatSortCount;
    }

    async onBuffersReady({ data, sceneName }) {
        const sceneType = data.sceneType;
        if (sceneType.virtualSequentialThreeD) {
            return;
        }
        this.chunkBased = Boolean(data.chunkBased) && (!data.sequential);
        this.currentSplatCount = data.num;

        // we allocate space for all splats on texture if chunk-based
        let splatCapacityRequired = this.currentSplatCount;
        if (data.chunkResolution) {
            splatCapacityRequired = data.chunkResolution.width * data.chunkResolution.height * 256;
        }

        const refreshOnly = splatCapacityRequired <= this.splatCount;
        if (!refreshOnly) {
            this.ready = false;
            this.splatCount = splatCapacityRequired;
            this.initSorter(this.splatCount);
        }

        this.worker.postMessage({
            'init': {
                'refreshOnly': refreshOnly,
                'sorterWasmUrl': this.sourceWasm,
                'splatCount': this.currentSplatCount,
                'useSharedMemory': this.sharedMemoryForWorkers,
                'distanceMapRange': 1 << 16,
                'centers': data.sortBuffer,
                'gsType': GSType[data.gsType],
                'chunkBased': this.chunkBased,  // whether enable BVH culling(even it's chunkBased, u can diable it)
                'chunks': data.chunkBuffer,
                'chunkNum': data.chunkNum,
                'chunkResolution': data.chunkResolution,    // if the scene is chunkBased, this must be valid
            }
        }/*, [data.sortBuffer]*/);
    }

    sort(mvpMatrix, cameraPositionArray, timestamp, sortForSkipFrame) {
        const sortMessage = {
            'modelViewProj': mvpMatrix.elements,
            'cameraPosition': cameraPositionArray,
            'timestamp': sortForSkipFrame ? 0 : timestamp,
        };
        // NOTE: when rendering 4dgs, we should always sort for current timestamp.
        // when sharedMemory is not available and the scene is large, 
        // the high frequency of copying and allocation of worker message may cause `out of memory`
        // in case of that we allocate once and transfer objects between main thread and worker
        const transferables = [];
        if (!this.sharedMemoryForWorkers) {
            if (!this.sortWorkerSortedIndexes.length) {
                return; // sortWorkerSortedIndexes is not yet transferred back
            }
            sortMessage.sortedIndexes = this.sortWorkerSortedIndexes;
            transferables.push(sortMessage.sortedIndexes.buffer);
        }
        this.worker.postMessage({
            'sort': sortMessage
        }, transferables);
        if (sortForSkipFrame) {
            this.sortForSkipFrame = true;
        }
    }

    initSorter(splatCount) {
        this.worker.onmessage = (e) => {
            if (e.data.sortDone) {
                if (this.sharedMemoryForWorkers) {
                    // TODO
                } else {
                    this.sortWorkerSortedIndexes = e.data.sortedIndexes;

                    const sortedIndexes = this.sortWorkerSortedIndexes.slice(0, e.data.splatSortCount);
                    this.eventBus.emit('sortDone', sortedIndexes);
                    if (this.sortForSkipFrame) {
                        this.eventBus.emit('sortForSkipFrameDone', {});
                        this.sortForSkipFrame = false;
                    }
                }
                this.splatSortCount = e.data.splatSortCount;
                this.lastSortTime = e.data.sortTime;
                this.lastCullTime = e.data.cullTime;
                this.sortRunning = false;
                if (this.enableDebugOutput) {
                    console.log(`visible: ${this.splatSortCount}/${this.splatCount} (${(this.splatSortCount/this.splatCount*100).toFixed(2)}%)`,  
                        `cullTime: ${this.lastCullTime.toFixed(2)}ms`,
                        `sortTime: ${this.lastSortTime.toFixed(2)}ms`
                    );
                }
            } else if (e.data.sortCanceled) {
                this.sortRunning = false;
            } else if (e.data.sortSetupPhase1Complete) {
                if (this.sharedMemoryForWorkers) {
                    this.sortWorkerSortedIndexes = new Uint32Array(e.data.sortedIndexesBuffer,
                                                                   e.data.sortedIndexesOffset, splatCount);
                } else {
                    this.sortWorkerSortedIndexes = new Uint32Array(splatCount);
                }

                this.ready = true;
                console.log('Sorting web worker initialized successfully.');
            }
        };

        this.worker.onerror = (event) => {
            console.error('Worker error:', event.message);
            console.error('Filename:', event.filename);
            console.error('Line:', event.lineno);
            console.error('Error object:', event.error);
        };

        const SorterWasm = './wasm/sorter.wasm';
        const SorterWasmNoSIMD = './wasm/sorter_no_simd.wasm';
        const SorterWasmNoSIMDNonShared = './wasm/sorter_no_simd_non_shared.wasm';
        const SorterWasmNonShared = './wasm/sorter_non_shared.wasm';
        this.sourceWasm = SorterWasm;

        if (!this.enableSIMDInSort) {
            this.sourceWasm = this.sharedMemoryForWorkers ? SorterWasmNoSIMD : SorterWasmNoSIMDNonShared;
        } else {
            this.sourceWasm = this.sharedMemoryForWorkers ? SorterWasm : SorterWasmNonShared;
        }
    }

}