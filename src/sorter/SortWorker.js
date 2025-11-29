

let wasmInstance;
let wasmMemory;
let useSharedMemory;
let splatCount;
let splatSortCount;

let indexesToSortOffset;
let centersOffset;
let modelViewProjOffset;
let mappedDistancesOffset;
let frequenciesOffset;
let sortedIndexesOffset;
let debugOffset;

let memsetZero;
let distanceMapRange;
let gsType;
let transferablesortedIndexesOut;
const Constants = {
    BytesPerFloat: 4,
    BytesPerInt: 4,
    MemoryPageSize: 65536, // 64KB
    MaxScenes: 32
};
// chunk
const splatsPerChunk = 256;
let chunkBased;
let chunkNum;
let bvhNodes;
let chunk2SplatsMapping;


function sort(modelViewProj, timestamp) {
    // TODO
    // if chunkBased, cull on wasm. related value: splatSortCount, indexToSort
    const cullStartTime = performance.now();
    if (chunkBased) {
        splatSortCount = cullByBVH(new Uint32Array(wasmMemory, indexesToSortOffset, splatCount), 
            modelViewProj, bvhNodes, chunk2SplatsMapping, Boolean(gsType === 2) ? timestamp : null);
    }
    const sortStartTime = performance.now();
    if (!memsetZero) memsetZero = new Uint32Array(distanceMapRange);
    new Float32Array(wasmMemory, modelViewProjOffset, 16).set(modelViewProj);
    new Uint32Array(wasmMemory, frequenciesOffset, distanceMapRange).set(memsetZero);
    const wasmStartTime = performance.now();
    wasmInstance.exports.sortIndexes(
        indexesToSortOffset, centersOffset, modelViewProjOffset,
        mappedDistancesOffset, frequenciesOffset, sortedIndexesOffset, 
        distanceMapRange, timestamp, splatSortCount, gsType, chunkBased,
        debugOffset
    );
    const sortMessage = {
        'sortDone': true,
        'splatSortCount': splatSortCount,
        'sortTime': 0,
        'cullTime': 0,
    };
    const transferables = [];
    if (!useSharedMemory) {
        transferablesortedIndexesOut.set(new Uint32Array(wasmMemory, sortedIndexesOffset, splatSortCount));
        transferables.push(transferablesortedIndexesOut.buffer);
        sortMessage.sortedIndexes = transferablesortedIndexesOut;
    }
    const sortEndTime = performance.now();
    sortMessage.sortTime = sortEndTime - sortStartTime;
    sortMessage.cullTime = sortStartTime - cullStartTime;
    self.postMessage(sortMessage, transferables);
}

let isProcessing = false;
const messageQueue = [];

async function handleMessageQueue() {
    if (isProcessing) {
        return; // 如果正在处理，则退出
    }

    isProcessing = true;

    while (messageQueue.length > 0) {
        const message = messageQueue.shift();
        await processMessage(message);
    }
    
    isProcessing = false;
}

self.onmessage = async (e) => {
    messageQueue.push(e.data);
    handleMessageQueue();
};

async function processMessage(msg) {
    if (msg.sort) {
        if (!useSharedMemory) {
            transferablesortedIndexesOut = msg.sort.sortedIndexes;
        }
        sort(msg.sort.modelViewProj, msg.sort.timestamp);
    } else if (msg.init) {
        const data = msg.init;
        const refreshOnly = data.refreshOnly;
        if (!refreshOnly && (wasmInstance || wasmMemory)) {
            wasmInstance = null;
            wasmMemory = null;
        }

        // Yep, this is super hacky and gross :(
        splatCount = data.splatCount;
        useSharedMemory = data.useSharedMemory;
        distanceMapRange = data.distanceMapRange;
        gsType = data.gsType;
        chunkBased = data.chunkBased;
        chunkNum = data.chunkNum;
        if (chunkBased) {
            bvhNodes = buildBVH(new Float32Array(data.chunks), gsType === 2);
        }
        if (data.chunkResolution) {
            // splat index is not of consistency if chunkBased, so we need to use all splats on the texture
            splatCount = data.chunkResolution.width * data.chunkResolution.height * 256;
        }
        const CENTERS_BYTES_PER_ENTRY = 13 * 4;
        const matrixSize = 16 * Constants.BytesPerFloat;

        const memoryRequiredForIndexesToSort = splatCount * Constants.BytesPerInt;
        const memoryRequiredForCenters = splatCount * CENTERS_BYTES_PER_ENTRY;
        const memoryRequiredForModelViewProjectionMatrix = matrixSize;
        const memoryRequiredForMappedDistances = splatCount * Constants.BytesPerInt;
        const memoryRequiredForIntermediateSortBuffers = distanceMapRange * Constants.BytesPerInt;
        const memoryRequiredForSortedIndexes = splatCount * Constants.BytesPerInt;
        const memoryRequiredForDebug = 8 * 4;
        const extraMemory = Constants.MemoryPageSize * 32;
        const totalRequiredMemory = memoryRequiredForIndexesToSort +
                                    memoryRequiredForCenters +
                                    memoryRequiredForModelViewProjectionMatrix +
                                    memoryRequiredForMappedDistances +
                                    memoryRequiredForIntermediateSortBuffers +
                                    memoryRequiredForSortedIndexes +
                                    memoryRequiredForDebug +
                                    extraMemory;
        const totalPagesRequired = Math.floor(totalRequiredMemory / Constants.MemoryPageSize ) + 1;

        indexesToSortOffset = 0;
        centersOffset = indexesToSortOffset + memoryRequiredForIndexesToSort;
        modelViewProjOffset = centersOffset + memoryRequiredForCenters;
        mappedDistancesOffset = modelViewProjOffset + memoryRequiredForModelViewProjectionMatrix;
        frequenciesOffset = mappedDistancesOffset + memoryRequiredForMappedDistances;
        sortedIndexesOffset = frequenciesOffset + memoryRequiredForIntermediateSortBuffers;
        debugOffset = sortedIndexesOffset + memoryRequiredForSortedIndexes;

        if (!refreshOnly) {    
            const memory = new WebAssembly.Memory({
                initial: totalPagesRequired,
                maximum: totalPagesRequired,
                shared: useSharedMemory, // Use the flag here
            });

            const sorterWasmImport = {
                module: {},
                env: { memory: memory }
            };

            // Efficiently load the Wasm module from the provided URL
            try {
                const { instance } = await WebAssembly.instantiateStreaming(fetch(data.sorterWasmUrl), sorterWasmImport);
                wasmInstance = instance;
            } catch (error) {
                // Fallback for browsers that don't support instantiateStreaming (e.g., some Safari versions)
                const response = await fetch(data.sorterWasmUrl);
                const wasmBytes = await response.arrayBuffer();
                const wasmModule = await WebAssembly.compile(wasmBytes);
                wasmInstance = await WebAssembly.instantiate(wasmModule, sorterWasmImport);
            }
            wasmMemory = sorterWasmImport.env.memory.buffer;
        }

        // update centers
        new Uint32Array(wasmMemory, centersOffset, memoryRequiredForCenters / Constants.BytesPerInt)
            .set(new Uint32Array(data.centers));
        // update indexToSort
        if (!chunkBased) {
            const index2sort = new Uint32Array(wasmMemory, indexesToSortOffset, splatCount);
            for (let i = 0; i < splatCount; ++i) {
                index2sort[i] = i;
            }
            splatSortCount = splatCount;
        } else {
            chunk2SplatsMapping = new Uint32Array(chunkNum * splatsPerChunk);
            fillChunk2SplatsMapping(chunk2SplatsMapping, chunkNum, data.chunkResolution.width);
        }
        if (!refreshOnly) {
            console.log('setup sort worker', data.sorterWasmUrl)
            if (useSharedMemory) {
                self.postMessage({
                    'sortSetupPhase1Complete': true,
                    'sortedIndexesBuffer': wasmMemory,
                    'sortedIndexesOffset': sortedIndexesOffset,
                    'transformsBuffer': wasmMemory,
                });
            } else {
                self.postMessage({
                    'sortSetupPhase1Complete': true
                });
            }
        }
    }
};

/**
 * 使用更符合 JavaScript 习惯的、面向对象的方式重构 BVH 构建过程。
 * @param {Float32Array} aabbData - 包含所有包围盒数据的扁平数组。
 * @returns {object | null} - 返回 BVH 树的根节点对象，如果没有任何图元则返回 null。
 */
function buildBVH(aabbData, dynamic = false) {
    const dataPerChunk = dynamic ? 8 : 6;    
    const numBoxes = aabbData.length / dataPerChunk;
    if (numBoxes === 0) return null;

    const primitives = [];
    for (let i = 0; i < numBoxes; i++) {
        const offset = i * dataPerChunk;
        const primitive = {
            min: [aabbData[offset], aabbData[offset + 1], aabbData[offset + 2]],
            max: [aabbData[offset + 3], aabbData[offset + 4], aabbData[offset + 5]],
            center: [
                (aabbData[offset] + aabbData[offset + 3]) / 2,
                (aabbData[offset + 1] + aabbData[offset + 4]) / 2,
                (aabbData[offset + 2] + aabbData[offset + 5]) / 2,
            ],
            index: i
        };
        if (dynamic) {
            primitive.t_range = [aabbData[offset + 6], aabbData[offset + 7]];
        }
        primitives.push(primitive);
    }

    const MAX_PRIMITIVES_IN_NODE = 1;

    // 2. 递归构建函数 (核心变化)
    // 这个函数现在直接返回一个节点对象
    function buildNode(startIndex, endIndex) {
        // a. 计算当前节点的总包围盒
        const range = { min: [...primitives[startIndex].min], max: [...primitives[startIndex].max] };
        for (let i = startIndex + 1; i < endIndex; i++) {
            for(let j = 0; j < 3; ++j) {
                range.min[j] = Math.min(range.min[j], primitives[i].min[j]);
                range.max[j] = Math.max(range.max[j], primitives[i].max[j]);
            }
        }

        const boundingBox = {
            center: [(range.max[0] + range.min[0]) / 2, (range.max[1] + range.min[1]) / 2, (range.max[2] + range.min[2]) / 2],
            extent: [(range.max[0] - range.min[0]) / 2, (range.max[1] - range.min[1]) / 2, (range.max[2] - range.min[2]) / 2],
        }

        if (dynamic) {
            let t_range = [...primitives[startIndex].t_range];
            for (let i = startIndex + 1; i < endIndex; i++) {
                t_range[0] = Math.min(t_range[0], primitives[i].t_range[0]);
                t_range[1] = Math.max(t_range[1], primitives[i].t_range[1]);
            }
            boundingBox.t_range = t_range;
        }

        const numPrimitives = endIndex - startIndex;

        // b. 递归终止条件 -> 创建叶子节点
        if (numPrimitives <= MAX_PRIMITIVES_IN_NODE) {
            // 叶子节点直接包含图元索引列表
            const primitiveIndices = [];
            for (let i = startIndex; i < endIndex; i++) {
                primitiveIndices.push(primitives[i].index);
            }
            return {
                isLeaf: true,
                boundingBox: boundingBox,
                primitiveIndices: primitiveIndices
            };
        }

        // c. 递归步骤 -> 创建中间节点
        // i. 找到最长的轴进行分割
        const extent = boundingBox.extent;
        let splitAxis = 0;
        if (extent[1] > extent[0]) splitAxis = 1;
        if (extent[2] > extent[splitAxis]) splitAxis = 2;
        
        // ii. 找到分割点
        const mid = Math.floor((startIndex + endIndex) / 2);
        quickSelectWithHoare(primitives, startIndex, endIndex - 1, mid, (a, b) => a.center[splitAxis] - b.center[splitAxis]);

        // iii. 递归创建子节点
        const leftChild = buildNode(startIndex, mid);
        const rightChild = buildNode(mid, endIndex);

        // iv. 返回包含对子节点直接引用的中间节点
        return {
            isLeaf: false,
            boundingBox: boundingBox,
            leftChild: leftChild,
            rightChild: rightChild
        };
    }

    // 3. 启动构建过程并返回根节点
    return buildNode(0, primitives.length);
}

/**
 * Hoare 分区方案
 * 对数组 arr 的 [left, right] 部分进行分区。
 * 它返回一个索引 j，使得 arr[left...j] 中的所有元素都 <= 基准值，
 * 而 arr[j+1...right] 中的所有元素都 >= 基准值。
 * @param {Array<T>} arr - 数组
 * @param {number} left - 起始索引
 * @param {number} right - 结束索引
 * @param {Function} compare - 比较函数
 * @returns {number} - 分割点索引
 */
function hoarePartition(arr, left, right, compare) {
    const pivotValue = arr[Math.floor((left + right) / 2)];

    let i = left - 1;
    let j = right + 1;

    while (true) {
        // 从左向右找到第一个 >= pivot 的元素
        do { i++; } while (compare(arr[i], pivotValue) < 0);
        // 从右向左找到第一个 <= pivot 的元素
        do { j--; } while (compare(arr[j], pivotValue) > 0);
        if (i >= j) {
            return j;
        }
        swap(arr, i, j);
    }
}

/**
 * 基于 Hoare 分区的快速选择算法
 * 在数组 arr 的 [left, right] 范围内查找第 k 小的元素 (0-indexed)。
 * @param {Array<T>} arr - 数组
 * @param {number} left - 起始索引
 * @param {number} right - 结束索引
 * @param {number} k - 要查找的第 k 小的元素的索引
 * @param {Function} compare - 比较函数
 * @returns {T} - 第 k 小的元素
 */
function quickSelectWithHoare(arr, left, right, k, compare) {
    while (left < right) {
        const splitPoint = hoarePartition(arr, left, right, compare);

        if (k <= splitPoint) {
            right = splitPoint;
        } else {
            left = splitPoint + 1;
        }
    }
    return arr[k];
}
function swap(arr, i, j) { const temp = arr[i]; arr[i] = arr[j]; arr[j] = temp; }

/**
 * 
 * @param {Uint32Array} chunk2SplatsMapping 
 * @param {Number} chunkNum 
 * @param {Number} chunkWidth 
 */
function fillChunk2SplatsMapping(chunk2SplatsMapping, chunkNum, chunkWidth) {
    for (let chunkIndex = 0; chunkIndex < chunkNum; ++chunkIndex) {
        const chunk_w = chunkIndex % chunkWidth;
        const chunk_h = Math.floor(chunkIndex / chunkWidth);
        for (let h = 0; h < 16; ++h) {
            for (let w = 0; w < 16; ++w) {
                const splatIndex = 1 * w + 16 * chunk_w + 16 * chunkWidth * h + 16 * chunkWidth * 16 * chunk_h;
                chunk2SplatsMapping[chunkIndex * 256 + h * 16 + w] = splatIndex;
            }
        }
    }
}

// Frustum 和 Plane 类的定义基本保持不变，但 isOutside 方法需要稍作修改

class Plane {
    constructor(p) {
        const mag = Math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        if (mag > 1e-6) {
            this.A = p[0] / mag;
            this.B = p[1] / mag;
            this.C = p[2] / mag;
            this.D = p[3] / mag;
        } else {
            this.A = 0;
            this.B = 0;
            this.C = 0;
            this.D = 0;
        }
        
        this.absA = Math.abs(this.A);
        this.absB = Math.abs(this.B);
        this.absC = Math.abs(this.C);
    }
    
    /**
     * 检查一个 BVH 节点对象是否完全在该平面的外部。
     * @param {object} node - BVH 节点对象，包含 boundingBox 属性
     * @returns {boolean}
     */
    isOutside(node) {
        const centerX = node.boundingBox.center[0];
        const centerY = node.boundingBox.center[1];
        const centerZ = node.boundingBox.center[2];
        const extentX = node.boundingBox.extent[0];
        const extentY = node.boundingBox.extent[1];
        const extentZ = node.boundingBox.extent[2];
        
        const dist = this.A * centerX + this.B * centerY + this.C * centerZ + this.D;
        const radius = this.absA * extentX + this.absB * extentY + this.absC * extentZ;
        
        return dist + radius < -1;
    }
}

class Frustum {
    constructor(mvpMatrix) {
        this.planes = new Array(6);
        
        // MVP 矩阵是列主序的，将其转换为行主序以方便计算
        const rows = [
            [mvpMatrix[0], mvpMatrix[4], mvpMatrix[8], mvpMatrix[12]],
            [mvpMatrix[1], mvpMatrix[5], mvpMatrix[9], mvpMatrix[13]],
            [mvpMatrix[2], mvpMatrix[6], mvpMatrix[10], mvpMatrix[14]],
            [mvpMatrix[3], mvpMatrix[7], mvpMatrix[11], mvpMatrix[15]],
        ];

        const p = new Float32Array(4);

        // 左平面: row3 + row0
        for (let i = 0; i < 4; i++) p[i] = rows[3][i] + rows[0][i];
        this.planes[0] = new Plane(p);

        // 右平面: row3 - row0
        for (let i = 0; i < 4; i++) p[i] = rows[3][i] - rows[0][i];
        this.planes[1] = new Plane(p);

        // 底平面: row3 + row1
        for (let i = 0; i < 4; i++) p[i] = rows[3][i] + rows[1][i];
        this.planes[2] = new Plane(p);

        // 顶平面: row3 - row1
        for (let i = 0; i < 4; i++) p[i] = rows[3][i] - rows[1][i];
        this.planes[3] = new Plane(p);

        // 近平面: row2 (在C++代码中是 row3 + row2，但通常近平面是 row2 或 row3)
        // 您的代码是 `row2`，我们保持一致
        for (let i = 0; i < 4; i++) p[i] = rows[2][i];
        this.planes[4] = new Plane(p);
        
        // 远平面: row3 - row2
        for (let i = 0; i < 4; i++) p[i] = rows[3][i] - rows[2][i];
        this.planes[5] = new Plane(p);
    }

    isOutside(node) {
        for (let i = 0; i < 6; i++) {
            if (this.planes[i].isOutside(node)) {
                return true;
            }
        }
        return false;
    }
}

function isBeyondLifeRange(node, timestamp = null) {
    if (timestamp && node.boundingBox.t_range) {
        const t_range = node.boundingBox.t_range;
        return t_range[0] > timestamp || t_range[1] < timestamp;
    }
    return false;
}

/**
 * 使用 JS 对象树结构的 BVH 对图元（块）进行视锥剔除。
 * @param {Uint32Array} indexesTosort - 用于存放可见 splat 索引的输出数组。
 * @param {Float32Array} modelViewProj - 4x4 的模型-视图-投影矩阵。
 * @param {object} bvhRootNode - BVH 树的根节点对象。
 * @param {Uint32Array} chunk2SplatsMapping - 从块索引到 splat 索引的映射。
 * @returns {number} 可见 splat 的总数。
 */
function cullByBVH(indexesTosort, modelViewProj, bvhRootNode, chunk2SplatsMapping, timestamp = null) {
    let visibleSplats = 0;
    if (!bvhRootNode) return 0;

    const frustum = new Frustum(modelViewProj);
    const stack = [bvhRootNode]; // 栈中现在直接存放节点对象

    while (stack.length > 0) {
        const node = stack.pop();

        if (frustum.isOutside(node) || isBeyondLifeRange(node, timestamp)) {
            continue;
        }

        if (node.isLeaf) { // 如果是叶子节点
            // 遍历叶子节点中的所有块索引
            for (const chunkIndex of node.primitiveIndices) {
                const mappingOffset = chunkIndex * 256;
                const splatIndexes = chunk2SplatsMapping.subarray(mappingOffset, mappingOffset + 256);
                indexesTosort.subarray(visibleSplats, visibleSplats + 256).set(splatIndexes);
                visibleSplats += 256;
            }
        } else { // 如果是中间节点
            // 将子节点对象压入栈中
            stack.push(node.leftChild);
            stack.push(node.rightChild);
        }
    }

    return visibleSplats;
}