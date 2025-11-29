#include <emscripten/emscripten.h>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN
#endif


EXTERN EMSCRIPTEN_KEEPALIVE void sortIndexes(
    unsigned int* indexes, void* centers, float* modelViewProj,
    int* mappedDistances, unsigned int * frequencies, unsigned int* indexesOut, 
    unsigned int distanceMapRange, float timestamp, unsigned int sortCount, int gsType, bool chunkBased,
    float* debug
) {
    int maxDistance = -2147483640;
    int minDistance = 2147483640;

    if (gsType == 1) {   // ThreeD
        // always use int centers in this case
        float fMVPRow3[4] = { modelViewProj[2] * 4096.f, modelViewProj[6] * 4096.f, modelViewProj[10] * 4096.f, 1 };
        float* fCenters = (float*)centers;
#ifdef __wasm_simd128__
        float tempOut[4];
        v128_t b = wasm_v128_load(&fMVPRow3[0]);
        for (unsigned int i = 0; i < sortCount; i++) {
            v128_t a = wasm_v128_load(&fCenters[4 * indexes[i]]);
            v128_t prod = wasm_f32x4_mul(a, b);
            wasm_v128_store(&tempOut[0], prod);
            int distance = (int)(tempOut[0] + tempOut[1] + tempOut[2]);
            mappedDistances[i] = distance;
            if (distance > maxDistance) maxDistance = distance;
            if (distance < minDistance) minDistance = distance;
        }
#else
        for (unsigned int i = 0; i < sortCount; i++) {
            unsigned int indexOffset = 4 * (unsigned int)indexes[i];
            int distance =
                (int)((fMVPRow3[0] * fCenters[indexOffset] +
                       fMVPRow3[1] * fCenters[indexOffset + 1] +
                       fMVPRow3[2] * fCenters[indexOffset + 2]));
            mappedDistances[i] = distance;
            if (distance > maxDistance) maxDistance = distance;
            if (distance < minDistance) minDistance = distance;
        }
#endif
    } else if (gsType == 2) {   // SpaceTime
        // always use float centers for precision
        float fMVPRow3[4] = { modelViewProj[2] * 4096.f, modelViewProj[6] * 4096.f, modelViewProj[10] * 4096.f, 1 };
        float* fCenters = (float*)centers;
        float tmpCenters[4];
#ifdef __wasm_simd128__
        float deltaTPow[4] = { 1.0 };
        float tempOut[4];
        v128_t deltaTPowSIMD;
        v128_t mvpSIMD = wasm_v128_load(fMVPRow3);
        for (unsigned int i = 0; i < sortCount; i++) {
            unsigned int indexOffset = 13 * (unsigned int)indexes[i];

            deltaTPow[1] = timestamp - fCenters[indexOffset + 12];
            deltaTPow[2] = deltaTPow[1] * deltaTPow[1];
            deltaTPow[3] = deltaTPow[2] * deltaTPow[1];
            deltaTPowSIMD = wasm_v128_load(deltaTPow);

            {   // unroll for loop
                v128_t b = wasm_v128_load(fCenters + indexOffset + 0 * 4);
                v128_t prod = wasm_f32x4_mul(deltaTPowSIMD, b);
                wasm_v128_store(tempOut, prod);
                tmpCenters[0] = tempOut[0] + tempOut[1] + tempOut[2] + tempOut[3];

                b = wasm_v128_load(fCenters + indexOffset + 1 * 4);
                prod = wasm_f32x4_mul(deltaTPowSIMD, b);
                wasm_v128_store(tempOut, prod);
                tmpCenters[1] = tempOut[0] + tempOut[1] + tempOut[2] + tempOut[3];

                b = wasm_v128_load(fCenters + indexOffset + 2 * 4);
                prod = wasm_f32x4_mul(deltaTPowSIMD, b);
                wasm_v128_store(tempOut, prod);
                tmpCenters[2] = tempOut[0] + tempOut[1] + tempOut[2] + tempOut[3];
            }
            v128_t centerSIMD = wasm_v128_load(tmpCenters);
            v128_t distSIMD = wasm_f32x4_mul(mvpSIMD, centerSIMD);
            wasm_v128_store(tempOut, distSIMD);
            int distance = (int)(tempOut[0] + tempOut[1] + tempOut[2]);
            mappedDistances[i] = distance;
            if (distance > maxDistance) maxDistance = distance;
            if (distance < minDistance) minDistance = distance;
        }
#else
        float deltaTPow, deltaT;
        for (unsigned int i = 0; i < sortCount; i++) {
            unsigned int indexOffset = 13 * (unsigned int)indexes[i];
            deltaTPow = 1.0f;
            deltaT = timestamp - fCenters[indexOffset + 12];
            tmpCenters[0] = tmpCenters[1] = tmpCenters[2] = 0.0f;
            for (int j = 0; j < 4; j++) {
                tmpCenters[0] += fCenters[indexOffset + 0 + j] * deltaTPow;
                tmpCenters[1] += fCenters[indexOffset + 4 + j] * deltaTPow;
                tmpCenters[2] += fCenters[indexOffset + 8 + j] * deltaTPow;
                deltaTPow *= deltaT;
            }
            int distance = (int)(fMVPRow3[0] * tmpCenters[0] + fMVPRow3[1] * tmpCenters[1] + fMVPRow3[2] * tmpCenters[2]);
            mappedDistances[i] = distance;
            if (distance > maxDistance) maxDistance = distance;
            if (distance < minDistance) minDistance = distance;
        }
#endif
    }

    float distancesRange = (float)maxDistance - (float)minDistance;
    float rangeMap = (float)(distanceMapRange - 1) / distancesRange;

    for (unsigned int i = 0; i < sortCount; i++) {
        unsigned int frequenciesIndex = (int)((float)(mappedDistances[i] - minDistance) * rangeMap);
        mappedDistances[i] = frequenciesIndex;
        frequencies[frequenciesIndex] = frequencies[frequenciesIndex] + 1;   
    }

    unsigned int cumulativeFreq = frequencies[0];
    for (unsigned int i = 1; i < distanceMapRange; i++) {
        unsigned int freq = frequencies[i];
        cumulativeFreq += freq;
        frequencies[i] = cumulativeFreq;
    }

    for (int i = (int)0 - 1; i >= 0; i--) {
        indexesOut[i] = indexes[i];
    }

    for (int i = (int)sortCount - 1; i >= (int)0; i--) {
        unsigned int frequenciesIndex = mappedDistances[i];
        unsigned int freq = frequencies[frequenciesIndex];
        indexesOut[sortCount - freq] = indexes[i];
        frequencies[frequenciesIndex] = freq - 1;
    }
}

struct BVHNode {
    int centerX;
    int centerY;
    int centerZ;
    int extentX;
    int extentY;
    int extentZ;
    int chunkStart;
    int chunkNum;
};

class Plane {
private:
    long long A, B, C, D;
    long long absA, absB, absC;
public:
    Plane() = default;
    Plane(const float p[4], float quantization_factor) {
        float temp[4] = {p[0], p[1], p[2], p[3]};

        float mag = std::sqrt(temp[0]*temp[0] + temp[1]*temp[1] + temp[2]*temp[2]);
        if (mag > 1e-6f) {
            for (int i = 0; i < 4; ++i) {
                temp[i] /= mag;
            }
        }
        A = static_cast<long long>(temp[0] * quantization_factor);
        B = static_cast<long long>(temp[1] * quantization_factor);
        C = static_cast<long long>(temp[2] * quantization_factor);
        D = static_cast<long long>(temp[3] * quantization_factor);
        absA = std::abs(A);
        absB = std::abs(B);
        absC = std::abs(C);
    }

    bool isOutside(const BVHNode* node) const {
        long long dist = A * (long long)node->centerX + 
                         B * (long long)node->centerY + 
                         C * (long long)node->centerZ + D;
        long long radius = (long long)node->extentX * absA + 
                           (long long)node->extentY * absB + 
                           (long long)node->extentZ * absC;
        
        return dist + radius < 0;
    }
};

class Frustum {
private:
    std::array<Plane, 6> planes;
    static constexpr float QUANTIZATION_FACTOR = 1000.0f;

public:
    Frustum(const float* mvpMatrix) {
        // row0 = (M[0], M[4], M[8],  M[12])
        // row1 = (M[1], M[5], M[9],  M[13])
        // row2 = (M[2], M[6], M[10], M[14])
        // row3 = (M[3], M[7], M[11], M[15])
        float rows[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                rows[i][j] = mvpMatrix[i + j * 4];
            }
        }
        
        float p[4];
        // Left Plane: row3 + row0
        for(int i=0; i<4; ++i) p[i] = rows[3][i] + rows[0][i];
        planes[0] = Plane(p, QUANTIZATION_FACTOR);
        // Right Plane: row3 - row0
        for(int i=0; i<4; ++i) p[i] = rows[3][i] - rows[0][i];
        planes[1] = Plane(p, QUANTIZATION_FACTOR);
        // Bottom Plane: row3 + row1
        for(int i=0; i<4; ++i) p[i] = rows[3][i] + rows[1][i];
        planes[2] = Plane(p, QUANTIZATION_FACTOR);
        // Top Plane: row3 - row1
        for(int i=0; i<4; ++i) p[i] = rows[3][i] - rows[1][i];
        planes[3] = Plane(p, QUANTIZATION_FACTOR);
        // Near Plane: row2
        for(int i=0; i<4; ++i) p[i] = rows[2][i];
        planes[4] = Plane(p, QUANTIZATION_FACTOR);
        // Far Plane: row3 - row2
        for(int i=0; i<4; ++i) p[i] = rows[3][i] - rows[2][i];
        planes[5] = Plane(p, QUANTIZATION_FACTOR);
    }

    bool isOutside(const BVHNode* node) const {
        for (int i = 0; i < 6; ++i) {
            if (planes[i].isOutside(node)) {
                return true;
            }
        }
        return false;
    }
};

unsigned int cullByBVH(unsigned int* indexes, float* modelViewProj,
    int* bvhNodes, int* chunkIndices, unsigned int* chunk2SplatsMapping,
    float* debug
) {
    unsigned int visibleSplats = 0;
    BVHNode* root = (BVHNode*)bvhNodes;
    Frustum frustum(modelViewProj);

    std::vector<int> stack;
    if (root) {
        stack.push_back(0);
    }

    while(!stack.empty()) {
        int nodeIndex = stack.back();
        stack.pop_back();

        const BVHNode* node = root + nodeIndex;

        if (frustum.isOutside(node)) {
            continue;
        }

        if (node->chunkNum > 0) { // Leaf node
            for (int i = 0; i < node->chunkNum; ++i) {
                unsigned int mappingOffset = chunkIndices[node->chunkStart + i] * 256;
                memcpy(indexes + visibleSplats, chunk2SplatsMapping + mappingOffset, 256 * 4);
                visibleSplats += 256;
            }
        } else { // recursive
            int leftChildIndex = node->chunkStart;
            int rightChildIndex = leftChildIndex + 1;
            stack.push_back(leftChildIndex);
            stack.push_back(rightChildIndex);
        }
    }
    return visibleSplats;
}