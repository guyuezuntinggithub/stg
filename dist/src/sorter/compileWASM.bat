@echo off
if "%EMSDK%" == "" (
    call emsdk_env.bat
    echo ======================================
    echo Finished setting up EMSDK environment.
)
echo Find path: EMSDK=%EMSDK%

REM sharedMemory: true, useSIMD: true
call em++ -std=c++11 sorter.cpp -Os -s WASM=1 -s SIDE_MODULE=2 -o wasm/sorter.wasm -s IMPORTED_MEMORY=1 -s USE_PTHREADS=1 -msimd128
echo Finished compiling sorter.wasm with shared memory and SIMD support.

REM sharedMemory: false, useSIMD: true
call em++ -std=c++11 sorter.cpp -Os -s WASM=1 -s SIDE_MODULE=2 -o wasm/sorter_non_shared.wasm -s IMPORTED_MEMORY=1 -msimd128
echo Finished compiling sorter_non_shared.wasm with no shared memory and SIMD support.

REM sharedMemory: true, useSIMD: false
call em++ -std=c++11 sorter.cpp -Os -s WASM=1 -s SIDE_MODULE=2 -o wasm/sorter_no_simd.wasm -s IMPORTED_MEMORY=1 -s USE_PTHREADS=1
echo Finished compiling sorter_no_simd.wasm with shared memory and no SIMD support.

REM sharedMemory: false, useSIMD: false
call em++ -std=c++11 sorter.cpp -Os -s WASM=1 -s SIDE_MODULE=2 -o wasm/sorter_no_simd_non_shared.wasm -s IMPORTED_MEMORY=1
echo Finished compiling sorter_no_simd_non_shared.wasm with no shared memory and no SIMD support.