@echo off
echo Building Bitsliced VerusHash Miner...

REM Check if CUDA compiler exists
where nvcc >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: nvcc not found. Please install CUDA Toolkit.
    pause
    exit /b 1
)

REM Build with Visual Studio compiler
nvcc -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64\cl.exe" ^
     bitsliced_verushash_miner.cu -o miner.exe ^
     -std=c++17 -O3 --ptxas-options=-v -arch=sm_89 -lws2_32 -DCUDA_ARCH=89

if %ERRORLEVEL% EQU 0 (
    echo Build successful! Run miner.exe to start mining.
) else (
    echo Build failed. Check error messages above.
)

pause