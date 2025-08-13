#!/bin/bash

echo "Building Bitsliced VerusHash Miner..."

# Check if CUDA compiler exists
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA Toolkit."
    exit 1
fi

# Build with GCC
nvcc bitsliced_verushash_miner.cu -o miner \
     -std=c++17 -O3 --ptxas-options=-v -arch=sm_89 -DCUDA_ARCH=89 -lfmt

if [ $? -eq 0 ]; then
    echo "Build successful! Run ./miner to start mining."
    chmod +x miner
else
    echo "Build failed. Check error messages above."
    exit 1
fi