# Bitsliced VerusHash GPU Miner

High-performance CUDA implementation of VerusHash v2.2 algorithm using bitslicing techniques for parallel processing.

## Features

- **Bitsliced Implementation**: Processes 64 hashes in parallel per warp using bitplane representation
- **Optimized CUDA Kernel**: Uses shared memory partitions to avoid race conditions
- **Full Stratum Support**: Complete mining pool integration with proper protocol handling  
- **Real Pool Difficulty**: Uses actual pool-provided share targets (no fake test targets)
- **Grid-Stride Processing**: Efficiently processes full batch sizes across all GPU cores
- **Cross-Platform**: Compatible with Windows and Linux

## Performance

- **Sustained Hashrate**: 2.1+ MH/s on RTX 5070
- **Memory Optimized**: Per-warp shared memory partitions (36KB total)
- **Full Batch Processing**: Processes complete 1M nonce batches

## Hardware Requirements

- NVIDIA GPU with Compute Capability 5.0+
- CUDA Toolkit 11.0+
- 4GB+ GPU memory recommended

## Pool Configuration

Current configuration connects to:
- **Pool**: pool.verus.io:9998
- **Protocol**: Stratum v1
- **Algorithm**: VerusHash v2.2

## Building

### Windows (MSVC)
```bash
nvcc -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64\cl.exe" ^
     bitsliced_verushash_miner.cu -o miner.exe ^
     -std=c++17 -O3 --ptxas-options=-v -arch=sm_89 -lws2_32 -DCUDA_ARCH=89
```

### Linux (GCC)
```bash
nvcc bitsliced_verushash_miner.cu -o miner \
     -std=c++17 -O3 --ptxas-options=-v -arch=sm_89 -DCUDA_ARCH=89
```

## Usage

1. Update wallet address in `bitsliced_verushash_miner.cu`:
   ```cpp
   const char* WALLET_ADDRESS = "YOUR_WALLET_ADDRESS.WORKER_NAME";
   ```

2. Compile and run:
   ```bash
   ./miner
   ```

## Technical Details

### Bitsliced VerusHash Algorithm
- Uses Haraka512/256 with bitplane S-box implementation
- Processes headers in 32-byte chunks with streaming approach
- 112-byte Verus block headers with 32-byte nonce field

### CUDA Kernel Optimization  
- **Shared Memory Layout**: Header template + per-warp partitions
- **Thread Configuration**: 128 threads/block (4 warps) for optimal occupancy
- **Memory Access**: Aligned 32-bit nonce stores using nonce_offset parameter

### Stratum Protocol Implementation
- Complete job parsing with merkle branch handling
- Proper extranonce2 generation and coinbase assembly
- Clean jobs detection for stale work prevention
- Real-time difficulty/target updates from pool

## Files

- `bitsliced_verushash_miner.cu` - Main miner implementation
- `bitsliced_haraka.cuh` - Bitsliced Haraka512 CUDA implementation  
- `include/verushash.h` - VerusHash constants and definitions

## License

This implementation is for educational and research purposes.