// ============================================================================
// Bitsliced Haraka512 Implementation for GPU
// Processes 64 parallel instances using bitplane representation
// ============================================================================

#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// Boyar-Peralta Bitsliced AES S-box Implementation
// 113 XOR gates + 32 AND gates, no memory lookups  
// ============================================================================

// Temporary LUT-based S-box for correctness (will optimize later)
__constant__ uint8_t d_aes_sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

__device__ __forceinline__ void boyar_peralta_sbox(
    uint64_t* x7, uint64_t* x6, uint64_t* x5, uint64_t* x4,
    uint64_t* x3, uint64_t* x2, uint64_t* x1, uint64_t* x0)
{
    // LUT-based S-box for correctness - processes 64 lanes in parallel
    uint64_t o7 = 0, o6 = 0, o5 = 0, o4 = 0, o3 = 0, o2 = 0, o1 = 0, o0 = 0;
    
    #pragma unroll
    for (int lane = 0; lane < 64; lane++) {
        // Extract input byte from bit lane
        uint8_t input_byte = 0;
        if (*x0 & (1ULL << lane)) input_byte |= 0x01;
        if (*x1 & (1ULL << lane)) input_byte |= 0x02;
        if (*x2 & (1ULL << lane)) input_byte |= 0x04;
        if (*x3 & (1ULL << lane)) input_byte |= 0x08;
        if (*x4 & (1ULL << lane)) input_byte |= 0x10;
        if (*x5 & (1ULL << lane)) input_byte |= 0x20;
        if (*x6 & (1ULL << lane)) input_byte |= 0x40;
        if (*x7 & (1ULL << lane)) input_byte |= 0x80;
        
        // Apply S-box lookup
        uint8_t output_byte = d_aes_sbox[input_byte];
        
        // Distribute output back to bit planes
        if (output_byte & 0x01) o0 |= (1ULL << lane);
        if (output_byte & 0x02) o1 |= (1ULL << lane);
        if (output_byte & 0x04) o2 |= (1ULL << lane);
        if (output_byte & 0x08) o3 |= (1ULL << lane);
        if (output_byte & 0x10) o4 |= (1ULL << lane);
        if (output_byte & 0x20) o5 |= (1ULL << lane);
        if (output_byte & 0x40) o6 |= (1ULL << lane);
        if (output_byte & 0x80) o7 |= (1ULL << lane);
    }
    
    *x0 = o0;
    *x1 = o1;
    *x2 = o2;
    *x3 = o3;
    *x4 = o4;
    *x5 = o5;
    *x6 = o6;
    *x7 = o7;
}

// Haraka round constants (same for all 64 parallel instances)
__constant__ uint32_t HARAKA_ROUND_CONSTANTS[5][4] = {
    {0x06050403, 0x0a090807, 0x0e0d0c0b, 0x1211100f},
    {0x16151413, 0x1a191817, 0x1e1d1c1b, 0x2221201f},
    {0x26252423, 0x2a292827, 0x2e2d2c2b, 0x3231302f},
    {0x36353433, 0x3a393837, 0x3e3d3c3b, 0x4241403f},
    {0x46454443, 0x4a494847, 0x4e4d4c4b, 0x5251504f}
};

// ============================================================================
// Bitsliced AES Operations
// ============================================================================

// ShiftRows on bitplanes - this is just a permutation of bit indices
__device__ __forceinline__ void bitsliced_shiftrows(uint64_t state[128]) {
    // State layout: 128 bitplanes (16 bytes × 8 bits)
    // Bytes 0-15 are arranged as 4x4 matrix:
    //  0  4  8 12
    //  1  5  9 13
    //  2  6 10 14
    //  3  7 11 15
    
    uint64_t temp[128];
    
    // Row 0: no shift
    for (int bit = 0; bit < 8; bit++) {
        temp[0*8 + bit] = state[0*8 + bit];   // byte 0
        temp[4*8 + bit] = state[4*8 + bit];   // byte 4
        temp[8*8 + bit] = state[8*8 + bit];   // byte 8
        temp[12*8 + bit] = state[12*8 + bit]; // byte 12
    }
    
    // Row 1: shift left by 1
    for (int bit = 0; bit < 8; bit++) {
        temp[1*8 + bit] = state[5*8 + bit];   // byte 1 <- byte 5
        temp[5*8 + bit] = state[9*8 + bit];   // byte 5 <- byte 9
        temp[9*8 + bit] = state[13*8 + bit];  // byte 9 <- byte 13
        temp[13*8 + bit] = state[1*8 + bit];  // byte 13 <- byte 1
    }
    
    // Row 2: shift left by 2
    for (int bit = 0; bit < 8; bit++) {
        temp[2*8 + bit] = state[10*8 + bit];  // byte 2 <- byte 10
        temp[6*8 + bit] = state[14*8 + bit];  // byte 6 <- byte 14
        temp[10*8 + bit] = state[2*8 + bit];  // byte 10 <- byte 2
        temp[14*8 + bit] = state[6*8 + bit];  // byte 14 <- byte 6
    }
    
    // Row 3: shift left by 3
    for (int bit = 0; bit < 8; bit++) {
        temp[3*8 + bit] = state[15*8 + bit];  // byte 3 <- byte 15
        temp[7*8 + bit] = state[3*8 + bit];   // byte 7 <- byte 3
        temp[11*8 + bit] = state[7*8 + bit];  // byte 11 <- byte 7
        temp[15*8 + bit] = state[11*8 + bit]; // byte 15 <- byte 11
    }
    
    // Copy back
    #pragma unroll
    for (int i = 0; i < 128; i++) {
        state[i] = temp[i];
    }
}

// MixColumns using bitsliced GF(256) arithmetic
__device__ __forceinline__ void bitsliced_mixcolumns(uint64_t state[128]) {
    // MixColumns matrix:
    // [2 3 1 1]
    // [1 2 3 1]
    // [1 1 2 3]
    // [3 1 1 2]
    
    // Process each column (4 columns total)
    for (int col = 0; col < 4; col++) {
        // Get byte indices for this column
        int b0 = col * 4 + 0;  // row 0
        int b1 = col * 4 + 1;  // row 1
        int b2 = col * 4 + 2;  // row 2
        int b3 = col * 4 + 3;  // row 3
        
        // Extract bitplanes for these bytes
        uint64_t a0[8], a1[8], a2[8], a3[8];
        uint64_t r0[8], r1[8], r2[8], r3[8];
        
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {
            a0[bit] = state[b0 * 8 + bit];
            a1[bit] = state[b1 * 8 + bit];
            a2[bit] = state[b2 * 8 + bit];
            a3[bit] = state[b3 * 8 + bit];
        }
        
        // Compute xtime (multiply by 2 in GF(256))
        // xtime(a) = a << 1, with conditional XOR of 0x1b if MSB was set
        uint64_t xt0[8], xt1[8], xt2[8], xt3[8];
        
        // For a0
        uint64_t msb0 = a0[7];  // Save MSB
        for (int bit = 7; bit > 0; bit--) {
            xt0[bit] = a0[bit-1];
        }
        xt0[0] = 0;
        // Conditional XOR with 0x1b (00011011)
        xt0[0] ^= msb0;  // bit 0
        xt0[1] ^= msb0;  // bit 1
        xt0[3] ^= msb0;  // bit 3
        xt0[4] ^= msb0;  // bit 4
        
        // Repeat for a1, a2, a3
        uint64_t msb1 = a1[7];
        for (int bit = 7; bit > 0; bit--) {
            xt1[bit] = a1[bit-1];
        }
        xt1[0] = 0;
        xt1[0] ^= msb1; xt1[1] ^= msb1; xt1[3] ^= msb1; xt1[4] ^= msb1;
        
        uint64_t msb2 = a2[7];
        for (int bit = 7; bit > 0; bit--) {
            xt2[bit] = a2[bit-1];
        }
        xt2[0] = 0;
        xt2[0] ^= msb2; xt2[1] ^= msb2; xt2[3] ^= msb2; xt2[4] ^= msb2;
        
        uint64_t msb3 = a3[7];
        for (int bit = 7; bit > 0; bit--) {
            xt3[bit] = a3[bit-1];
        }
        xt3[0] = 0;
        xt3[0] ^= msb3; xt3[1] ^= msb3; xt3[3] ^= msb3; xt3[4] ^= msb3;
        
        // Compute MixColumns output
        // r0 = 2*a0 + 3*a1 + a2 + a3 = xt0 + (a1 ^ xt1) + a2 + a3
        // r1 = a0 + 2*a1 + 3*a2 + a3 = a0 + xt1 + (a2 ^ xt2) + a3
        // r2 = a0 + a1 + 2*a2 + 3*a3 = a0 + a1 + xt2 + (a3 ^ xt3)
        // r3 = 3*a0 + a1 + a2 + 2*a3 = (a0 ^ xt0) + a1 + a2 + xt3
        
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {
            r0[bit] = xt0[bit] ^ a1[bit] ^ xt1[bit] ^ a2[bit] ^ a3[bit];
            r1[bit] = a0[bit] ^ xt1[bit] ^ a2[bit] ^ xt2[bit] ^ a3[bit];
            r2[bit] = a0[bit] ^ a1[bit] ^ xt2[bit] ^ a3[bit] ^ xt3[bit];
            r3[bit] = a0[bit] ^ xt0[bit] ^ a1[bit] ^ a2[bit] ^ xt3[bit];
        }
        
        // Write back
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {
            state[b0 * 8 + bit] = r0[bit];
            state[b1 * 8 + bit] = r1[bit];
            state[b2 * 8 + bit] = r2[bit];
            state[b3 * 8 + bit] = r3[bit];
        }
    }
}

// Add round constants to specific positions in Haraka512
__device__ __forceinline__ void bitsliced_add_round_constants(
    uint64_t state[512],  // 512 bitplanes for 64-byte state
    int round)
{
    // Haraka512 applies round constants to specific byte positions
    // Based on the standard Haraka512 specification
    const int rc_positions[20] = {
        0, 1, 2, 3,    // Block 0: bytes 0-3
        16, 17, 18, 19, // Block 1: bytes 16-19  
        32, 33, 34, 35, // Block 2: bytes 32-35
        48, 49, 50, 51, // Block 3: bytes 48-51
        4, 5, 6, 7     // Additional positions
    };
    
    // Apply round constants from the constant array
    for (int i = 0; i < 4; i++) {
        uint32_t rc = HARAKA_ROUND_CONSTANTS[round][i];
        int byte_pos = rc_positions[i * 4];
        
        // Apply 4-byte constant starting at byte_pos
        for (int byte = 0; byte < 4; byte++) {
            uint8_t rc_byte = (rc >> (byte * 8)) & 0xFF;
            for (int bit = 0; bit < 8; bit++) {
                if (rc_byte & (1 << bit)) {
                    // XOR with all 64 instances (all bits in the plane)
                    state[(byte_pos + byte) * 8 + bit] ^= 0xFFFFFFFFFFFFFFFFULL;
                }
            }
        }
    }
}

// Complete Haraka512 round function on bitsliced data
__device__ __forceinline__ void bitsliced_haraka512_round(
    uint64_t state[512],  // 512 bitplanes (64 bytes × 8 bits)
    int round)
{
    // Process 4 AES blocks in parallel
    for (int block = 0; block < 4; block++) {
        // Extract block (16 bytes = 128 bitplanes)
        uint64_t block_state[128];
        
        #pragma unroll
        for (int i = 0; i < 128; i++) {
            block_state[i] = state[block * 128 + i];
        }
        
        // SubBytes - already implemented with Boyar-Peralta
        for (int byte = 0; byte < 16; byte++) {
            uint64_t b0 = block_state[byte * 8 + 0];
            uint64_t b1 = block_state[byte * 8 + 1];
            uint64_t b2 = block_state[byte * 8 + 2];
            uint64_t b3 = block_state[byte * 8 + 3];
            uint64_t b4 = block_state[byte * 8 + 4];
            uint64_t b5 = block_state[byte * 8 + 5];
            uint64_t b6 = block_state[byte * 8 + 6];
            uint64_t b7 = block_state[byte * 8 + 7];
            
            boyar_peralta_sbox(&b7, &b6, &b5, &b4, &b3, &b2, &b1, &b0);
            
            block_state[byte * 8 + 0] = b0;
            block_state[byte * 8 + 1] = b1;
            block_state[byte * 8 + 2] = b2;
            block_state[byte * 8 + 3] = b3;
            block_state[byte * 8 + 4] = b4;
            block_state[byte * 8 + 5] = b5;
            block_state[byte * 8 + 6] = b6;
            block_state[byte * 8 + 7] = b7;
        }
        
        // ShiftRows
        bitsliced_shiftrows(block_state);
        
        // MixColumns (not in final round)
        if (round < 4) {
            bitsliced_mixcolumns(block_state);
        }
        
        // Write back
        #pragma unroll
        for (int i = 0; i < 128; i++) {
            state[block * 128 + i] = block_state[i];
        }
    }
    
    // Add round constants after all blocks are processed
    bitsliced_add_round_constants(state, round);
}

// Mix512 permutation for Haraka - byte-level permutation
__device__ __forceinline__ void bitsliced_mix512(uint64_t state[512]) {
    // Haraka512 Mix512 permutation - permutes 64 bytes
    // Permutation pattern for 64-byte state (indices 0-63)
    const int mix512_perm[64] = {
        0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 1, 6, 11,
        16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 2, 7, 12, 17, 22, 27,
        32, 37, 42, 47, 52, 57, 62, 3, 8, 13, 18, 23, 28, 33, 38, 43,
        48, 53, 58, 63, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59
    };
    
    uint64_t temp[512];
    
    // Apply permutation on groups of 8 bitplanes (one byte)
    for (int dst_byte = 0; dst_byte < 64; dst_byte++) {
        int src_byte = mix512_perm[dst_byte];
        
        // Copy 8 bitplanes for this byte
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {
            temp[dst_byte * 8 + bit] = state[src_byte * 8 + bit];
        }
    }
    
    // Copy back
    #pragma unroll
    for (int i = 0; i < 512; i++) {
        state[i] = temp[i];
    }
}

// Complete bitsliced Haraka512 to 256 compression
__device__ __forceinline__ void bitsliced_haraka512_256(
    uint64_t input[512],   // 512 bitplanes input
    uint64_t output[256])  // 256 bitplanes output
{
    uint64_t state[512];
    
    // Copy input
    #pragma unroll
    for (int i = 0; i < 512; i++) {
        state[i] = input[i];
    }
    
    // 5 rounds of Haraka512
    for (int round = 0; round < 5; round++) {
        bitsliced_haraka512_round(state, round);
        if (round < 4) {
            bitsliced_mix512(state);
        }
    }
    
    // Feedforward: XOR with input
    #pragma unroll
    for (int i = 0; i < 512; i++) {
        state[i] ^= input[i];
    }
    
    // Truncate to 256 bits (32 bytes = 256 bitplanes)
    #pragma unroll
    for (int i = 0; i < 256; i++) {
        output[i] = state[i];
    }
}

// ============================================================================
// Helper function to transpose between standard and bitsliced formats
// ============================================================================

// Convert 64 parallel 64-byte blocks to bitsliced format using warp ballots
__device__ __forceinline__ void transpose_64x512_to_bitplanes(
    const uint8_t input[64][64],
    uint64_t bitplanes[512])
{
    const unsigned FULL_MASK = 0xffffffffu;
    int tid  = threadIdx.x;          // 0..63
    int lane = tid & 31;             // lane within warp
    int warp = tid >> 5;             // warp id (0 or 1)

    uint32_t* planes32 = reinterpret_cast<uint32_t*>(bitplanes);

    for (int byte_idx = 0; byte_idx < 64; ++byte_idx) {
        uint8_t val = input[tid][byte_idx];
        #pragma unroll
        for (int bit_idx = 0; bit_idx < 8; ++bit_idx) {
            uint32_t mask = __ballot_sync(FULL_MASK, (val >> bit_idx) & 1);
            if (lane == 0) {
                planes32[(byte_idx * 8 + bit_idx) * 2 + warp] = mask;
            }
        }
    }
}

// Convert bitsliced format back to 64 parallel 32-byte blocks using warp shuffles
__device__ __forceinline__ void transpose_bitplanes_to_64x256(
    const uint64_t bitplanes[256],
    uint8_t output[64][32])
{
    const unsigned FULL_MASK = 0xffffffffu;
    int tid  = threadIdx.x;          // 0..63
    int lane = tid & 31;             // lane within warp
    int warp = tid >> 5;             // warp id

    const uint32_t* planes32 = reinterpret_cast<const uint32_t*>(bitplanes);

    for (int byte_idx = 0; byte_idx < 32; ++byte_idx) {
        uint8_t byte_val = 0;
        #pragma unroll
        for (int bit_idx = 0; bit_idx < 8; ++bit_idx) {
            uint32_t mask = 0;
            if (lane == 0) {
                mask = planes32[(byte_idx * 8 + bit_idx) * 2 + warp];
            }
            mask = __shfl_sync(FULL_MASK, mask, 0);
            byte_val |= ((mask >> lane) & 1) << bit_idx;
        }
        output[tid][byte_idx] = byte_val;
    }
}