// ============================================================================
// Bitsliced Haraka512 Implementation for GPU
// Processes 64 parallel instances using bitplane representation
// ============================================================================

#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#define __device__
#define __forceinline__ inline
#define __constant__
#endif

// ============================================================================
// Boyar-Peralta Bitsliced AES S-box Implementation
// 113 XOR gates + 32 AND gates, no memory lookups  
// ============================================================================

__device__ __forceinline__ void boyar_peralta_sbox(
    uint64_t* x7, uint64_t* x6, uint64_t* x5, uint64_t* x4,
    uint64_t* x3, uint64_t* x2, uint64_t* x1, uint64_t* x0)
{
    // Load bitplanes
    uint64_t U0 = *x7, U1 = *x6, U2 = *x5, U3 = *x4;
    uint64_t U4 = *x3, U5 = *x2, U6 = *x1, U7 = *x0;

    // Linear preprocessing
    uint64_t T1 = U0 ^ U3;
    uint64_t T2 = U0 ^ U5;
    uint64_t T3 = U0 ^ U6;
    uint64_t T4 = U3 ^ U5;
    uint64_t T5 = U4 ^ U6;
    uint64_t T6 = T1 ^ T5;
    uint64_t T7 = U1 ^ U2;
    uint64_t T8 = U7 ^ T6;
    uint64_t T9 = U7 ^ T7;
    uint64_t T10 = T6 ^ T7;
    uint64_t T11 = U1 ^ U5;
    uint64_t T12 = U2 ^ U5;
    uint64_t T13 = T3 ^ T4;
    uint64_t T14 = T6 ^ T11;
    uint64_t T15 = T5 ^ T11;
    uint64_t T16 = T5 ^ T12;
    uint64_t T17 = T9 ^ T16;
    uint64_t T18 = U3 ^ U7;
    uint64_t T19 = T7 ^ T18;
    uint64_t T20 = T1 ^ T19;
    uint64_t T21 = U6 ^ U7;
    uint64_t T22 = T7 ^ T21;
    uint64_t T23 = T2 ^ T22;
    uint64_t T24 = T2 ^ T10;
    uint64_t T25 = T20 ^ T17;
    uint64_t T26 = T3 ^ T16;
    uint64_t T27 = T1 ^ T12;
    uint64_t D  = U7;

    // Non-linear transformation
    uint64_t M1 = T13 & T6;
    uint64_t M6 = T3 & T16;
    uint64_t M11 = T1 & T15;
    uint64_t M13 = (T4 & T27) ^ M11;
    uint64_t M15 = (T2 & T10) ^ M11;
    uint64_t M20 = T14 ^ M1 ^ (T23 & T8) ^ M13;
    uint64_t M21 = (T19 & D) ^ M1 ^ T24 ^ M15;
    uint64_t M22 = T26 ^ M6 ^ (T22 & T9) ^ M13;
    uint64_t M23 = (T20 & T17) ^ M6 ^ M15 ^ T25;
    uint64_t M25 = M22 & M20;
    uint64_t M37 = M21 ^ ((M20 ^ M21) & (M23 ^ M25));
    uint64_t M38 = M20 ^ M25 ^ (M21 | (M20 & M23));
    uint64_t M39 = M23 ^ ((M22 ^ M23) & (M21 ^ M25));
    uint64_t M40 = M22 ^ M25 ^ (M23 | (M21 & M22));
    uint64_t M41 = M38 ^ M40;
    uint64_t M42 = M37 ^ M39;
    uint64_t M43 = M37 ^ M38;
    uint64_t M44 = M39 ^ M40;
    uint64_t M45 = M42 ^ M41;
    uint64_t M46 = M44 & T6;
    uint64_t M47 = M40 & T8;
    uint64_t M48 = M39 & D;
    uint64_t M49 = M43 & T16;
    uint64_t M50 = M38 & T9;
    uint64_t M51 = M37 & T17;
    uint64_t M52 = M42 & T15;
    uint64_t M53 = M45 & T27;
    uint64_t M54 = M41 & T10;
    uint64_t M55 = M44 & T13;
    uint64_t M56 = M40 & T23;
    uint64_t M57 = M39 & T19;
    uint64_t M58 = M43 & T3;
    uint64_t M59 = M38 & T22;
    uint64_t M60 = M37 & T20;
    uint64_t M61 = M42 & T1;
    uint64_t M62 = M45 & T4;
    uint64_t M63 = M41 & T2;

    // Linear postprocessing
    uint64_t L0 = M61 ^ M62;
    uint64_t L1 = M50 ^ M56;
    uint64_t L2 = M46 ^ M48;
    uint64_t L3 = M47 ^ M55;
    uint64_t L4 = M54 ^ M58;
    uint64_t L5 = M49 ^ M61;
    uint64_t L6 = M62 ^ L5;
    uint64_t L7 = M46 ^ L3;
    uint64_t L8 = M51 ^ M59;
    uint64_t L9 = M52 ^ M53;
    uint64_t L10 = M53 ^ L4;
    uint64_t L11 = M60 ^ L2;
    uint64_t L12 = M48 ^ M51;
    uint64_t L13 = M50 ^ L0;
    uint64_t L14 = M52 ^ M61;
    uint64_t L15 = M55 ^ L1;
    uint64_t L16 = M56 ^ L0;
    uint64_t L17 = M57 ^ L1;
    uint64_t L18 = M58 ^ L8;
    uint64_t L19 = M63 ^ L4;
    uint64_t L20 = L0 ^ L1;
    uint64_t L21 = L1 ^ L7;
    uint64_t L22 = L3 ^ L12;
    uint64_t L23 = L18 ^ L2;
    uint64_t L24 = L15 ^ L9;
    uint64_t L25 = L6 ^ L10;
    uint64_t L26 = L7 ^ L9;
    uint64_t L27 = L8 ^ L10;
    uint64_t L28 = L11 ^ L14;
    uint64_t L29 = L11 ^ L17;

    *x7 = L6 ^ L24;
    *x6 = ~(L16 ^ L26);
    *x5 = ~(L19 ^ L28);
    *x4 = L6 ^ L21;
    *x3 = L20 ^ L22;
    *x2 = L25 ^ L29;
    *x1 = ~(L13 ^ L27);
    *x0 = ~(L6 ^ L23);
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

// Convert 64 parallel 64-byte blocks to bitsliced format
__device__ __forceinline__ void transpose_64x512_to_bitplanes(
    const uint8_t input[64][64],  // 64 instances of 64 bytes each
    uint64_t bitplanes[512])      // 512 bitplanes (64 bytes × 8 bits)
{
    for (int byte_idx = 0; byte_idx < 64; byte_idx++) {
        for (int bit_idx = 0; bit_idx < 8; bit_idx++) {
            uint64_t plane = 0;
            for (int instance = 0; instance < 64; instance++) {
                if (input[instance][byte_idx] & (1 << bit_idx)) {
                    plane |= (1ULL << instance);
                }
            }
            bitplanes[byte_idx * 8 + bit_idx] = plane;
        }
    }
}

// Convert bitsliced format back to 64 parallel 32-byte blocks
__device__ __forceinline__ void transpose_bitplanes_to_64x256(
    const uint64_t bitplanes[256],  // 256 bitplanes (32 bytes × 8 bits)
    uint8_t output[64][32])         // 64 instances of 32 bytes each
{
    for (int byte_idx = 0; byte_idx < 32; byte_idx++) {
        for (int instance = 0; instance < 64; instance++) {
            uint8_t byte_val = 0;
            for (int bit_idx = 0; bit_idx < 8; bit_idx++) {
                if (bitplanes[byte_idx * 8 + bit_idx] & (1ULL << instance)) {
                    byte_val |= (1 << bit_idx);
                }
            }
            output[instance][byte_idx] = byte_val;
        }
    }
}