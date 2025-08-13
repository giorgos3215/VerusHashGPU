// ================================================================
//       RTX 5070 - Bitsliced VerusHash Miner
//          Processing 64 Hashes in Parallel
//     Using Boyar-Peralta Bitsliced AES S-box
// ================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand_kernel.h>

#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#include <fstream>
#include <cctype>
#include <cstdio>
#include <cstdarg>


#include <fmt/core.h>
#include <fmt/color.h>

#include <fstream>
#include <cctype>


#include <fstream>
#include <cctype>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <fcntl.h>
#define SOCKET int
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#define closesocket close
#endif
#include "include/verushash.h"
#include "bitsliced_haraka.cuh"
#include "stratum_client.h"

// Missing function implementation
void verus_header_init(verus_header_t* header) {
    if (!header) return;
    memset(header->header_data, 0, VERUS_HEADER_SIZE);
    header->nonce_offset = 108;  // Standard Bitcoin/Verus nonce position
}

struct MinerConfig {
    std::string pool_url{"pool.verus.io"};
    int port{9998};
    std::string wallet;
    std::string worker;
    uint32_t batch_size{1048576};
};

// Bitslicing configuration
#define BITSLICE_WIDTH 64  // Process 64 hashes in parallel
#define WARP_SIZE 32

// Global stats
std::atomic<bool> g_mining_active{true};
std::atomic<uint64_t> g_total_hashes{0};
std::atomic<uint32_t> g_shares_found{0};
std::atomic<uint32_t> g_shares_accepted{0};
std::atomic<uint32_t> g_shares_rejected{0};

std::atomic<bool> g_summary_active{false};

void log_status(const char* fmt, ...) {
    if (g_summary_active.exchange(false)) std::printf("\n");
    va_list args;
    va_start(args, fmt);
    std::vprintf(fmt, args);
    va_end(args);
    std::printf("\n");
}

void log_error(const char* fmt, ...) {
    if (g_summary_active.exchange(false)) std::fprintf(stderr, "\n");
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
}

void log_success(const char* fmt, ...) {
    if (g_summary_active.exchange(false)) std::printf("\n");
    va_list args;
    va_start(args, fmt);
    std::vprintf(fmt, args);
    va_end(args);
    std::printf("\n");
}

void log_warn(const char* fmt, ...) {
    if (g_summary_active.exchange(false)) std::printf("\n");
    va_list args;
    va_start(args, fmt);
    std::vprintf(fmt, args);
    va_end(args);
    std::printf("\n");
}

void print_summary(double hashrate) {
    std::printf("\rHashrate: %.2f MH/s | Shares F:%u A:%u R:%u",
               hashrate,
               g_shares_found.load(),
               g_shares_accepted.load(),
               g_shares_rejected.load());
    std::fflush(stdout);
    g_summary_active = true;
}

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            log_error("CUDA Error at %s:%d - %s", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// Boyar-Peralta Bitsliced AES S-box (implemented in bitsliced_haraka.cuh)
// ============================================================================

// ============================================================================
// Bitplane Transposition Functions (moved to bitsliced_haraka.cuh)
// ============================================================================

// ============================================================================
// Bitsliced Haraka512 Implementation (moved to bitsliced_haraka.cuh)
// ============================================================================

// ============================================================================
// Bitsliced VerusHash v2.2 Implementation
// ============================================================================

__device__ __forceinline__ void bitsliced_verushash_v22(
    uint8_t headers[64][VERUS_HEADER_SIZE],  // 64 different headers (only nonce differs)
    uint8_t outputs[64][32],                 // 64 hash outputs
    uint64_t* state_planes,                  // 256 bitplanes for 32-byte state
    uint64_t* temp_planes,                   // 512 bitplanes for 64-byte working buffer
    uint8_t*  haraka_buf)                    // 64 instances of 64 bytes each
{
    // VerusHash v2.2 streaming - process headers in 32-byte chunks
    // Each chunk goes through Haraka512 to get 32-byte digest

    // Initialize state to zero
    #pragma unroll
    for (int i = 0; i < 256; i++) {
        state_planes[i] = 0;
    }

    auto haraka_inputs = (uint8_t (*)[64])haraka_buf;

    // Process each 32-byte chunk of the 112-byte headers
    for (int chunk_idx = 0; chunk_idx < 4; chunk_idx++) {  // 112/32 = 3.5, so 4 chunks
        int chunk_offset = chunk_idx * 32;

        // Prepare 64-byte input for Haraka512 (32 bytes state + 32 bytes chunk)
        for (int instance = 0; instance < 64; instance++) {
            // First 32 bytes: current state (converted from bitplanes)
            for (int byte_idx = 0; byte_idx < 32; byte_idx++) {
                uint8_t byte_val = 0;
                for (int bit_idx = 0; bit_idx < 8; bit_idx++) {
                    if (state_planes[byte_idx * 8 + bit_idx] & (1ULL << instance)) {
                        byte_val |= (1 << bit_idx);
                    }
                }
                haraka_inputs[instance][byte_idx] = byte_val;
            }

            // Second 32 bytes: chunk from header (with padding for last chunk)
            for (int j = 0; j < 32; j++) {
                if (chunk_offset + j < VERUS_HEADER_SIZE) {
                    haraka_inputs[instance][32 + j] = headers[instance][chunk_offset + j];
                } else {
                    haraka_inputs[instance][32 + j] = 0;  // Padding
                }
            }
        }

        // Transpose to bitplanes
        transpose_64x512_to_bitplanes(haraka_inputs, temp_planes);

        // Apply bitsliced Haraka512
        bitsliced_haraka512_256(temp_planes, state_planes);
    }

    // Convert final state back to normal format
    transpose_bitplanes_to_64x256(state_planes, outputs);
}

// ============================================================================
// Target Comparison Helper
// ============================================================================

__device__ __forceinline__ bool hash_leq_target_le(const uint8_t h[32], const uint8_t t[32]){
    // Both hash and target are in little-endian format
    // Compare as little-endian 256-bit numbers (i = 0 .. 31)
    for (int i = 31; i >= 0; --i) {
        if (h[i] < t[i]) return true;
        if (h[i] > t[i]) return false;
    }
    return true; // equal
}

// ============================================================================
// Bitsliced Mining Kernel
// ============================================================================

__global__ void __launch_bounds__(128, 4) bitsliced_mining_kernel(
    const uint8_t* __restrict__ verus_header,
    uint32_t nonce_offset,
    uint64_t nonce_start,
    uint32_t batch_size,
    const uint8_t* __restrict__ target_le,
    uint32_t* __restrict__ found_nonces,
    uint8_t* __restrict__ found_hashes,
    uint32_t* __restrict__ found_count)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    // Shared memory layout:
    // [header_template][per-warp headers][per-warp hashes][per-warp work buffers]
    extern __shared__ uint8_t smem[];
    uint8_t* shared_header = smem;

    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_in_block   = threadIdx.x / WARP_SIZE;

    const size_t PER_WARP_HDR_BYTES   = BITSLICE_WIDTH * VERUS_HEADER_SIZE; // 64 * 112
    const size_t PER_WARP_HASH_BYTES  = BITSLICE_WIDTH * 32;                // 64 * 32
    const size_t STATE_PLANES_BYTES   = 256 * sizeof(uint64_t);             // 2048
    const size_t TEMP_PLANES_BYTES    = 512 * sizeof(uint64_t);             // 4096
    const size_t HARAKA_INPUTS_BYTES  = 64 * 64;                             // 4096
    const size_t PER_WARP_WORK_BYTES  = STATE_PLANES_BYTES + TEMP_PLANES_BYTES + HARAKA_INPUTS_BYTES;

    uint8_t* region_base    = smem + VERUS_HEADER_SIZE;
    uint8_t* headers_flat   = region_base + warp_in_block * PER_WARP_HDR_BYTES;
    uint8_t* hashes_flat    = region_base + warps_per_block * PER_WARP_HDR_BYTES
                                           + warp_in_block * PER_WARP_HASH_BYTES;
    uint8_t* work_base      = region_base + warps_per_block * (PER_WARP_HDR_BYTES + PER_WARP_HASH_BYTES);
    uint8_t* work_flat      = work_base + warp_in_block * PER_WARP_WORK_BYTES;
    uint64_t* state_planes  = reinterpret_cast<uint64_t*>(work_flat);
    uint64_t* temp_planes   = reinterpret_cast<uint64_t*>(work_flat + STATE_PLANES_BYTES);
    uint8_t*  haraka_inputs = work_flat + STATE_PLANES_BYTES + TEMP_PLANES_BYTES;

    // Load header template once per block
    if (threadIdx.x < VERUS_HEADER_SIZE) {
        shared_header[threadIdx.x] = verus_header[threadIdx.x];
    }
    __syncthreads();
    
    const int warps_per_grid = (gridDim.x * blockDim.x) / WARP_SIZE;

    for (uint64_t warp_idx = warp;
         warp_idx * BITSLICE_WIDTH < batch_size;
         warp_idx += warps_per_grid)
    {
        const uint64_t base_nonce = nonce_start + warp_idx * BITSLICE_WIDTH;

        if (lane == 0) {
            // Build 64 headers into this warp's partition
            for (int i = 0; i < BITSLICE_WIDTH; ++i) {
                uint8_t* hdr = &headers_flat[i * VERUS_HEADER_SIZE];
                memcpy(hdr, shared_header, VERUS_HEADER_SIZE);

                const uint32_t n = (uint32_t)(base_nonce + i);
                // Use the nonce_offset parameter, aligned, single store
                *reinterpret_cast<uint32_t*>(hdr + nonce_offset) = n;
            }

            // Hash directly from shared memory
            auto Hdr  = (uint8_t (*)[VERUS_HEADER_SIZE])headers_flat;
            auto Hash = (uint8_t (*)[32])               hashes_flat;
            bitsliced_verushash_v22(Hdr, Hash, state_planes, temp_planes, haraka_inputs);

            // Compare/store results
            for (int i = 0; i < BITSLICE_WIDTH; ++i) {
                if (hash_leq_target_le(&hashes_flat[i*32], target_le)) {
                    uint32_t slot = atomicAdd(found_count, 1);
                    if (slot < 8) {
                        found_nonces[slot] = (uint32_t)(base_nonce + i);
                        memcpy(&found_hashes[slot * 32], &hashes_flat[i*32], 32);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Complete Stratum Client Implementation
// ============================================================================

class SimpleJsonParser {
public:
    static std::string extract_string(const std::string& json, const std::string& key) {
        std::string search = "\"" + key + "\":\"";
        size_t pos = json.find(search);
        if (pos != std::string::npos) {
            size_t start = pos + search.length();
            size_t end = json.find("\"", start);
            if (end != std::string::npos) {
                return json.substr(start, end - start);
            }
        }
        return "";
    }
    static int extract_int(const std::string& json, const std::string& key) {
        std::string search = "\"" + key + "\":";
        size_t pos = json.find(search);
        if (pos != std::string::npos) {
            size_t start = pos + search.length();
            while (start < json.size() && std::isspace(json[start])) start++;
            size_t end = start;
            while (end < json.size() && (std::isdigit(json[end]) || json[end]=='-' || json[end]=='+')) end++;
            if (end > start) {
                return std::stoi(json.substr(start, end - start));
            }
        }
        return 0;
    }
};

MinerConfig parse_config(int argc, char** argv) {
    MinerConfig cfg;
    std::string config_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--pool" && i + 1 < argc) {
            cfg.pool_url = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            cfg.port = std::stoi(argv[++i]);
        } else if (arg == "--wallet" && i + 1 < argc) {
            cfg.wallet = argv[++i];
        } else if (arg == "--worker" && i + 1 < argc) {
            cfg.worker = argv[++i];
        } else if (arg == "--batch" && i + 1 < argc) {
            cfg.batch_size = static_cast<uint32_t>(std::stoul(argv[++i]));
        }
    }

    if (!config_path.empty()) {
        std::ifstream f(config_path);
        if (f) {
            std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            std::string s;
            s = SimpleJsonParser::extract_string(json, "pool");
            if (!s.empty()) cfg.pool_url = s;
            s = SimpleJsonParser::extract_string(json, "wallet");
            if (!s.empty()) cfg.wallet = s;
            s = SimpleJsonParser::extract_string(json, "worker");
            if (!s.empty()) cfg.worker = s;
            int port = SimpleJsonParser::extract_int(json, "port");
            if (port > 0) cfg.port = port;
            int batch = SimpleJsonParser::extract_int(json, "batch");
            if (batch > 0) cfg.batch_size = static_cast<uint32_t>(batch);
        } else {
            std::cerr << "Could not open config file: " << config_path << std::endl;
        }
    }

    return cfg;
}

class BitslicedStratumClient {
private:
    SOCKET sock;
    bool connected;
    int message_id;

    std::string pool_host;
    int pool_port;
    std::string wallet_address;
    std::string worker_name;
    
    std::string job_id;
    std::string prevhash;
    std::string coinb1;
    std::string coinb2;
    std::string version;
    std::string nbits;
    std::string ntime;
    std::string current_ntime;
    
    // Proper Stratum fields
    std::string extranonce1;
    int extranonce2_size;
    std::vector<std::string> merkle_branch;
    double current_difficulty;
    std::string current_extranonce2;
    uint64_t extranonce2_counter;
    
    std::string hash_reserved_hex;
    uint8_t hash_reserved_bytes[32];
    bool have_hash_reserved;
    
    uint8_t share_target_le[32];
    bool have_share_target;
    
    uint8_t block_target_le[32];
    bool have_block_target;
    
    uint32_t difficulty_target;
    
    std::string receive_buffer;
    uint8_t target_le[32];
    uint32_t share_counter;
    bool need_fresh_job;
    bool have_target;
    bool clean_jobs_flag;
    
    // --- helpers: hex <-> bytes (portable) ---
    static inline int hexval(unsigned char c){
        if (c>='0' && c<='9') return c-'0';
        if (c>='a' && c<='f') return c-'a'+10;
        if (c>='A' && c<='F') return c-'A'+10;
        return -1;
    }
    
    static std::string to_hex(const uint8_t* p, size_t n) {
        static const char* hexd="0123456789abcdef";
        std::string s; s.resize(n*2);
        for (size_t i=0;i<n;i++){ s[2*i]=hexd[p[i]>>4]; s[2*i+1]=hexd[p[i]&0xF]; }
        return s;
    }
    
    // make extranonce2 of the exact size (bytes), LE counter encoded as bytes
    std::string make_extranonce2(uint64_t ctr, int bytes) {
        std::string s; s.resize(bytes*2);
        for (int i=0;i<bytes;i++){
            uint8_t b = (ctr >> (8*i)) & 0xff; // little-endian byte order
            static const char* hexd="0123456789abcdef";
            s[2*i]   = hexd[b>>4];
            s[2*i+1] = hexd[b&0xF];
        }
        return s;
    }
    
    // Compute little-endian 32-byte target for a given difficulty.
    // Base is Bitcoin's 0x1d00ffff target (share target scales from this).
    static void target_from_diff(double diff, uint8_t out[32]) {
        // Little-endian base target for nBits = 0x1d00ffff
        static const uint8_t BASE[32] = {
            0xff,0xff,0x00,0x00,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
        };
        if (!(diff > 0.0)) diff = 1.0;

        // Big-int division by a floating diff: process bytes MSB->LSB in LE array.
        // Keep a long double remainder R, invariant: 0 <= R < diff.
        // For each more-significant byte, form value = R*256 + BASE[i], then q=floor(value/diff).
        // Store q as the output byte and update R := value - q*diff.
        long double D = (long double)diff;
        long double R = 0.0L;

        // Work from most-significant byte to least; in LE, that's index 31 down to 0.
        for (int i = 31; i >= 0; --i) {
            long double value = R * 256.0L + (long double)BASE[i];
            long double q_ld = floorl(value / D);
            if (q_ld > 255.0L) q_ld = 255.0L;      // safety clamp
            unsigned int q = (unsigned int)q_ld;
            out[i] = (uint8_t)q;
            R = value - q_ld * D;                  // new remainder in [0, D)
        }
    }
    
    // Proper SHA256d implementation (double SHA256)
    static inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
    static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
    static inline uint32_t sigma0(uint32_t x) { return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22); }
    static inline uint32_t sigma1(uint32_t x) { return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25); }
    static inline uint32_t gamma0(uint32_t x) { return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3); }
    static inline uint32_t gamma1(uint32_t x) { return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10); }

    static void sha256_transform(uint32_t state[8], const uint8_t block[64]) {
        static const uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        };
        
        uint32_t w[64];
        uint32_t a, b, c, d, e, f, g, h, t1, t2;
        
        // Copy block to w (big-endian)
        for (int i = 0; i < 16; i++) {
            w[i] = (block[i*4] << 24) | (block[i*4+1] << 16) | (block[i*4+2] << 8) | block[i*4+3];
        }
        
        // Extend w
        for (int i = 16; i < 64; i++) {
            w[i] = gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16];
        }
        
        // Initialize working variables
        a = state[0]; b = state[1]; c = state[2]; d = state[3];
        e = state[4]; f = state[5]; g = state[6]; h = state[7];
        
        // Main loop
        for (int i = 0; i < 64; i++) {
            t1 = h + sigma1(e) + ch(e, f, g) + K[i] + w[i];
            t2 = sigma0(a) + maj(a, b, c);
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        
        // Add to state
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    static void sha256(const uint8_t* data, size_t len, uint8_t hash[32]) {
        uint32_t state[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        };
        
        uint8_t block[64];
        size_t i = 0;
        
        // Process full blocks
        while (i + 64 <= len) {
            sha256_transform(state, data + i);
            i += 64;
        }
        
        // Process final block with padding
        memset(block, 0, 64);
        memcpy(block, data + i, len - i);
        block[len - i] = 0x80;
        
        // If no room for length, process this block and start another
        if (len - i >= 56) {
            sha256_transform(state, block);
            memset(block, 0, 64);
        }
        
        // Append length in bits (big-endian)
        uint64_t bit_len = len * 8;
        for (int j = 7; j >= 0; j--) {
            block[56 + j] = bit_len & 0xff;
            bit_len >>= 8;
        }
        sha256_transform(state, block);
        
        // Output hash (big-endian)
        for (int j = 0; j < 8; j++) {
            hash[j*4] = (state[j] >> 24) & 0xff;
            hash[j*4+1] = (state[j] >> 16) & 0xff;
            hash[j*4+2] = (state[j] >> 8) & 0xff;
            hash[j*4+3] = state[j] & 0xff;
        }
    }

    static void sha256d(const uint8_t* in, size_t len, uint8_t out32[32]) {
        uint8_t temp[32];
        sha256(in, len, temp);
        sha256(temp, 32, out32);
    }
    
    std::string hex_encode(const uint8_t* data, size_t len) {
        std::string result;
        char hex[3];
        for (size_t i = 0; i < len; i++) {
            snprintf(hex, sizeof(hex), "%02x", data[i]);
            result += hex;
        }
        return result;
    }
    
    void hex_decode(const std::string& hex, uint8_t* out, size_t max_len) {
        size_t n = std::min(hex.size()/2, max_len);
        for (size_t i=0;i<n;i++){
            int hi = hexval((unsigned char)hex[2*i]);
            int lo = hexval((unsigned char)hex[2*i+1]);
            if (hi >= 0 && lo >= 0) {
                out[i] = (uint8_t)((hi<<4) | lo);
            }
        }
    }
    
    bool send_message(const std::string& message) {
        std::string msg = message + "\n";
        int result = send(sock, msg.c_str(), (int)msg.length(), 0);
        return result != SOCKET_ERROR;
    }
    
    std::string receive_line() {
        while (true) {
            size_t newline_pos = receive_buffer.find('\n');
            if (newline_pos != std::string::npos) {
                std::string line = receive_buffer.substr(0, newline_pos);
                receive_buffer.erase(0, newline_pos + 1);
                return line;
            }
            
            char buffer[4096];
            int result = recv(sock, buffer, sizeof(buffer) - 1, 0);
            if (result > 0) {
                buffer[result] = '\0';
                receive_buffer += std::string(buffer);
            } else {
                break;
            }
        }
        return "";
    }
    
    void build_verus_header_from_job(const std::string& extranonce2_hex, verus_header_t* header) {
        memset(header->header_data, 0, VERUS_HEADER_SIZE);

        // coinbase = coinb1 + extranonce1 + extranonce2 + coinb2
        std::string coinbase_hex = coinb1 + extranonce1 + extranonce2_hex + coinb2;

        // H(coinbase), then fold merkle_branch
        std::vector<uint8_t> buf(coinbase_hex.size()/2);
        hex_decode(coinbase_hex, buf.data(), buf.size());
        uint8_t root[32]; 
        sha256d(buf.data(), buf.size(), root);

        for (const auto& br : merkle_branch) {
            uint8_t b[32]; 
            hex_decode(br, b, 32);
            uint8_t cat[64];
            memcpy(cat,     root, 32);
            memcpy(cat+32,  b,    32);
            sha256d(cat, 64, root);
        }

        // Fill 112-byte header:
        // [0..3] version (LE of 4B)
        uint8_t tmp4[4], t32[32];
        hex_decode(version, tmp4, 4);
        for (int i=0;i<4;i++) header->header_data[i] = tmp4[3-i];  // LE

        // [4..35] prevhash (as little-endian)
        hex_decode(prevhash, t32, 32);
        for (int i=0;i<32;i++) header->header_data[4+i] = t32[31-i];

        // [36..67] merkle root (as little-endian)
        for (int i=0;i<32;i++) header->header_data[36+i] = root[31-i];

        // [68..99] hashReserved (zero unless pool supplied a field)
        if (hash_reserved_hex.size() >= 64) {
            hex_decode(hash_reserved_hex, t32, 32);
            for (int i=0;i<32;i++) header->header_data[68+i] = t32[31-i];
        }

        // [100..103] ntime (LE)
        hex_decode(ntime, tmp4, 4);
        for (int i=0;i<4;i++) header->header_data[100+i] = tmp4[3-i];

        // [104..107] nbits (LE)
        hex_decode(nbits, tmp4, 4);
        for (int i=0;i<4;i++) header->header_data[104+i] = tmp4[3-i];

        header->nonce_offset = 108; // 4 bytes
    }
    
public:
    BitslicedStratumClient(const std::string& host, int port,
                           const std::string& wallet, const std::string& worker)
        : sock(INVALID_SOCKET), connected(false), message_id(1),
          pool_host(host), pool_port(port), wallet_address(wallet), worker_name(worker),
          extranonce2_size(4), difficulty_target(0x00000400), share_counter(0),
          need_fresh_job(false), have_target(false), have_hash_reserved(false),
          have_share_target(false), have_block_target(false), current_difficulty(1.0),
          extranonce2_counter(0), clean_jobs_flag(false) {}
    
    bool connect_to_pool() {

        log_status("Connecting to VerusPool...");

        std::cout << "Connecting to " << pool_host << ":" << pool_port << "..." << std::endl;


        
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            log_error("WSAStartup failed");
            return false;
        }
#endif
        
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == INVALID_SOCKET) {
            log_error("Socket creation failed");
#ifdef _WIN32
            WSACleanup();
#endif
            return false;
        }
        
#ifdef _WIN32
        DWORD timeout = 30000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeout));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (char*)&timeout, sizeof(timeout));
#endif
        
        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(pool_port);
        
        // Resolve hostname
        struct hostent* host_entry = gethostbyname(pool_host.c_str());
        if (host_entry == nullptr) {
            log_error("DNS resolution failed");
            closesocket(sock);
#ifdef _WIN32
            WSACleanup();
#endif
            return false;
        }
        
        memcpy(&server_addr.sin_addr, host_entry->h_addr_list[0], host_entry->h_length);
        
        if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            log_error("Connection failed");
            closesocket(sock);
#ifdef _WIN32
            WSACleanup();
#endif
            return false;
        }

        // Set socket to non-blocking mode for asynchronous polling
#ifdef _WIN32
        u_long mode = 1;
        ioctlsocket(sock, FIONBIO, &mode);
#else
        int flags = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, flags | O_NONBLOCK);
#endif

        connected = true;

        log_status("CONNECTED to {}:{}", pool_host, pool_port);



        log_status("CONNECTED to %s:%d", POOL_HOST, POOL_PORT);


        log_status("CONNECTED to {}:{}", POOL_HOST, POOL_PORT);

        std::cout << "CONNECTED to " << pool_host << ":" << pool_port << std::endl;



        // Subscribe
        std::string subscribe_msg = "{\"id\": " + std::to_string(message_id++) + 
                                   ", \"method\": \"mining.subscribe\", \"params\": [\"BitslicedMiner/1.0\"]}";
        
        if (!send_message(subscribe_msg)) {
            log_error("Failed to send subscribe");
            return false;
        }
        
        std::string response = receive_line();
        log_status("Subscribe response: %s", response.c_str());
        
        // Parse subscribe result: [[subscriptions], extranonce1, extranonce2_size]
        if (response.find("\"result\"") != std::string::npos) {
            size_t rpos = response.find("\"result\"");
            size_t lb = response.find('[', rpos);
            size_t rb = response.rfind(']');
            if (lb != std::string::npos && rb != std::string::npos && rb > lb) {
                std::string arr = response.substr(lb, rb - lb + 1);
                
                // Find second quoted string (extranonce1)
                size_t q1 = arr.find('"'), q2 = arr.find('"', q1 + 1);
                q1 = arr.find('"', q2 + 1); 
                q2 = arr.find('"', q1 + 1);
                if (q1 != std::string::npos && q2 != std::string::npos) {
                    extranonce1 = arr.substr(q1 + 1, q2 - q1 - 1);
                    log_status("Extracted extranonce1: %s", extranonce1.c_str());
                }
                
                // Last integer is extranonce2_size
                size_t lastComma = arr.find_last_of(",]");
                if (lastComma != std::string::npos) {
                    extranonce2_size = atoi(arr.c_str() + lastComma + 1);
                    if (extranonce2_size <= 0 || extranonce2_size > 16) extranonce2_size = 4;
                    log_status("Extracted extranonce2_size: %d", extranonce2_size);
                }
            }
        }
        
        // Authorize
        std::string user = wallet_address;
        if (!worker_name.empty()) user += "." + worker_name;
        std::string auth_msg = "{\"id\": " + std::to_string(message_id++) +
                              ", \"method\": \"mining.authorize\", \"params\": [\"" + user + "\", \"\"]}";
        
        if (!send_message(auth_msg)) {
            log_error("Failed to send authorize");
            return false;
        }
        
        response = receive_line();
        log_status("Auth response: %s", response.c_str());
        
        // Process the auth response - might be a mining.set_target message
        if (response.find("mining.set_target") != std::string::npos) {
            // Look for params array: ["0000040000000000000000000000000000000000000000000000000000000000"]
            size_t bracket_start = response.find("[\"");
            size_t bracket_end = response.find("\"]");
            if (bracket_start != std::string::npos && bracket_end != std::string::npos && bracket_end > bracket_start) {
                std::string targ_hex = response.substr(bracket_start + 2, bracket_end - bracket_start - 2);
                log_status("Found target hex in auth: %s", targ_hex.c_str());
                // strict decode: 64 hex chars -> 32 bytes LE
                if (targ_hex.size() >= 64) {
                    for (int i=0;i<32;i++) {
                        int hi = hexval((unsigned char)targ_hex[i*2]);
                        int lo = hexval((unsigned char)targ_hex[i*2+1]);
                        if (hi >= 0 && lo >= 0) {
                            share_target_le[i] = (uint8_t)((hi<<4) | lo);
                        }
                    }
                    have_share_target = true;
                    log_status("Set target from pool: %s...", targ_hex.substr(0, 16).c_str());
                }
            }
        }
        
        return true;
    }
    
    bool get_work(verus_header_t* header) {
        if (!connected) return false;
        
        // Process incoming messages
        std::string message = receive_line();
        if (message.empty()) return false;
        
        log_status("Received message: %s", message.c_str());
        
        if (message.find("mining.set_difficulty") != std::string::npos) {
            size_t lb = message.find('['), rb = message.find(']');
            if (lb != std::string::npos && rb != std::string::npos) {
                current_difficulty = atof(message.substr(lb+1, rb-lb-1).c_str());
                have_share_target = false; // recompute from diff next time
                log_status("Set difficulty: %.2f", current_difficulty);
            }
            return false;
        }

        if (message.find("mining.set_target") != std::string::npos) {
            // params: ["<32-byte hex, little-endian>"]
            size_t q1 = message.find('"'), q2 = message.find('"', q1+1);
            if (q1 != std::string::npos && q2 != std::string::npos && q2 > q1+1) {
                std::string targ_hex = message.substr(q1+1, q2-q1-1);
                // strict decode: 64 hex chars -> 32 bytes LE
                if (targ_hex.size() >= 64) {
                    for (int i=0;i<32;i++) {
                        int hi = hexval((unsigned char)targ_hex[i*2]);
                        int lo = hexval((unsigned char)targ_hex[i*2+1]);
                        if (hi >= 0 && lo >= 0) {
                            share_target_le[i] = (uint8_t)((hi<<4) | lo);
                        }
                    }
                    have_share_target = true;
                    log_status("Set target from pool: %s...", targ_hex.substr(0, 16).c_str());
                    log_status("Parsed target (LE): %#x", *(uint32_t*)&share_target_le[28]);
                }
            }
            return false;
        }
        
        if (message.find("mining.notify") != std::string::npos) {
            log_status("Parsing mining.notify job...");
            
            // Parse Stratum params: [job_id, version, prevhash, coinb1, coinb2, merkle_branch[], ntime, nbits, clean_jobs]
            size_t params_start = message.find("\"params\":[");
            if (params_start != std::string::npos) {
                std::string params_section = message.substr(params_start + 9); // Skip "params":[
                
                // Parse each field in order with proper handling
                std::vector<std::string> string_fields;
                std::vector<std::string> merkle_array;
                size_t pos = 1; // Skip opening [
                
                // Parse first 5 string fields: job_id, version, prevhash, coinb1, coinb2
                for (int i = 0; i < 5; i++) {
                    // Skip whitespace and commas
                    while (pos < params_section.length() && (params_section[pos] == ' ' || params_section[pos] == ',' || params_section[pos] == '\n')) pos++;
                    
                    if (params_section[pos] == '"') {
                        size_t start = pos + 1;
                        size_t end = params_section.find('"', start);
                        if (end != std::string::npos) {
                            string_fields.push_back(params_section.substr(start, end - start));
                            pos = end + 1;
                        } else break;
                    } else break;
                }
                
                // Parse merkle_branch array (element 5)
                while (pos < params_section.length() && (params_section[pos] == ' ' || params_section[pos] == ',' || params_section[pos] == '\n')) pos++;
                
                if (params_section[pos] == '[') {
                    pos++; // Skip opening [
                    
                    while (pos < params_section.length()) {
                        // Skip whitespace and commas
                        while (pos < params_section.length() && (params_section[pos] == ' ' || params_section[pos] == ',' || params_section[pos] == '\n')) pos++;
                        
                        if (params_section[pos] == ']') {
                            pos++; // Skip closing ]
                            break;
                        } else if (params_section[pos] == '"') {
                            size_t start = pos + 1;
                            size_t end = params_section.find('"', start);
                            if (end != std::string::npos) {
                                merkle_array.push_back(params_section.substr(start, end - start));
                                pos = end + 1;
                            } else break;
                        } else {
                            pos++;
                        }
                    }
                }
                
                // Parse final 2 string fields: ntime, nbits
                for (int i = 0; i < 2; i++) {
                    // Skip whitespace and commas
                    while (pos < params_section.length() && (params_section[pos] == ' ' || params_section[pos] == ',' || params_section[pos] == '\n')) pos++;
                    
                    if (params_section[pos] == '"') {
                        size_t start = pos + 1;
                        size_t end = params_section.find('"', start);
                        if (end != std::string::npos) {
                            string_fields.push_back(params_section.substr(start, end - start));
                            pos = end + 1;
                        } else break;
                    } else break;
                }
                
                // Assign fields if we have enough
                if (string_fields.size() >= 7) {
                    job_id = string_fields[0];     // job_id
                    version = string_fields[1];    // version  
                    prevhash = string_fields[2];   // prevhash
                    coinb1 = string_fields[3];     // coinb1
                    coinb2 = string_fields[4];     // coinb2
                    ntime = string_fields[5];      // ntime (after merkle_branch)
                    nbits = string_fields[6];      // nbits
                    
                    merkle_branch = merkle_array;  // Store merkle_branch
                    
                    // Parse clean_jobs boolean after nbits
                    bool clean_jobs = false;
                    {
                        // crude scan for true/false after nbits
                        size_t after_nbits = params_section.find(string_fields.back());
                        if (after_nbits != std::string::npos) {
                            size_t tf = params_section.find("true",  after_nbits);
                            size_t ff = params_section.find("false", after_nbits);
                            clean_jobs = (tf != std::string::npos) && (ff == std::string::npos || tf < ff);
                        }
                    }
                    need_fresh_job = clean_jobs;     // mark any queued solutions as stale
                    extranonce2_counter = 0;         // (optional) restart per-job counter
                    clean_jobs_flag = clean_jobs;
                    
                    log_status("Job ID: %s", job_id.c_str());
                    log_status("Version: %s", version.c_str());
                    log_status("Prevhash: %s...", prevhash.substr(0, 20).c_str());
                    log_status("Coinb1 length: %zu", coinb1.length());
                    log_status("Coinb2 length: %zu", coinb2.length());
                    log_status("Merkle branch entries: %zu", merkle_branch.size());
                    log_status("Ntime: %s", ntime.c_str());
                    log_status("Nbits: %s", nbits.c_str());
                    log_status("Clean jobs: %s", clean_jobs_flag ? "true" : "false");
                } else {
                    log_error("Failed to parse all required fields (got %zu/7)", string_fields.size());
                }
            }
            
            // Create one extranonce2 for this job and build the header with it
            current_extranonce2 = make_extranonce2(extranonce2_counter++, extranonce2_size);
            log_status("Generated extranonce2: %s", current_extranonce2.c_str());
            
            build_verus_header_from_job(current_extranonce2, header);
            return true;
        }
        
        return false;
    }
    
    void submit_share(uint32_t nonce) {
        if (!connected) return;

        // encode NONCE as the 4 bytes you actually put in the header (little-endian), then hex
        uint8_t nb[4] = { (uint8_t)nonce, (uint8_t)(nonce>>8), (uint8_t)(nonce>>16), (uint8_t)(nonce>>24) };
        std::string nonce_hex = to_hex(nb, 4);

        // Stratum v1 submit format: [user, job_id, extranonce2, ntime, nonce]
        std::string user = wallet_address;
        if (!worker_name.empty()) user += "." + worker_name;
        std::string msg = "{\"id\":" + std::to_string(message_id++) +
            ",\"method\":\"mining.submit\",\"params\":[\"" + user +
            "\",\"" + job_id + "\",\"" + current_extranonce2 + "\",\"" + ntime + "\",\"" + nonce_hex + "\"]}";
        
        send_message(msg);

        std::string resp = receive_line();
        if (resp.find("true") != std::string::npos) {
            log_success("Share ACCEPTED!");
            g_shares_accepted++;
        } else {
            log_error("Share REJECTED!");
            g_shares_rejected++;
        }
    }
    
    void get_share_target_le(uint8_t target[32]) {
        if (have_share_target) {
            // set_target from pool (already LE)
            memcpy(target, share_target_le, 32);
            return;
        }
        // Fall back to difficulty
        double d = (current_difficulty > 0.0) ? current_difficulty : 1.0;
        target_from_diff(d, target);
    }
    
    double get_current_difficulty() const {
        return current_difficulty;
    }
    
    bool need_new_job() const {
        return need_fresh_job;
    }
};


// ============================================================================

// Complete Mining Implementation
// ============================================================================

void run_bitsliced_mining(const MinerConfig& cfg) {

    BitslicedStratumClient stratum(cfg.pool_url, cfg.port, cfg.wallet, cfg.worker);


    // Device buffers and CUDA events
    uint8_t *d_header = nullptr, *d_found_hashes = nullptr, *d_target_le = nullptr;
    uint32_t *d_found_nonces = nullptr, *d_found_count = nullptr;
    cudaStream_t stream;
    cudaEvent_t start_evt, stop_evt, copy_done_evt;

    // Host-side buffers
    verus_header_t current_header, pending_header;
    bool have_pending = false;
    uint32_t h_found_nonces[8];
    uint8_t h_found_hashes[256];
    uint32_t h_found_count = 0;
    uint8_t target_le_host[32];
    uint8_t pending_target_le[32];


    if (!stratum.connect_to_pool()) {
        log_error("Failed to connect to pool");
        return;
    }

    CUDA_CHECK(cudaMalloc(&d_header, VERUS_HEADER_SIZE));
    CUDA_CHECK(cudaMalloc(&d_found_nonces, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_found_hashes, 8 * 32));
    CUDA_CHECK(cudaMalloc(&d_found_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_target_le, 32));

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));
    CUDA_CHECK(cudaEventCreate(&copy_done_evt));

    // Obtain initial work
    while (!stratum.get_work(&current_header)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    stratum.get_share_target_le(target_le_host);


    log_status("Current difficulty: {}", stratum.get_current_difficulty());
    log_status("Starting bitsliced mining loop...");

    const uint32_t BATCH_SIZE = cfg.batch_size;
    const int THREADS_PER_BLOCK = 128;
    const int WARPS_PER_BLOCK   = THREADS_PER_BLOCK / WARP_SIZE;
    const size_t PER_WARP_SMEM  = BITSLICE_WIDTH * VERUS_HEADER_SIZE + BITSLICE_WIDTH * 32;
    const size_t SHMEM          = VERUS_HEADER_SIZE + WARPS_PER_BLOCK * PER_WARP_SMEM;


    
    log_status("Current difficulty: %.2f", stratum.get_current_difficulty());


    std::cout << "Current difficulty: " << stratum.get_current_difficulty() << std::endl;


    log_status("Current difficulty: {}", stratum.get_current_difficulty());


    CUDA_CHECK(cudaMemcpy(d_target_le, target_le_host, 32, cudaMemcpyHostToDevice));

    log_status("Starting bitsliced mining loop...");

    const uint32_t BATCH_SIZE = cfg.batch_size;
    const int THREADS_PER_BLOCK = 128;                // 4 warps (keeps smem < 48KB)
    const int WARPS_PER_BLOCK   = THREADS_PER_BLOCK / WARP_SIZE;
    const size_t PER_WARP_SMEM  = BITSLICE_WIDTH * VERUS_HEADER_SIZE + BITSLICE_WIDTH * 32; // 9216 bytes
    const size_t SHMEM          = VERUS_HEADER_SIZE + WARPS_PER_BLOCK * PER_WARP_SMEM;      // 112 + 4*9216 = 36,976


    uint64_t nonce_base = 0;

    while (g_mining_active) {
        if (have_pending) {
            current_header = pending_header;
            memcpy(target_le_host, pending_target_le, 32);
            have_pending = false;
            nonce_base = 0;
        } else {
            verus_header_t tmp;
            if (stratum.get_work(&tmp)) {
                current_header = tmp;
                stratum.get_share_target_le(target_le_host);
                nonce_base = 0;
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(d_target_le, target_le_host, 32, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_header, current_header.header_data, VERUS_HEADER_SIZE, cudaMemcpyHostToDevice, stream));
        h_found_count = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_found_count, &h_found_count, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

        cudaEventRecord(start_evt, stream);
        bitsliced_mining_kernel<<<192, THREADS_PER_BLOCK, SHMEM, stream>>>(
            d_header, current_header.nonce_offset, nonce_base,
            BATCH_SIZE, d_target_le,
            d_found_nonces, d_found_hashes, d_found_count);
        cudaEventRecord(stop_evt, stream);

        CUDA_CHECK(cudaMemcpyAsync(&h_found_count, d_found_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        cudaEventRecord(copy_done_evt, stream);

        while (cudaEventQuery(copy_done_evt) == cudaErrorNotReady) {
            verus_header_t tmp;
            if (stratum.get_work(&tmp)) {
                pending_header = tmp;
                stratum.get_share_target_le(pending_target_le);
                have_pending = true;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        CUDA_CHECK(cudaEventSynchronize(copy_done_evt));

        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start_evt, stop_evt));
        double batch_seconds = kernel_ms / 1000.0;

        if (h_found_count > 0) {
            log_warn("FOUND {} potential shares!", h_found_count);


            log_warn("FOUND %u potential shares!", h_found_count);


            uint32_t to_copy = std::min(h_found_count, 8u);
            CUDA_CHECK(cudaMemcpy(h_found_nonces, d_found_nonces, to_copy * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_found_hashes, d_found_hashes, to_copy * 32, cudaMemcpyDeviceToHost));
            for (uint32_t i = 0; i < to_copy; i++) {
                uint32_t found_nonce = h_found_nonces[i];
                std::string hash_hex;
                for (int j = 0; j < 32; j++) {
                    char buf[3];
                    std::snprintf(buf, sizeof(buf), "%02x", h_found_hashes[i * 32 + j]);
                    hash_hex += buf;
                }

                log_status("Share found! Nonce: {:#x} Hash: {}", found_nonce, hash_hex);


                log_status("Share found! Nonce: %#x Hash: %s", found_nonce, hash_hex.c_str());

                if (!stratum.need_new_job()) {
                    log_status("Submitting share...");
                    stratum.submit_share(found_nonce);
                    g_shares_found++;
                } else {
                    log_warn("Skipping stale share (new job available)");
                }
            }
        }

        g_total_hashes += BATCH_SIZE;
        double current_hashrate = BATCH_SIZE / batch_seconds / 1000000.0;
        print_summary(current_hashrate);

        if (!have_pending) {
            nonce_base += BATCH_SIZE;
        }
    }

    // Cleanup
    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
    cudaEventDestroy(copy_done_evt);
    cudaStreamDestroy(stream);
    cudaFree(d_header);
    cudaFree(d_found_nonces);
    cudaFree(d_found_hashes);
    cudaFree(d_found_count);
    cudaFree(d_target_le);
}

// ============================================================================
// Main Mining Function
// ============================================================================

int main(int argc, char** argv) {
    log_status("================================================================");
    log_status("       RTX 5070 - Bitsliced VerusHash Miner");
    log_status("          64 Parallel Hashes Per Warp");
    log_status("      Boyar-Peralta Bitsliced S-box");
    log_status("================================================================");

    // Initialize CUDA
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);


    log_status("Mining Device: {}", prop.name);
    log_status("Architecture: sm_{}{}", prop.major, prop.minor);
    fmt::print("\n");


    
    log_status("Mining Device: %s", prop.name);
    log_status("Architecture: sm_%d%d", prop.major, prop.minor);
    std::cout << std::endl;
    

    MinerConfig cfg = parse_config(argc, argv);

    if (cfg.pool_url.empty()) {
        std::cerr << "Pool URL is required" << std::endl;
        return 1;
    }
    if (cfg.port <= 0 || cfg.port > 65535) {
        std::cerr << "Invalid port" << std::endl;
        return 1;
    }
    if (cfg.wallet.empty()) {
        std::cerr << "Wallet address is required" << std::endl;
        return 1;
    }

    const uint32_t BATCH_SIZE = cfg.batch_size;
    const int THREADS_PER_BLOCK = 256;
    const int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
    const int BLOCKS_PER_GRID = prop.multiProcessorCount * 4;

    log_status("Bitsliced Configuration:");

    log_status("- Hashes per warp: {}", BITSLICE_WIDTH);
    log_status("- Warps per block: {}", WARPS_PER_BLOCK);
    log_status("- Blocks: {}", BLOCKS_PER_GRID);
    log_status("- Total parallel hashes: {}", BLOCKS_PER_GRID * WARPS_PER_BLOCK * BITSLICE_WIDTH);
    log_status("- Batch size: {} nonces", BATCH_SIZE);
    fmt::print("\n");


    log_status("- Hashes per warp: %d", BITSLICE_WIDTH);
    log_status("- Warps per block: %d", WARPS_PER_BLOCK);
    log_status("- Blocks: %d", BLOCKS_PER_GRID);
    log_status("- Total parallel hashes: %d", BLOCKS_PER_GRID * WARPS_PER_BLOCK * BITSLICE_WIDTH);
    log_status("- Batch size: %u nonces", BATCH_SIZE);
    std::cout << std::endl;
    
    // Start bitsliced mining

    log_status("Starting bitsliced VerusHash mining...");

    try {
        run_bitsliced_mining(cfg);
    } catch (const std::exception& e) {
        log_error("Mining error: %s", e.what());
    }

    log_status("Mining stopped.");
    return 0;
}
