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

#include <fmt/core.h>
#include <fmt/color.h>

=======
#include <fstream>
#include <cctype>
=======

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

=======
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

enum class LogType { Status, Error, Success, Warn };

template <typename... Args>
void log_message(LogType type, fmt::format_string<Args...> fmt_str, Args&&... args) {
    if (g_summary_active.exchange(false)) fmt::print("\n");
    fmt::text_style style;
    switch (type) {
        case LogType::Status:
            style = fmt::fg(fmt::color::cyan);
            break;
        case LogType::Error:
            style = fmt::fg(fmt::color::red) | fmt::emphasis::bold;
            break;
        case LogType::Success:
            style = fmt::fg(fmt::color::green);
            break;
        case LogType::Warn:
            style = fmt::fg(fmt::color::yellow);
            break;
    }
    fmt::print(style, fmt_str, std::forward<Args>(args)...);
    fmt::print("\n");
}

template <typename... Args>
void log_status(fmt::format_string<Args...> fmt_str, Args&&... args) {
    log_message(LogType::Status, fmt_str, std::forward<Args>(args)...);
}

template <typename... Args>
void log_error(fmt::format_string<Args...> fmt_str, Args&&... args) {
    log_message(LogType::Error, fmt_str, std::forward<Args>(args)...);
}

template <typename... Args>
void log_success(fmt::format_string<Args...> fmt_str, Args&&... args) {
    log_message(LogType::Success, fmt_str, std::forward<Args>(args)...);
}

template <typename... Args>
void log_warn(fmt::format_string<Args...> fmt_str, Args&&... args) {
    log_message(LogType::Warn, fmt_str, std::forward<Args>(args)...);
}

void print_summary(double hashrate) {
    fmt::print("\rHashrate: {:.2f} MH/s | Shares F:{} A:{} R:{}",
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
            log_error("CUDA Error at {}:{} - {}", __FILE__, __LINE__, cudaGetErrorString(error)); \
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
=======
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


// ============================================================================

// Complete Mining Implementation
// ============================================================================

void run_bitsliced_mining() {
    BitslicedStratumClient stratum;


void run_bitsliced_mining(const MinerConfig& cfg) {
    
main
    if (!stratum.connect_to_pool()) {
        log_error("Failed to connect to pool");
        return;
    }

    // GPU memory allocation
    uint8_t *d_header, *d_found_hashes, *d_target_le;
    uint32_t *d_found_nonces, *d_found_count;

    CUDA_CHECK(cudaMalloc(&d_header, VERUS_HEADER_SIZE));
    CUDA_CHECK(cudaMalloc(&d_found_nonces, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_found_hashes, 8 * 32));
    CUDA_CHECK(cudaMalloc(&d_found_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_target_le, 32));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start_evt, stop_evt, copy_done_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));
    CUDA_CHECK(cudaEventCreate(&copy_done_evt));

    verus_header_t current_header, pending_header;
    bool have_pending = false;
    uint32_t h_found_nonces[8];
    uint8_t h_found_hashes[256];
    uint32_t h_found_count;
    uint8_t target_le_host[32];
    uint8_t pending_target_le[32];

    // Obtain initial work
    while (!stratum.get_work(&current_header)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    stratum.get_share_target_le(target_le_host);

    
    log_status("Current difficulty: {}", stratum.get_current_difficulty());
=======

    std::cout << "Current difficulty: " << stratum.get_current_difficulty() << std::endl;


    uint64_t nonce_base = 0;

    std::cout << "Starting bitsliced mining loop..." << std::endl;

    // Use smaller batch size for faster job switching
    const uint32_t BATCH_SIZE = 1048576; // 1M nonces instead of 4M for faster turnaround
    const int THREADS_PER_BLOCK = 128;                // 4 warps (keeps smem < 48KB)
    const int WARPS_PER_BLOCK   = THREADS_PER_BLOCK / WARP_SIZE;
    const size_t PER_WARP_SMEM  = BITSLICE_WIDTH * VERUS_HEADER_SIZE + BITSLICE_WIDTH * 32; // 9216 bytes
    const size_t SHMEM          = VERUS_HEADER_SIZE + WARPS_PER_BLOCK * PER_WARP_SMEM;      // 112 + 4*9216 = 36,976



    
    CUDA_CHECK(cudaMemcpy(d_target_le, target_le_host, 32, cudaMemcpyHostToDevice));

    uint64_t nonce_base = 0;

    
    log_status("Starting bitsliced mining loop...");
=======
    const uint32_t BATCH_SIZE = cfg.batch_size;

    std::cout << "Starting bitsliced mining loop..." << std::endl;

    
    main
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

        CUDA_CHECK(cudaMemcpy(d_found_count, &h_found_count, sizeof(uint32_t), cudaMemcpyHostToDevice));
        
        // Launch bitsliced mining kernel with per-warp shared memory partitions
        const int THREADS_PER_BLOCK = 128;                // 4 warps
        const int WARPS_PER_BLOCK   = THREADS_PER_BLOCK / WARP_SIZE;

        const size_t PER_WARP_SMEM  = BITSLICE_WIDTH * VERUS_HEADER_SIZE +
                                     BITSLICE_WIDTH * 32 +
                                     256 * sizeof(uint64_t) +
                                     512 * sizeof(uint64_t) +
                                     64 * 64;                            // 19,456 bytes
        const size_t SHMEM          = VERUS_HEADER_SIZE + WARPS_PER_BLOCK * PER_WARP_SMEM; // 112 + 4*19,456 = 77,936
        
        bitsliced_mining_kernel<<<192, THREADS_PER_BLOCK, SHMEM>>>(
            d_header, h_header.nonce_offset, nonce_base,
        main
            BATCH_SIZE, d_target_le,
            d_found_nonces, d_found_hashes, d_found_count
        );
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

            uint32_t to_copy = std::min(h_found_count, 8u);
            CUDA_CHECK(cudaMemcpy(h_found_nonces, d_found_nonces, to_copy * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_found_hashes, d_found_hashes, to_copy * 32, cudaMemcpyDeviceToHost));

            for (uint32_t i = 0; i < to_copy; i++) {
                uint32_t found_nonce = h_found_nonces[i];
                std::string hash_hex;

            std::cout << "FOUND " << h_found_count << " potential shares!" << std::endl;

            uint32_t to_copy = std::min(h_found_count, 8u);
            CUDA_CHECK(cudaMemcpyAsync(h_found_nonces, d_found_nonces, to_copy * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_found_hashes, d_found_hashes, to_copy * 32, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            for (uint32_t i = 0; i < to_copy; i++) {
                uint32_t found_nonce = h_found_nonces[i];

                std::cout << "Share found! Nonce: " << std::hex << found_nonce << std::dec;
                std::cout << " Hash: ";

                for (int j = 0; j < 32; j++) {
                    hash_hex += fmt::format("{:02x}", h_found_hashes[i * 32 + j]);
                }

                log_status("Share found! Nonce: {:#x} Hash: {}", found_nonce, hash_hex);

                // Check if we still have the same job before submitting

                std::cout << std::endl;


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
        
        nonce_base += BATCH_SIZE;

        std::cout << "Hashrate: " << std::fixed << std::setprecision(2)
                  << current_hashrate << " MH/s" << std::endl;

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

int main() {
    log_status("================================================================");
    log_status("       RTX 5070 - Bitsliced VerusHash Miner");
    log_status("          64 Parallel Hashes Per Warp");
    log_status("      Boyar-Peralta Bitsliced S-box");
    log_status("================================================================");
=======
int main(int argc, char** argv) {
    std::cout << "================================================================" << std::endl;
    std::cout << "       RTX 5070 - Bitsliced VerusHash Miner" << std::endl;
    std::cout << "          64 Parallel Hashes Per Warp" << std::endl;
    std::cout << "      Boyar-Peralta Bitsliced S-box" << std::endl;
    std::cout << "================================================================" << std::endl;

    
    // Initialize CUDA
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    log_status("Mining Device: {}", prop.name);
    log_status("Architecture: sm_{}{}", prop.major, prop.minor);
    fmt::print("\n");
    
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
    if (cfg.batch_size == 0 || cfg.batch_size % BITSLICE_WIDTH != 0) {
        std::cerr << "Batch size must be a multiple of " << BITSLICE_WIDTH << std::endl;
        return 1;
    }

    // Configuration for bitsliced mining
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
    
    // Start bitsliced mining
    log_status("Starting bitsliced VerusHash mining...");
    
    try {
        run_bitsliced_mining(cfg);
    } catch (const std::exception& e) {
        log_error("Mining error: {}", e.what());
    }

    log_status("Mining stopped.");
    return 0;
}