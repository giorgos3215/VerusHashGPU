#ifndef VERUSHASH_H
#define VERUSHASH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// VerusHash v2.1 constants
#define VERUSHASH_BLOCK_SIZE 64    // VerusHash uses 64-byte internal blocks
#define VERUSHASH_OUTPUT_SIZE 32   // Final output is 32 bytes
#define VERUSHASH_KEY_SIZE 32      // Key size for keyed Haraka
#define HARAKA512_ROUNDS 5
#define HARAKA256_ROUNDS 5

// VerusHash v2.1 block header constants
#define VERUS_HEADER_SIZE 112      // Verus block headers are 112 bytes
#define VERUS_NONCE_SIZE 32        // Nonce field is 32 bytes

// GPU optimization parameters
#define CUDA_THREADS_PER_BLOCK 256
#define CUDA_BLOCKS_PER_GRID 128
#define HASHES_PER_THREAD 4
#define WARP_SIZE 32

// Core VerusHash functions
void verushash(void* result, const void* data, size_t len);
void verushash_half(void* result, const void* data, size_t len);

// VerusHash v2.1 functions with proper algorithm
void verushash_v2(void* result, const void* data, size_t len);
void verushash_v2_keyed(void* result, const void* data, size_t len, const void* key);

// VerusHash v2.1 context for incremental hashing
typedef struct {
    uint8_t buffer[64];      // Internal 64-byte buffer
    uint8_t key[32];         // Current key for keyed operations  
    size_t buf_len;          // Current buffer length
    uint32_t total_len;      // Total bytes processed
    uint32_t key_schedule;   // Key generation state
} verushash_v2_ctx;

// GPU mining interface
typedef struct {
    uint64_t hashrate;
    uint32_t valid_shares;
    uint32_t invalid_shares;
    double power_consumption;
    double temperature;
} mining_stats_t;

typedef struct {
    char pool_url[256];
    char wallet_address[128];
    char worker_name[64];
    uint16_t pool_port;
    uint32_t difficulty;
} pool_config_t;

// GPU functions
int gpu_init(void);
void gpu_cleanup(void);
int gpu_mine_batch(const void* header, size_t header_len, uint64_t nonce_start, 
                   uint32_t batch_size, void* results, uint32_t* found_count);

// VerusHash v2.1 mining functions
int gpu_mine_verus_batch(const void* verus_header, uint64_t nonce_start,
                        uint32_t batch_size, uint32_t difficulty_target,
                        void* found_nonces, uint32_t* found_count);

// Verus block header parsing
typedef struct {
    uint8_t header_data[VERUS_HEADER_SIZE];  // Full 1487-byte header
    uint8_t nonce_field[VERUS_NONCE_SIZE];   // 32-byte nonce field  
    uint32_t nonce_offset;                   // Offset of nonce in header
    uint32_t timestamp_offset;               // Offset of timestamp in header
    uint32_t difficulty_offset;              // Offset of difficulty target
} verus_header_t;

// Performance monitoring
void perf_monitor_init(void);
void perf_monitor_update(const mining_stats_t* stats);
void perf_monitor_print(void);

// Pool connection
int pool_connect(const pool_config_t* config);
void pool_disconnect(void);
int pool_submit_share(const void* share_data);
int pool_get_work(void* work_data, size_t* work_size);

// VerusHash v2.1 pool functions
int pool_get_verus_work(verus_header_t* header, uint32_t* difficulty);
int pool_submit_verus_share(const verus_header_t* header, const uint8_t* nonce, 
                           const uint8_t* hash_result);

// Verus header utility functions
void verus_header_init(verus_header_t* header);
void verus_header_set_nonce(verus_header_t* header, const uint8_t* nonce);
void verus_header_get_nonce(const verus_header_t* header, uint8_t* nonce);
int verus_header_validate(const verus_header_t* header);

#ifdef __cplusplus
}
#endif

#endif // VERUSHASH_H