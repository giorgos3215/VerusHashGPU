#ifndef STRATUM_CLIENT_H
#define STRATUM_CLIENT_H

#include <cstdint>
#include <string>
#include <vector>

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
#define SOCKET int
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#define closesocket close
#endif

#include "include/verushash.h"

class BitslicedStratumClient {
public:
    BitslicedStratumClient(const std::string& host, int port,
                           const std::string& wallet, const std::string& worker);
    ~BitslicedStratumClient();

    bool connect_to_pool();
    bool get_work(verus_header_t* header);
    void submit_share(uint32_t nonce);
    void get_share_target_le(uint8_t target[32]);
    double get_current_difficulty() const;
    bool need_new_job() const;

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

    static int hexval(unsigned char c);
    static std::string to_hex(const uint8_t* p, size_t n);
    std::string make_extranonce2(uint64_t ctr, int bytes);
    static void target_from_diff(double diff, uint8_t out[32]);

    static uint32_t rotr32(uint32_t x, int n);
    static uint32_t ch(uint32_t x, uint32_t y, uint32_t z);
    static uint32_t maj(uint32_t x, uint32_t y, uint32_t z);
    static uint32_t sigma0(uint32_t x);
    static uint32_t sigma1(uint32_t x);
    static uint32_t gamma0(uint32_t x);
    static uint32_t gamma1(uint32_t x);
    static void sha256_transform(uint32_t state[8], const uint8_t block[64]);
    static void sha256(const uint8_t* data, size_t len, uint8_t hash[32]);
    static void sha256d(const uint8_t* in, size_t len, uint8_t out32[32]);

    std::string hex_encode(const uint8_t* data, size_t len);
    void hex_decode(const std::string& hex, uint8_t* out, size_t max_len);
    bool send_message(const std::string& message);
    std::string receive_line();
    void build_verus_header_from_job(const std::string& extranonce2_hex, verus_header_t* header);
};

#endif // STRATUM_CLIENT_H
