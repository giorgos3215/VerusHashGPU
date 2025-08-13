#include "stratum_client.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <chrono>
#include <mutex>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <atomic>
#include <vector>
#include <cstdlib>
#include <cstdio>

// Pool configuration
static const char* POOL_HOST = "pool.verus.io";
static const int POOL_PORT = 9998;
static const char* WALLET_ADDRESS = "RB4M9dk5EDqywuWY7MVQ368wsEmGKPDuhg.RTX5070";

extern std::atomic<uint32_t> g_shares_accepted;
extern std::atomic<uint32_t> g_shares_rejected;

BitslicedStratumClient::BitslicedStratumClient() : sock(INVALID_SOCKET), connected(false), message_id(1),
                                                   extranonce2_size(4), difficulty_target(0x00000400), share_counter(0),
                                                   need_fresh_job(false), have_target(false), have_hash_reserved(false),
                                                   have_share_target(false), have_block_target(false), current_difficulty(1.0),
                                                   extranonce2_counter(0), clean_jobs_flag(false) {}

BitslicedStratumClient::~BitslicedStratumClient() {
    if (connected && sock != INVALID_SOCKET) {
        closesocket(sock);
#ifdef _WIN32
        WSACleanup();
#endif
    }
}

int BitslicedStratumClient::hexval(unsigned char c){
    if (c>='0' && c<='9') return c-'0';
    if (c>='a' && c<='f') return c-'a'+10;
    if (c>='A' && c<='F') return c-'A'+10;
    return -1;
}

std::string BitslicedStratumClient::to_hex(const uint8_t* p, size_t n) {
    static const char* hexd="0123456789abcdef";
    std::string s; s.resize(n*2);
    for (size_t i=0;i<n;i++){ s[2*i]=hexd[p[i]>>4]; s[2*i+1]=hexd[p[i]&0xF]; }
    return s;
}

std::string BitslicedStratumClient::make_extranonce2(uint64_t ctr, int bytes) {
    std::string s; s.resize(bytes*2);
    for (int i=0;i<bytes;i++){
        uint8_t b = (ctr >> (8*i)) & 0xff; // little-endian byte order
        static const char* hexd="0123456789abcdef";
        s[2*i]   = hexd[b>>4];
        s[2*i+1] = hexd[b&0xF];
    }
    return s;
}

void BitslicedStratumClient::target_from_diff(double diff, uint8_t out[32]) {
    static const uint8_t BASE[32] = {
        0xff,0xff,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
    };
    if (!(diff > 0.0)) diff = 1.0;

    long double D = (long double)diff;
    long double R = 0.0L;

    for (int i = 31; i >= 0; --i) {
        long double value = R * 256.0L + (long double)BASE[i];
        long double q_ld = floorl(value / D);
        if (q_ld > 255.0L) q_ld = 255.0L;
        unsigned int q = (unsigned int)q_ld;
        out[i] = (uint8_t)q;
        R = value - q_ld * D;
    }
}

uint32_t BitslicedStratumClient::rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
uint32_t BitslicedStratumClient::ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
uint32_t BitslicedStratumClient::maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
uint32_t BitslicedStratumClient::sigma0(uint32_t x) { return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22); }
uint32_t BitslicedStratumClient::sigma1(uint32_t x) { return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25); }
uint32_t BitslicedStratumClient::gamma0(uint32_t x) { return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3); }
uint32_t BitslicedStratumClient::gamma1(uint32_t x) { return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10); }

void BitslicedStratumClient::sha256_transform(uint32_t state[8], const uint8_t block[64]) {
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

    for (int i = 0; i < 16; ++i) {
        w[i] = (block[i * 4] << 24) | (block[i * 4 + 1] << 16) |
               (block[i * 4 + 2] << 8) | block[i * 4 + 3];
    }
    for (int i = 16; i < 64; ++i) {
        w[i] = gamma1(w[i - 2]) + w[i - 7] + gamma0(w[i - 15]) + w[i - 16];
    }

    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    for (int i = 0; i < 64; ++i) {
        t1 = h + sigma1(e) + ch(e, f, g) + K[i] + w[i];
        t2 = sigma0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

void BitslicedStratumClient::sha256(const uint8_t* data, size_t len, uint8_t hash[32]) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    uint8_t block[64];
    size_t i;
    for (i = 0; i + 64 <= len; i += 64) {
        memcpy(block, data + i, 64);
        sha256_transform(state, block);
    }

    size_t rem = len - i;
    memcpy(block, data + i, rem);
    block[rem++] = 0x80;
    if (rem > 56) {
        memset(block + rem, 0, 64 - rem);
        sha256_transform(state, block);
        rem = 0;
    }
    memset(block + rem, 0, 56 - rem);
    uint64_t bit_len = len * 8;
    for (int j = 7; j >= 0; j--) {
        block[56 + j] = bit_len & 0xff;
        bit_len >>= 8;
    }
    sha256_transform(state, block);

    for (int j = 0; j < 8; j++) {
        hash[j*4] = (state[j] >> 24) & 0xff;
        hash[j*4+1] = (state[j] >> 16) & 0xff;
        hash[j*4+2] = (state[j] >> 8) & 0xff;
        hash[j*4+3] = state[j] & 0xff;
    }
}

void BitslicedStratumClient::sha256d(const uint8_t* in, size_t len, uint8_t out32[32]) {
    uint8_t temp[32];
    sha256(in, len, temp);
    sha256(temp, 32, out32);
}

std::string BitslicedStratumClient::hex_encode(const uint8_t* data, size_t len) {
    std::string result;
    char hex[3];
    for (size_t i = 0; i < len; i++) {
        snprintf(hex, sizeof(hex), "%02x", data[i]);
        result += hex;
    }
    return result;
}

void BitslicedStratumClient::hex_decode(const std::string& hex, uint8_t* out, size_t max_len) {
    size_t n = std::min(hex.size()/2, max_len);
    for (size_t i=0;i<n;i++){
        int hi = hexval((unsigned char)hex[2*i]);
        int lo = hexval((unsigned char)hex[2*i+1]);
        if (hi >= 0 && lo >= 0) {
            out[i] = (uint8_t)((hi<<4) | lo);
        }
    }
}

bool BitslicedStratumClient::send_message(const std::string& message) {
    std::string msg = message + "\n";
    int result = send(sock, msg.c_str(), (int)msg.length(), 0);
    return result != SOCKET_ERROR;
}

std::string BitslicedStratumClient::receive_line() {
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

void BitslicedStratumClient::build_verus_header_from_job(const std::string& extranonce2_hex, verus_header_t* header) {
    memset(header->header_data, 0, VERUS_HEADER_SIZE);

    std::string coinbase_hex = coinb1 + extranonce1 + extranonce2_hex + coinb2;

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

    uint8_t tmp4[4], t32[32];
    hex_decode(version, tmp4, 4);
    for (int i=0;i<4;i++) header->header_data[i] = tmp4[3-i];

    hex_decode(prevhash, t32, 32);
    for (int i=0;i<32;i++) header->header_data[4+i] = t32[31-i];

    for (int i=0;i<32;i++) header->header_data[36+i] = root[31-i];

    if (hash_reserved_hex.size() >= 64) {
        hex_decode(hash_reserved_hex, t32, 32);
        for (int i=0;i<32;i++) header->header_data[68+i] = t32[31-i];
    }

    hex_decode(ntime, tmp4, 4);
    for (int i=0;i<4;i++) header->header_data[100+i] = tmp4[3-i];

    hex_decode(nbits, tmp4, 4);
    for (int i=0;i<4;i++) header->header_data[104+i] = tmp4[3-i];

    header->nonce_offset = 108;
}

bool BitslicedStratumClient::connect_to_pool() {
    std::cout << "Connecting to VerusPool..." << std::endl;

#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cout << "ERROR: WSAStartup failed" << std::endl;
        return false;
    }
#endif

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET) {
        std::cout << "ERROR: Socket creation failed" << std::endl;
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
    server_addr.sin_port = htons(POOL_PORT);

    struct hostent* host_entry = gethostbyname(POOL_HOST);
    if (host_entry == nullptr) {
        std::cout << "ERROR: DNS resolution failed" << std::endl;
        closesocket(sock);
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }

    memcpy(&server_addr.sin_addr, host_entry->h_addr_list[0], host_entry->h_length);

    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cout << "ERROR: Connection failed" << std::endl;
        closesocket(sock);
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }

    connected = true;
    std::cout << "CONNECTED to " << POOL_HOST << ":" << POOL_PORT << std::endl;

    std::string subscribe_msg = "{\"id\": " + std::to_string(message_id++) +
                               ", \"method\": \"mining.subscribe\", \"params\": [\"BitslicedMiner/1.0\"]}";

    if (!send_message(subscribe_msg)) {
        std::cout << "ERROR: Failed to send subscribe" << std::endl;
        return false;
    }

    std::string response = receive_line();
    std::cout << "Subscribe response: " << response << std::endl;

    if (response.find("\"result\"") != std::string::npos) {
        size_t rpos = response.find("\"result\"");
        size_t lb = response.find('[', rpos);
        size_t rb = response.rfind(']');
        if (lb != std::string::npos && rb != std::string::npos && rb > lb) {
            std::string arr = response.substr(lb, rb - lb + 1);

            size_t q1 = arr.find('"'), q2 = arr.find('"', q1 + 1);
            q1 = arr.find('"', q2 + 1);
            q2 = arr.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos) {
                extranonce1 = arr.substr(q1 + 1, q2 - q1 - 1);
                std::cout << "Extracted extranonce1: " << extranonce1 << std::endl;
            }

            size_t lastComma = arr.find_last_of(",]");
            if (lastComma != std::string::npos) {
                extranonce2_size = atoi(arr.c_str() + lastComma + 1);
                if (extranonce2_size <= 0 || extranonce2_size > 16) extranonce2_size = 4;
                std::cout << "Extracted extranonce2_size: " << extranonce2_size << std::endl;
            }
        }
    }

    std::string auth_msg = "{\"id\": " + std::to_string(message_id++) +
                          ", \"method\": \"mining.authorize\", \"params\": [\"" + WALLET_ADDRESS + "\", \"\"]}";

    if (!send_message(auth_msg)) {
        std::cout << "ERROR: Failed to send authorize" << std::endl;
        return false;
    }

    response = receive_line();
    std::cout << "Auth response: " << response << std::endl;

    if (response.find("mining.set_target") != std::string::npos) {
        size_t bracket_start = response.find("[\"");
        size_t bracket_end = response.find("\"]");
        if (bracket_start != std::string::npos && bracket_end != std::string::npos && bracket_end > bracket_start) {
            std::string targ_hex = response.substr(bracket_start + 2, bracket_end - bracket_start - 2);
            std::cout << "Found target hex in auth: " << targ_hex << std::endl;
            if (targ_hex.size() >= 64) {
                for (int i=0;i<32;i++) {
                    int hi = hexval((unsigned char)targ_hex[2*i]);
                    int lo = hexval((unsigned char)targ_hex[2*i+1]);
                    if (hi>=0 && lo>=0) share_target_le[i] = (uint8_t)((hi<<4)|lo);
                }
                have_share_target = true;
            }
        }
    }

    return true;
}

bool BitslicedStratumClient::get_work(verus_header_t* header) {
    std::string line = receive_line();
    if (line.empty()) return false;

    if (line.find("mining.notify") != std::string::npos) {
        size_t p = line.find("\"params\":");
        size_t lbr = line.find('[', p);
        size_t rbr = line.rfind(']');
        if (p!=std::string::npos && lbr!=std::string::npos && rbr!=std::string::npos) {
            std::string params_section = line.substr(lbr+1, rbr-lbr-1);
            std::vector<std::string> string_fields;
            std::vector<std::string> merkle_array;
            size_t pos = 0;
            while (pos < params_section.length()) {
                while (pos < params_section.length() &&
                       (params_section[pos] == ' ' || params_section[pos] == ',' || params_section[pos] == '\n')) pos++;
                if (pos >= params_section.length()) break;

                if (params_section[pos] == '"') {
                    size_t start = pos + 1;
                    size_t end = params_section.find('"', start);
                    if (end != std::string::npos) {
                        string_fields.push_back(params_section.substr(start, end - start));
                        pos = end + 1;
                    } else break;
                } else if (params_section[pos] == '[') {
                    size_t start = pos + 1;
                    size_t end = params_section.find(']', start);
                    if (end != std::string::npos) {
                        std::string arr = params_section.substr(start, end - start);
                        size_t spos = 0;
                        while (spos < arr.size()) {
                            while (spos < arr.size() && (arr[spos] == ' ' || arr[spos] == ',' || arr[spos]=='\n')) spos++;
                            if (spos>=arr.size()) break;
                            if (arr[spos]=='"') {
                                size_t st = spos+1;
                                size_t en = arr.find('"', st);
                                if (en!=std::string::npos) {
                                    merkle_array.push_back(arr.substr(st, en-st));
                                    spos = en+1;
                                } else break;
                            } else break;
                        }
                        pos = end + 1;
                    } else break;
                } else {
                    pos++;
                }
            }

            for (int i = 0; i < 2; i++) {
                while (pos < params_section.length() && (params_section[pos] == ' ' || params_section[pos] == ',' ||
                       params_section[pos] == '\n')) pos++;
                if (params_section[pos] == '"') {
                    size_t start = pos + 1;
                    size_t end = params_section.find('"', start);
                    if (end != std::string::npos) {
                        string_fields.push_back(params_section.substr(start, end - start));
                        pos = end + 1;
                    } else break;
                } else break;
            }

            if (string_fields.size() >= 7) {
                job_id = string_fields[0];
                version = string_fields[1];
                prevhash = string_fields[2];
                coinb1 = string_fields[3];
                coinb2 = string_fields[4];
                ntime = string_fields[5];
                nbits = string_fields[6];

                merkle_branch = merkle_array;

                bool clean_jobs = false;
                {
                    size_t after_nbits = params_section.find(string_fields.back());
                    if (after_nbits != std::string::npos) {
                        size_t tf = params_section.find("true",  after_nbits);
                        size_t ff = params_section.find("false", after_nbits);
                        clean_jobs = (tf != std::string::npos) && (ff == std::string::npos || tf < ff);
                    }
                }
                need_fresh_job = clean_jobs;
                extranonce2_counter = 0;
                clean_jobs_flag = clean_jobs;

                std::cout << "Job ID: " << job_id << std::endl;
                std::cout << "Version: " << version << std::endl;
                std::cout << "Prevhash: " << prevhash.substr(0, 20) << "..." << std::endl;
                std::cout << "Coinb1 length: " << coinb1.length() << std::endl;
                std::cout << "Coinb2 length: " << coinb2.length() << std::endl;
                std::cout << "Merkle branch entries: " << merkle_branch.size() << std::endl;
                std::cout << "Ntime: " << ntime << std::endl;
                std::cout << "Nbits: " << nbits << std::endl;
                std::cout << "Clean jobs: " << (clean_jobs_flag ? "true" : "false") << std::endl;
            } else {
                std::cout << "Failed to parse all required fields (got " << string_fields.size() << "/7)" << std::endl;
            }
        }

        current_extranonce2 = make_extranonce2(extranonce2_counter++, extranonce2_size);
        std::cout << "Generated extranonce2: " << current_extranonce2 << std::endl;

        build_verus_header_from_job(current_extranonce2, header);
        return true;
    }

    if (line.find("set_difficulty") != std::string::npos) {
        size_t lp = line.find("[");
        size_t rp = line.find("]");
        if (lp!=std::string::npos && rp!=std::string::npos && rp>lp) {
            double diff = atof(line.substr(lp+1, rp-lp-1).c_str());
            current_difficulty = diff;
            std::cout << "New difficulty: " << diff << std::endl;
        }
    }

    if (line.find("set_target") != std::string::npos) {
        size_t bracket_start = line.find("[\"");
        size_t bracket_end = line.find("\"]");
        if (bracket_start != std::string::npos && bracket_end != std::string::npos && bracket_end > bracket_start) {
            std::string targ_hex = line.substr(bracket_start + 2, bracket_end - bracket_start - 2);
            if (targ_hex.size() >= 64) {
                for (int i=0;i<32;i++) {
                    int hi = hexval((unsigned char)targ_hex[2*i]);
                    int lo = hexval((unsigned char)targ_hex[2*i+1]);
                    if (hi>=0 && lo>=0) share_target_le[i] = (uint8_t)((hi<<4)|lo);
                }
                have_share_target = true;
            }
        }
    }

    return false;
}

void BitslicedStratumClient::submit_share(uint32_t nonce) {
    if (!connected) return;

    uint8_t nb[4] = { (uint8_t)nonce, (uint8_t)(nonce>>8), (uint8_t)(nonce>>16), (uint8_t)(nonce>>24) };
    std::string nonce_hex = to_hex(nb, 4);

    std::string msg = "{\"id\":" + std::to_string(message_id++) +
        ",\"method\":\"mining.submit\",\"params\":[\"" + std::string(WALLET_ADDRESS) +
        "\",\"" + job_id + "\",\"" + current_extranonce2 + "\",\"" + ntime + "\",\"" + nonce_hex + "\"]}";

    send_message(msg);

    std::string resp = receive_line();
    if (resp.find("true") != std::string::npos) {
        std::cout << "Share ACCEPTED!" << std::endl;
        g_shares_accepted++;
    } else {
        std::cout << "Share REJECTED!" << std::endl;
        g_shares_rejected++;
    }
}

void BitslicedStratumClient::get_share_target_le(uint8_t target[32]) {
    if (have_share_target) {
        memcpy(target, share_target_le, 32);
        return;
    }
    double d = (current_difficulty > 0.0) ? current_difficulty : 1.0;
    target_from_diff(d, target);
}

double BitslicedStratumClient::get_current_difficulty() const {
    return current_difficulty;
}

bool BitslicedStratumClient::need_new_job() const {
    return need_fresh_job;
}

