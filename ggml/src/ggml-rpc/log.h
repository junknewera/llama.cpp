#pragma once

// Log infrastructure for the RPC backend: all RPC logging goes through this header.
//
// LOG_ERROR / LOG_WARN / LOG_INFO are unconditional severity logs.
//
// LOG_DBG* are debug logs gated by the GGML_RPC_DEBUG verbosity variable:
//   unset / 0 - disabled (only warnings and errors are printed)
//   1         - high-level events: connections, handshake, buffer allocations, graph computes, cache evictions
//   2         - per-command trace: every RPC message sent/received, tensor transfers, cache hits/misses
//   3         - fine detail: per-command timings, graph node dumps, transport byte counts
// non-numeric values fall back to 1
//
// Byte counts are printed via format_size() below (KiB / MiB / GiB) to keep the per-command
// trace readable. Tensor identity (name, op, type, shape) is printed via format_tensor() in
// ggml-rpc.cpp.

#include "ggml-impl.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>

static inline int rpc_debug_level() {
    static const int level = [] {
        const char * env = std::getenv("GGML_RPC_DEBUG");
        if (env == nullptr) {
            return 0;
        }
        if (env[0] == '\0') {
            return 1;
        }
        int value = 0;
        for (const char * p = env; *p != '\0'; p++) {
            if (*p < '0' || *p > '9') {
                return 1;
            }
            value = value * 10 + (*p - '0');
            if (value > 3) {
                value = 3;
            }
        }
        return value;
    }();
    return level;
}

#define LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#define LOG_DBG(...)  do { if (rpc_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)
#define LOG_DBG2(...) do { if (rpc_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)
#define LOG_DBG3(...) do { if (rpc_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)

// Byte counts are logged in binary units to keep the per-message trace readable:
//   format_size(8208) -> "8.02 KiB", format_size(109314048) -> "104.25 MiB"
// Sizes below 1 KiB stay in plain bytes. Use with "%s", format_size(n).c_str()
static inline std::string format_size(size_t size) {
    static const char * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    double value = (double) size;
    size_t unit  = 0;
    while (value >= 1024.0 && unit + 1 < sizeof(units) / sizeof(units[0])) {
        value /= 1024.0;
        unit++;
    }

    char buf[32];
    if (unit == 0) {
        snprintf(buf, sizeof(buf), "%zu B", size);
    } else {
        snprintf(buf, sizeof(buf), "%.2f %s", value, units[unit]);
    }
    return std::string(buf);
}
