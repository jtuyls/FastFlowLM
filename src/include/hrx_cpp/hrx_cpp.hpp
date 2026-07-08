/// \file hrx_cpp.hpp
/// \brief Minimal C++ `namespace hrx` covering the runtime API subset that
///        FastFlowLM uses, implemented directly on top of libhrx (hrx_runtime.h).
///
/// This uses direct HRX calls. aiebu is dropped:
/// NPU control code goes straight from npu_sequence::dump() into an HRX XADX
/// "direct executable".
///
/// Coherence model: buffers are device-visible, host-coherent, mapped once
/// (persistent). FastFlowLM already brackets device work with explicit
/// sync_to_device()/sync_from_device() calls, so we map those directly to
/// hrx_buffer_flush_range()/hrx_buffer_invalidate_range(). Dispatch is
/// hrx_stream_dispatch() + hrx_stream_synchronize().
#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "hrx_runtime.h"
#include "hrx_cpp/hrx_xadx_builder.hpp"

// ---- ert_cmd_state: FLM's npu_utils returns/maps these (native HRX compatibility).
#ifndef FLM_ERT_CMD_STATE_DEFINED
#define FLM_ERT_CMD_STATE_DEFINED
enum ert_cmd_state {
    ERT_CMD_STATE_NEW = 1,
    ERT_CMD_STATE_QUEUED = 2,
    ERT_CMD_STATE_RUNNING = 3,
    ERT_CMD_STATE_COMPLETED = 4,
    ERT_CMD_STATE_ERROR = 5,
    ERT_CMD_STATE_ABORT = 6,
    ERT_CMD_STATE_SUBMITTED = 7,
    ERT_CMD_STATE_TIMEOUT = 8,
    ERT_CMD_STATE_NORESPONSE = 9,
    ERT_CMD_STATE_SKERROR = 10,
    ERT_CMD_STATE_SKCRASHED = 11,
    ERT_CMD_STATE_MAX = 12,
};
#endif

namespace hrx {

// ---------------------------------------------------------------------------
// Process-wide HRX runtime (one device + one stream), lazily initialized.
// ---------------------------------------------------------------------------
class Runtime {
public:
    static Runtime& get() {
        static Runtime r;
        return r;
    }
    hrx_device_t dev = nullptr;
    hrx_stream_t stream = nullptr;
    bool ok = false;

    void ensure() {
        if (dev) return;
        hrx_status_t init_status = hrx_gpu_initialize(0);
        bool init_ok = hrx_status_is_ok(init_status);
        if (!init_ok) {
            init_ok = hrx_status_code(init_status) == HRX_STATUS_ALREADY_EXISTS;
            hrx_status_ignore(init_status);
        }
        if (init_ok &&
            hrx_status_is_ok(hrx_gpu_device_get(0, &dev)) &&
            hrx_status_is_ok(hrx_stream_create(dev, 0, &stream))) {
            ok = true;
        } else {
            std::fprintf(stderr, "[hrx] device init FAILED\n");
            ok = false;
        }
    }

private:
    Runtime() = default;
};

inline Runtime& rt() {
    Runtime& r = Runtime::get();
    r.ensure();
    return r;
}

// Report (do not swallow) an HRX error. Returns true if status was an error.
// Surface dispatch/synchronize failures instead of letting failed command chains
// look like successful no-op dispatches.
inline bool hrx_report(hrx_status_t s, const char* where) {
    if (hrx_status_is_ok(s)) return false;
    char* m = nullptr;
    size_t mn = 0;
    hrx_status_to_string(s, &m, &mn);
    std::fprintf(stderr, "[hrx][ERROR] %s: %s\n", where, m ? m : "?");
    hrx_status_free_message(m);
    hrx_status_ignore(s);
    return true;
}

// ---------------------------------------------------------------------------
// Executable cache: build one HRX XADX executable per distinct
// (xclbin, control-code, patch-table) tuple and resolve its export ordinal once.
// ---------------------------------------------------------------------------
struct CachedExe {
    hrx_executable_t exe = nullptr;
    uint32_t ord = 0;
};

inline void hash_bytes(size_t& seed, const void* data, size_t size) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        seed ^= static_cast<size_t>(p[i]) + 0x9e3779b97f4a7c15ull +
                (seed << 6) + (seed >> 2);
    }
}

inline hrx_executable_t build_or_get_executable(
    const std::vector<uint8_t>& xclbin_bytes, const uint32_t* cc, size_t n,
    const uint32_t* patch, size_t patch_n, uint32_t* ord_out) {
    static std::mutex mu;
    static std::unordered_map<size_t, CachedExe> cache;
    size_t h = 0;
    hash_bytes(h, xclbin_bytes.data(), xclbin_bytes.size());
    hash_bytes(h, cc, n * sizeof(uint32_t));
    hash_bytes(h, patch, patch_n * sizeof(uint32_t));
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(h);
    if (it != cache.end()) {
        if (ord_out) *ord_out = it->second.ord;
        return it->second.exe;
    }
    flm_hrx::XadxEntryPoint ep;
    ep.name = "MLIR_AIE";
    ep.pdi_index = 0;
    ep.xclbin_index = 0;
    flm_hrx::XadxRun run;
    run.control_code.assign(cc, cc + n);  // raw TXN from npu_sequence::dump()
    if (patch && patch_n) run.patch_table.assign(patch, patch + patch_n);
    // A real FLM kernel always has buffer args that need host patching. An empty
    // patch table means dump_patch_table() failed to find the DDR-patch ops; the
    // amdxdna cmd-chain path then fails its patch_table_count==control_code_count
    // precondition and the whole chain becomes a silent no-op.
    if (!patch || patch_n == 0) {
        std::fprintf(stderr,
                     "[hrx][WARN] building executable with EMPTY patch table "
                     "(%zu control words) -- cmd-chain will fail\n", n);
    }
    ep.runs.push_back(std::move(run));

    hrx_executable_t exe = nullptr;
    uint32_t ord = 0;
    try {
        std::vector<uint8_t> xadx = flm_hrx::build_xadx(xclbin_bytes, {ep});
        Runtime& runtime = rt();
        if (!runtime.ok || !runtime.dev) {
            std::fprintf(stderr, "[hrx] executable load_data FAILED: HRX device unavailable\n");
            cache[h] = {nullptr, 0};
            if (ord_out) *ord_out = 0;
            return nullptr;
        }
        hrx_status_t s = hrx_executable_load_data(runtime.dev, xadx.data(),
                                                  xadx.size(),
                                                  "amdxdna-xclbin-fb", &exe);
        if (!hrx_status_is_ok(s)) {
            char* m = nullptr;
            size_t mn = 0;
            hrx_status_to_string(s, &m, &mn);
            std::fprintf(stderr, "[hrx] executable load_data FAILED: %s\n",
                         m ? m : "?");
            hrx_status_free_message(m);
            hrx_status_ignore(s);
            exe = nullptr;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[hrx] build_xadx threw: %s\n", e.what());
        exe = nullptr;
    }
    if (exe) hrx_executable_lookup_export_by_name(exe, "MLIR_AIE", &ord);
    cache[h] = {exe, ord};
    if (ord_out) *ord_out = ord;
    return exe;
}

class xclbin {
public:
    std::shared_ptr<std::vector<uint8_t>> bytes_ =
        std::make_shared<std::vector<uint8_t>>();

    xclbin() = default;
    explicit xclbin(const std::string& path) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("hrx::xclbin: cannot open " + path);
        std::fseek(f, 0, SEEK_END);
        long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (n > 0) {
            bytes_->resize(static_cast<size_t>(n));
            size_t rd = std::fread(bytes_->data(), 1, bytes_->size(), f);
            (void)rd;
        }
        std::fclose(f);
    }

    // FLM searches kernels for one whose name starts with "MLIR_AIE"; the HRX
    // dispatch path always uses the "MLIR_AIE" export, so a single placeholder
    // kernel is sufficient.
    class kernel {
    public:
        std::string name = "MLIR_AIE";
        std::string get_name() const { return name; }
    };
    std::vector<kernel> get_kernels() const { return {kernel{}}; }
    const std::vector<uint8_t>& bytes() const { return *bytes_; }
    std::shared_ptr<std::vector<uint8_t>> bytes_shared() const { return bytes_; }
};

class device {
public:
    device() = default;
    explicit device(unsigned int /*index*/) { rt(); }
    void register_xclbin(const xclbin& /*xc*/) {}
    void reset() {}
};

class hw_context {
public:
    std::shared_ptr<std::vector<uint8_t>> xclbin_bytes_;

    hw_context() = default;
    hw_context(const device& /*dev*/, const xclbin& xc)
        : xclbin_bytes_(xc.bytes_shared()) {}

    const std::vector<uint8_t>& xclbin_bytes() const {
        static const std::vector<uint8_t> empty;
        return xclbin_bytes_ ? *xclbin_bytes_ : empty;
    }
};

// ---------------------------------------------------------------------------
// Per-buffer host<->device coherence state.
//
// This path uses one persistent host-mapped, device-visible buffer per BO;
// coherence is kept with flush/invalidate range ops. The correctness comes
// from gating those ops per buffer:
//
//   host_dirty : the host wrote bytes that are not yet on the device. Must be
//                flushed (h2d) before any dispatch reads the buffer, and only
//                then.  Cleared by the flush.
//   dev_dirty  : a dispatch wrote bytes that are not yet visible to the host.
//                Must be invalidated (d2h) before the host reads, and only
//                then.  Cleared by the invalidate.
//
// Why gating matters:
//   * Unconditional flush-before-dispatch pushes a STALE host copy over a
//     device-resident buffer (e.g. kv_caches the previous layer just wrote) ->
//     corrupts the KV cache.
//   * Unconditional invalidate-on-read DROPS the host's own not-yet-flushed
//     writes -> the device never sees them on the next turn.
// The HRX path avoids both by only acting when the matching dirty bit is set.
// On a single turn the bits happen to line up either way, but turn>1 reuses
// buffers whose bits diverge.
struct BufCoh {
    bool host_dirty = true;   // freshly-allocated content must reach the device once
    bool dev_dirty = false;
};
inline std::mutex& buf_coh_mu() {
    static std::mutex m;
    return m;
}
inline std::unordered_map<hrx_buffer_t, BufCoh>& buf_coh() {
    static std::unordered_map<hrx_buffer_t, BufCoh> m;
    return m;
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------
class bo {
public:
    hrx_buffer_t hbuf_ = nullptr;
    void* mapped_ = nullptr;
    size_t size_ = 0;
    bool owns_ = false;

    bo() = default;
    virtual ~bo() {
        if (owns_ && hbuf_) {
            {
                std::lock_guard<std::mutex> lk(buf_coh_mu());
                buf_coh().erase(hbuf_);
            }
            hrx_buffer_release(hbuf_);
        }
        hbuf_ = nullptr;
        mapped_ = nullptr;
    }
    bo(const bo&) = delete;
    bo& operator=(const bo&) = delete;

    template <typename T>
    T map() {
        return reinterpret_cast<T>(mapped_);
    }
    size_t size() const { return size_; }
    hrx_buffer_t handle() const { return hbuf_; }

    // sync_to_device(): the host has new data. Make it visible to the device at
    // the explicit sync callsite. Dispatch-time dirty flushing remains only as a
    // safety net for wrapper-internal writes and freshly zeroed buffers.
    void flush() {
        if (!hbuf_) return;
        std::lock_guard<std::mutex> lk(buf_coh_mu());
        BufCoh& st = buf_coh()[hbuf_];
        st.host_dirty = true;
        st.dev_dirty = false;
        if (size_ != 0) {
            hrx_status_t s = hrx_buffer_flush_range(hbuf_, 0, size_);
            if (!hrx_report(s, "bo::flush hrx_buffer_flush_range"))
                st.host_dirty = false;
        } else {
            st.host_dirty = false;
        }
    }
    // sync_from_device(): the host wants to read. Only invalidate if a dispatch
    // actually produced new device data (dev_dirty); otherwise this would drop
    // the host's own writes.
    void invalidate() {
        if (!hbuf_) return;
        std::lock_guard<std::mutex> lk(buf_coh_mu());
        BufCoh& st = buf_coh()[hbuf_];
        if (st.dev_dirty) {
            hrx_status_t s = hrx_buffer_invalidate_range(hbuf_, 0, size_);
            if (!hrx_report(s, "bo::invalidate hrx_buffer_invalidate_range"))
                st.dev_dirty = false;
        }
    }
};

namespace ext {
class bo : public hrx::bo {
public:
    bo(const device& /*dev*/, size_t sz) {
        Runtime& r = rt();
        size_ = sz;
        owns_ = true;
        if (!r.ok) throw std::runtime_error("hrx::ext::bo: HRX device unavailable");
        // Device-visible, host-coherent, persistent mapping (one mmap kept for
        // the buffer's lifetime). Coherence maintained via flush/invalidate.
        hrx_status_t s = hrx_buffer_allocate(
            r.stream, sz,
            HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
            HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
            &hbuf_);
        if (!hrx_status_is_ok(s) || !hbuf_) {
            hrx_status_ignore(s);
            throw std::runtime_error("hrx::ext::bo: hrx_buffer_allocate failed");
        }
        void* p = nullptr;
        s = hrx_buffer_map_persistent(hbuf_, HRX_MAP_READ | HRX_MAP_WRITE, &p);
        if (!hrx_status_is_ok(s) || !p) {
            hrx_status_ignore(s);
            hrx_buffer_release(hbuf_);
            hbuf_ = nullptr;
            throw std::runtime_error("hrx::ext::bo: map_persistent failed");
        }
        mapped_ = p;
        std::memset(p, 0, sz);
        // The zeroed contents are host-side only until the first dispatch; mark
        // host_dirty so the gated h2d flushes them to the device exactly once.
        {
            std::lock_guard<std::mutex> lk(buf_coh_mu());
            buf_coh()[hbuf_] = BufCoh{/*host_dirty=*/true, /*dev_dirty=*/false};
        }
    }
};
}  // namespace ext

// ---------------------------------------------------------------------------
// run / runlist
// ---------------------------------------------------------------------------
// Gated host->device flush for a dispatch's bindings: flush only buffers the
// host still has dirtied. This is normally just a safety net; explicit
// sync_to_device() calls flush immediately.
inline bool hrx_h2d_bindings(const std::vector<hrx_buffer_ref_t>& binds) {
    std::lock_guard<std::mutex> lk(buf_coh_mu());
    auto& coh = buf_coh();
    bool ok = true;
    for (const auto& b : binds) {
        if (!b.buffer) continue;
        BufCoh& st = coh[b.buffer];
        if (st.host_dirty) {
            if (b.length != 0) {
                hrx_status_t s =
                    hrx_buffer_flush_range(b.buffer, b.offset, b.length);
                if (hrx_report(s, "hrx_h2d_bindings hrx_buffer_flush_range")) {
                    ok = false;
                    continue;
                }
            }
            st.host_dirty = false;
        }
    }
    return ok;
}

// After a dispatch completes, the device holds the freshest copy of every bound
// buffer: clear host_dirty and mark dev_dirty so the next host read invalidates
// (lazily, via sync_from_device). Mirrors the shim's post-dispatch readback
// bookkeeping. We over-approximate by treating every binding as a potential
// output; that is safe because an extra dev_dirty only triggers a redundant
// invalidate that re-reads identical bytes.
inline void hrx_mark_dispatched(const std::vector<hrx_buffer_ref_t>& binds) {
    std::lock_guard<std::mutex> lk(buf_coh_mu());
    auto& coh = buf_coh();
    for (const auto& b : binds) {
        if (!b.buffer) continue;
        BufCoh& st = coh[b.buffer];
        st.host_dirty = false;
        st.dev_dirty = true;
    }
}

class run {
public:
    hrx_executable_t exe_ = nullptr;
    uint32_t ord_ = 0;
    std::vector<hrx_buffer_ref_t> binds_;
    bool start_error_ = false;

    run() = default;
    explicit run(hrx_executable_t exe, uint32_t ord) : exe_(exe), ord_(ord) {}

    void add_binding(hrx_buffer_t b, size_t size) {
        binds_.push_back({b, 0, size});
    }

    // Record the dispatch on the stream (no synchronize); wait() flushes.
    void start() {
        start_error_ = false;
        if (!exe_) {
            std::fprintf(stderr, "[hrx][ERROR] run::start with null executable\n");
            start_error_ = true;
            return;
        }
        if (binds_.empty()) {
            std::fprintf(stderr, "[hrx][ERROR] run::start with no bindings\n");
            start_error_ = true;
            return;
        }
        if (!hrx_h2d_bindings(binds_)) {
            start_error_ = true;
            return;
        }
        hrx_dispatch_config_t cfg = {{1, 1, 1}, {1, 1, 1}, 0};
        hrx_status_t s = hrx_stream_dispatch(rt().stream, exe_, ord_, &cfg,
                                             nullptr, 0, binds_.data(),
                                             binds_.size(), HRX_DISPATCH_FLAG_NONE);
        start_error_ = hrx_report(s, "run::start hrx_stream_dispatch");
    }

    ert_cmd_state wait() {
        hrx_status_t s = hrx_stream_synchronize(rt().stream);
        bool err = hrx_report(s, "run::wait hrx_stream_synchronize");
        if (!start_error_) hrx_mark_dispatched(binds_);
        return (start_error_ || err) ? ERT_CMD_STATE_ERROR : ERT_CMD_STATE_COMPLETED;
    }
};

class runlist {
public:
    std::vector<run> runs_;

    runlist() = default;
    explicit runlist(const hw_context& /*ctx*/) {}

    void add(const run& r) { runs_.push_back(r); }
    void add(run&& r) { runs_.push_back(std::move(r)); }
    void reset() { runs_.clear(); }

    // Record every dispatch (no per-run synchronize) so HRX submits them as a
    // batch; wait() runs one synchronize for the whole list.
    void execute() {
        for (auto& r : runs_) r.start();
    }
    ert_cmd_state wait() {
        hrx_status_t s = hrx_stream_synchronize(rt().stream);
        bool err = hrx_report(s, "runlist::wait hrx_stream_synchronize");
        for (auto& r : runs_) {
            if (!r.start_error_) hrx_mark_dispatched(r.binds_);
            err = err || r.start_error_;
        }
        return err ? ERT_CMD_STATE_ERROR : ERT_CMD_STATE_COMPLETED;
    }
};

}  // namespace hrx
