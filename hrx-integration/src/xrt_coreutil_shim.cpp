// libxrt_coreutil.so.2 capture shim for FastFlowLM (B1).
//
// FLM's closed-source NPU engines (libqwen3_npu.so, ...) and the flm binary
// import ~33 xrt:: symbols from libxrt_coreutil.so.2. This shim provides those
// symbols with an ABI-compatible layout (every xrt pimpl object is a single
// std::shared_ptr; xrt::uuid is 16 bytes) but, instead of touching hardware,
// RECORDS the precompile artifacts we need to replay FLM's kernels through HRX:
//
//   * the xclbin bound to each kernel,
//   * the control-code ELF (the aiebu instruction-transaction blob) the engine
//     hands to xrt::elf(buf,size),
//   * the ordered buffer bindings (sizes) of each dispatch.
//
// Dispatch is stubbed (no compute), so a real `flm run` produces garbage output
// but emits a complete capture of its kernel set. Artifacts go to
// $FLM_CAPTURE_DIR (default /tmp/flm_capture).
//
// We deliberately DO NOT include the real XRT headers: the engines were built
// against an older xrt::elf (elf(const void*, size_t)) than the installed 2.23
// headers (elf(string_view)). We self-declare minimal types so the mangled
// symbol names match exactly.

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// HRX forwarding (enabled at runtime via FLM_FORWARD=1).
#include "hrx_runtime.h"
#include "hrx_xadx_builder.hpp"

// ---- C enums referenced in mangled signatures (names must match) ----
enum xclBOSyncDirection {
    XCL_BO_SYNC_BO_TO_DEVICE = 0,
    XCL_BO_SYNC_BO_FROM_DEVICE = 1,
};
enum ert_cmd_state {
    ERT_CMD_STATE_NEW = 1,
    ERT_CMD_STATE_COMPLETED = 4,
};

// ================= capture registry =================
namespace cap {

struct XclbinImpl {
    std::string path;
    std::vector<uint8_t> bytes;
    size_t content_hash = 0;
};
struct DeviceImpl {};
struct HwCtxImpl {
    std::shared_ptr<XclbinImpl> xclbin;
    hrx_stream_t stream = nullptr;
    ~HwCtxImpl() {
        if (stream)
            hrx_stream_release(stream);
    }
};
struct ElfImpl {
    std::vector<uint8_t> bytes;
    std::vector<uint32_t> control_code;
    std::vector<uint32_t> patch_table;
    size_t control_hash = 0;
    bool parse_ok = false;
};
struct ModuleImpl {
    std::shared_ptr<ElfImpl> elf;
};
struct KernelImpl {
    std::shared_ptr<ModuleImpl> module;
    std::shared_ptr<HwCtxImpl> hwctx;
    std::shared_ptr<XclbinImpl> xclbin;
    std::string name;
    hrx_executable_t executable = nullptr;
    uint32_t executable_ordinal = 0;
    bool executable_resolved = false;
};
struct BoImpl {
    std::vector<uint8_t> data;  // host staging the engine maps + read/writes
    size_t size = 0;
    uint64_t id = 0;
    hrx_buffer_t hbuf = nullptr;  // forward-mode device buffer
    // Direct-map mode: persistent host-coherent mapping of hbuf that the engine
    // reads/writes in place (no host staging copy). When non-null, `data` is
    // unused and coherence is kept via hrx_buffer_flush_range/invalidate_range.
    // Null => legacy staging path (the `data` vector is the engine's buffer).
    void* mapped = nullptr;
    // forward-mode dirty tracking:
    //  host_dirty: host data not yet on device (needs h2d before dispatch).
    //  dev_dirty:  device data not yet on host (needs d2h before host read).
    bool host_dirty = true;
    bool dev_dirty = false;
};
struct Binding {
    int index = 0;
    bool is_bo = false;
    size_t size = 0;   // bo size, or scalar byte size
    uint64_t value = 0;  // scalar value (small args)
    hrx_buffer_t hbuf = nullptr;  // forward mode: bo's HRX device buffer
    void* host = nullptr;         // forward mode: bo's host staging (b->data)
    BoImpl* bi = nullptr;         // forward mode: for dirty tracking
};
struct PendingSubmission {
    bool active = false;
    hrx_timeline_point_t completion = {};
    std::vector<BoImpl*> outputs;
    uint64_t graph_node = 0;
    bool is_chain = false;
};
struct RunImpl {
    std::shared_ptr<KernelImpl> kernel;
    std::vector<Binding> args;
    uint64_t graph_node = 0;
    PendingSubmission pending;
};
struct RunlistImpl {
    std::vector<std::shared_ptr<RunImpl>> runs;
    PendingSubmission pending;
    std::shared_ptr<HwCtxImpl> hwctx;
};

class Registry {
public:
    static Registry& get() {
        static Registry r;
        return r;
    }

    std::string dir;
    std::ofstream manifest;
    std::mutex mu;
    std::atomic<uint64_t> bo_counter{1};
    std::atomic<uint64_t> uuid_counter{1};
    std::map<std::array<uint8_t, 16>, std::shared_ptr<XclbinImpl>> uuid_to_xclbin;
    std::weak_ptr<XclbinImpl> last_xclbin;  // most-recently-constructed xclbin
    std::map<size_t, int> seen_control;  // hash -> id
    int control_id = 0;
    int dispatch_seq = 0;

    Registry() {
        const char* d = std::getenv("FLM_CAPTURE_DIR");
        dir = d ? d : "/tmp/flm_capture";
        std::string mkdir = "mkdir -p '" + dir + "'";
        (void)system(mkdir.c_str());
        manifest.open(dir + "/manifest.jsonl", std::ios::app);
        std::fprintf(stderr, "[xrt-capture] artifacts -> %s\n", dir.c_str());
    }

    std::array<uint8_t, 16> fresh_uuid(std::shared_ptr<XclbinImpl> xc) {
        std::array<uint8_t, 16> u{};
        uint64_t n = uuid_counter++;
        std::memcpy(u.data(), &n, sizeof(n));
        std::lock_guard<std::mutex> lk(mu);
        uuid_to_xclbin[u] = xc;
        return u;
    }

    void capture(RunImpl* run) {
        if (!run || !run->kernel || !run->kernel->module ||
            !run->kernel->module->elf)
            return;
        auto& elf = run->kernel->module->elf->bytes;
        size_t h = std::hash<std::string_view>{}(
            std::string_view(reinterpret_cast<const char*>(elf.data()),
                             elf.size()));
        std::lock_guard<std::mutex> lk(mu);
        int id;
        auto it = seen_control.find(h);
        bool is_new = (it == seen_control.end());
        if (is_new) {
            id = control_id++;
            seen_control[h] = id;
            char fn[64];
            std::snprintf(fn, sizeof(fn), "/control_%04d.elf", id);
            std::ofstream f(dir + fn, std::ios::binary);
            f.write(reinterpret_cast<const char*>(elf.data()), elf.size());
        } else {
            id = it->second;
        }
        // Always log the dispatch (control id may repeat with different bindings).
        manifest << "{\"seq\":" << dispatch_seq++ << ",\"control_id\":" << id
                 << ",\"new\":" << (is_new ? "true" : "false")
                 << ",\"kernel\":\"" << run->kernel->name << "\""
                 << ",\"xclbin\":\""
                 << (run->kernel->xclbin ? run->kernel->xclbin->path : "")
                 << "\",\"elf_bytes\":" << elf.size() << ",\"bindings\":[";
        for (size_t i = 0; i < run->args.size(); i++) {
            const auto& b = run->args[i];
            if (i) manifest << ",";
            manifest << "{\"i\":" << b.index << ",\"bo\":"
                     << (b.is_bo ? "true" : "false") << ",\"size\":" << b.size
                     << "}";
        }
        manifest << "]}\n";
        manifest.flush();
    }
};

inline void* alloc_impl(std::shared_ptr<void>& h, std::shared_ptr<void> impl) {
    h = std::move(impl);
    return h.get();
}

// ---- ELF -> control_code + patch_table (mirrors flm_hrx_replay) ----
template <typename T>
static T elf_rd(const uint8_t* b, size_t off) {
    T v;
    std::memcpy(&v, b + off, sizeof(T));
    return v;
}

static bool parse_control_elf(const std::vector<uint8_t>& e, int scalar_args,
                              std::vector<uint32_t>* control_code,
                              std::vector<uint32_t>* patch_table) {
    if (e.size() < 52 || e[0] != 0x7f || e[1] != 'E') return false;
    const uint8_t* d = e.data();
    uint32_t e_shoff = elf_rd<uint32_t>(d, 0x20);
    uint16_t e_shentsize = elf_rd<uint16_t>(d, 0x2e);
    uint16_t e_shnum = elf_rd<uint16_t>(d, 0x30);
    uint16_t e_shstrndx = elf_rd<uint16_t>(d, 0x32);
    auto sh = [&](int i, int f) { return e_shoff + (size_t)i * e_shentsize + f; };
    size_t shstr_off = elf_rd<uint32_t>(d, sh(e_shstrndx, 0x10));
    auto sname = [&](int i) {
        uint32_t nm = elf_rd<uint32_t>(d, sh(i, 0));
        return std::string((const char*)d + shstr_off + nm);
    };
    int ctrl = -1, rela = -1, dynsym = -1, dynstr = -1;
    for (int i = 0; i < e_shnum; i++) {
        std::string n = sname(i);
        if (n == ".ctrltext") ctrl = i;
        else if (n == ".rela.dyn") rela = i;
        else if (n == ".dynsym") dynsym = i;
        else if (n == ".dynstr") dynstr = i;
    }
    if (ctrl < 0) return false;
    uint32_t coff = elf_rd<uint32_t>(d, sh(ctrl, 0x10));
    uint32_t csize = elf_rd<uint32_t>(d, sh(ctrl, 0x14));
    control_code->resize(csize / 4);
    std::memcpy(control_code->data(), d + coff, csize);
    if (rela >= 0 && dynsym >= 0 && dynstr >= 0) {
        uint32_t roff = elf_rd<uint32_t>(d, sh(rela, 0x10));
        uint32_t rsize = elf_rd<uint32_t>(d, sh(rela, 0x14));
        uint32_t symoff = elf_rd<uint32_t>(d, sh(dynsym, 0x10));
        uint32_t symentsz = elf_rd<uint32_t>(d, sh(dynsym, 0x24));
        if (symentsz == 0) symentsz = 16;
        uint32_t strtab = elf_rd<uint32_t>(d, sh(dynstr, 0x10));
        for (uint32_t o = 0; o + 12 <= rsize; o += 12) {
            uint32_t r_offset = elf_rd<uint32_t>(d, roff + o + 0);
            uint32_t r_info = elf_rd<uint32_t>(d, roff + o + 4);
            int32_t r_addend = elf_rd<int32_t>(d, roff + o + 8);
            uint32_t symidx = r_info >> 8;
            uint32_t st_name = elf_rd<uint32_t>(d, symoff + symidx * symentsz);
            std::string an((const char*)d + strtab + st_name);
            int arg_idx = std::atoi(an.c_str()) - scalar_args;
            if (arg_idx < 0) continue;
            patch_table->push_back(r_offset);
            patch_table->push_back((uint32_t)arg_idx);
            patch_table->push_back((uint32_t)r_addend);
        }
    }
    return true;
}

// ---- live forwarding of FLM dispatches to HRX ----
class Forwarder {
public:
    static Forwarder& get() {
        static Forwarder* f = new Forwarder();
        return *f;
    }
    bool enabled = false;
    hrx_device_t dev = nullptr;
    hrx_stream_t stream = nullptr;
    std::mutex mu;
    // L5: cache the executable AND its resolved "MLIR_AIE" export ordinal, so the
    // string-keyed lookup runs once per distinct control program, not per dispatch.
    struct CachedExe { hrx_executable_t exe = nullptr; uint32_t ord = 0; };
    struct ExecutableKey {
        size_t xclbin_hash = 0;
        size_t control_hash = 0;
        bool operator==(const ExecutableKey& other) const {
            return xclbin_hash == other.xclbin_hash &&
                   control_hash == other.control_hash;
        }
    };
    struct ExecutableKeyHash {
        size_t operator()(const ExecutableKey& key) const {
            return key.xclbin_hash ^
                   (key.control_hash + 0x9e3779b97f4a7c15ull +
                    (key.xclbin_hash << 6) + (key.xclbin_hash >> 2));
        }
    };
    std::unordered_map<ExecutableKey, CachedExe, ExecutableKeyHash> exe_cache;
    std::vector<BoImpl*> pending_out;  // outputs written by un-flushed dispatches
    std::atomic<uint64_t> dispatched{0}, skipped{0}, exe_fail{0};
    std::atomic<uint64_t> alloc_ok{0}, alloc_fail{0};
    std::atomic<uint64_t> sync_to{0}, sync_from{0}, h2d_copies{0}, h2d_skips{0};
    // grouping instrumentation
    std::atomic<uint64_t> n_run_start{0}, n_runlist_exec{0}, n_runlist_runs{0};
    std::atomic<uint64_t> n_run_wait{0}, n_runlist_wait{0}, max_runlist{0};
    // timing (microseconds) to see where decode time goes
    std::atomic<uint64_t> t_h2d{0}, t_disp{0}, t_sync{0}, t_d2h{0};
    std::atomic<uint64_t> t_h2d_bytes{0}, t_d2h_bytes{0};
    // per-dispatch context-switch tracking (xclbin change vs same context)
    XclbinImpl* last_xclbin_disp = nullptr;
    std::atomic<uint64_t> n_same{0}, n_switch{0};
    // chained-path (forward_runlist): #chains formed, #dispatches chained
    std::atomic<uint64_t> n_chain{0}, n_chain_disp{0};
    bool graph_enabled = false;
    std::string graph_path;
    std::ofstream graph;
    uint64_t graph_next_node = 1;
    uint64_t graph_records = 0;

    static void print_stats() {
        auto& f = get();
        std::fprintf(stderr,
                     "[fwd] stats: dispatched=%llu skipped=%llu exe_fail=%llu "
                     "bo_alloc_ok=%llu | sync_to=%llu sync_from=%llu | "
                     "h2d_copies=%llu h2d_skips=%llu\n",
                     (unsigned long long)f.dispatched, (unsigned long long)f.skipped,
                     (unsigned long long)f.exe_fail, (unsigned long long)f.alloc_ok,
                     (unsigned long long)f.sync_to, (unsigned long long)f.sync_from,
                     (unsigned long long)f.h2d_copies, (unsigned long long)f.h2d_skips);
        std::fprintf(stderr,
                     "[fwd] group: run_start=%llu runlist_exec=%llu "
                     "runlist_runs=%llu (max=%llu) run_wait=%llu runlist_wait=%llu\n",
                     (unsigned long long)f.n_run_start,
                     (unsigned long long)f.n_runlist_exec,
                     (unsigned long long)f.n_runlist_runs,
                     (unsigned long long)f.max_runlist,
                     (unsigned long long)f.n_run_wait,
                     (unsigned long long)f.n_runlist_wait);
        std::fprintf(stderr,
                     "[fwd] time(ms): h2d=%llu (%lluMB) dispatch=%llu sync=%llu "
                     "d2h=%llu (%lluMB)\n",
                     (unsigned long long)f.t_h2d / 1000,
                     (unsigned long long)f.t_h2d_bytes / (1024 * 1024),
                     (unsigned long long)f.t_disp / 1000,
                     (unsigned long long)f.t_sync / 1000,
                     (unsigned long long)f.t_d2h / 1000,
                     (unsigned long long)f.t_d2h_bytes / (1024 * 1024));
        std::fprintf(stderr,
                     "[fwd] ctx: same-ctx dispatches=%llu ctx-switch dispatches=%llu\n",
                     (unsigned long long)f.n_same, (unsigned long long)f.n_switch);
        std::fprintf(stderr, "[fwd] chaining: chains=%llu chained_dispatches=%llu\n",
                     (unsigned long long)f.n_chain,
                     (unsigned long long)f.n_chain_disp);
        std::fprintf(stderr,
                     "[fwd] readback (lazy=%d): runlist-deferred=%llu "
                     "host-flushed=%llu\n",
                     f.lazy_readback ? 1 : 0,
                     (unsigned long long)f.d2h_deferred,
                     (unsigned long long)f.d2h_flushed);
        if (f.graph_enabled) {
            std::fprintf(stderr, "[fwd] graph: records=%llu path=%s\n",
                         (unsigned long long)f.graph_records,
                         f.graph_path.c_str());
        }
    }

    bool chain = true;  // batch runlists into one ERT_CMD_CHAIN (FLM_CHAIN=0 off)
    // L2: defer readback of CHAINED runlist outputs. A runlist output is usually
    // consumed by a later runlist (input bound to the device buffer directly), so
    // copying it to host eagerly is waste. We instead mark it dev_dirty and copy
    // it back lazily only if/when the host actually reads it (bo::map or
    // bo::sync(FROM_DEVICE)). Pure on-device intermediates are thus never copied.
    // Singles (run::start) stay eager: those carry host-read results (e.g. the
    // sampled logits) that FLM reads via a cached host pointer without re-sync.
    // ON by default; FLM_LAZY_READBACK=0 disables (read every output back).
    bool lazy_readback = true;
    std::atomic<uint64_t> d2h_deferred{0}, d2h_flushed{0};
    Forwarder() {
        enabled = std::getenv("FLM_FORWARD") != nullptr;
        const char* c = std::getenv("FLM_CHAIN");
        chain = !c || c[0] != '0';  // chaining ON by default; FLM_CHAIN=0 disables
        const char* lz = std::getenv("FLM_LAZY_READBACK");
        lazy_readback = !lz || lz[0] != '0';  // ON by default; =0 disables
        const char* gd = std::getenv("FLM_GRAPH_DUMP");
        if (gd && gd[0]) {
            graph_path = (std::strcmp(gd, "1") == 0) ? "/tmp/flm_graph.jsonl" : gd;
            std::error_code ec;
            std::filesystem::path path(graph_path);
            if (path.has_parent_path())
                std::filesystem::create_directories(path.parent_path(), ec);
            graph.open(graph_path, std::ios::out | std::ios::trunc);
            if (graph) {
                graph_enabled = true;
            } else {
                std::fprintf(stderr, "[fwd] graph dump open FAILED: %s\n",
                             graph_path.c_str());
            }
        }
        if (enabled) std::atexit(&Forwarder::print_stats);
    }

    void ensure_device() {
        if (dev) return;
        if (!hrx_status_is_ok(hrx_gpu_initialize(0)) ||
            !hrx_status_is_ok(hrx_gpu_device_get(0, &dev)) ||
            !hrx_status_is_ok(hrx_stream_create(dev, 0, &stream))) {
            std::fprintf(stderr, "[fwd] HRX device init FAILED\n");
            enabled = false;
        }
    }

    // Build (or fetch cached) executable for this xclbin + parsed control ELF,
    // and resolve its export ordinal once. Returns the executable; *ord_out gets
    // the cached "MLIR_AIE" ordinal.
    hrx_executable_t executable_for(const XclbinImpl& xclbin,
                                    const ElfImpl& elf,
                                    std::vector<uint32_t>* patch_out,
                                    uint32_t* ord_out) {
        static bool dbg = std::getenv("FLM_FORWARD_DEBUG") != nullptr;
        if (!elf.parse_ok || elf.control_code.empty()) {
            if (dbg)
                std::fprintf(stderr,
                             "[fwd] executable_for: parse FAILED (elf=%zuB cc=%zu)\n",
                             elf.bytes.size(), elf.control_code.size());
            return nullptr;
        }
        if (patch_out)
            *patch_out = elf.patch_table;
        ExecutableKey key = {xclbin.content_hash, elf.control_hash};
        auto it = exe_cache.find(key);
        if (it != exe_cache.end()) { if (ord_out) *ord_out = it->second.ord; return it->second.exe; }
        flm_hrx::XadxEntryPoint ep;
        ep.name = "MLIR_AIE";
        ep.pdi_index = 0;
        ep.xclbin_index = 0;
        flm_hrx::XadxRun run;
        run.control_code = elf.control_code;
        run.patch_table = elf.patch_table;
        ep.runs.push_back(run);
        hrx_executable_t exe = nullptr;
        try {
            std::vector<uint8_t> xadx = flm_hrx::build_xadx(xclbin.bytes, {ep});
            // hrx_executable_load_data now requires the exact HAL executable
            // format; build_xadx packages an xclbin-based amdxdna executable.
            hrx_status_t s = hrx_executable_load_data(
                dev, xadx.data(), xadx.size(), "amdxdna-xclbin-fb", &exe);
            if (!hrx_status_is_ok(s)) {
                if (dbg) {
                    char* m = nullptr; size_t n = 0;
                    hrx_status_to_string(s, &m, &n);
                    std::fprintf(stderr,
                                 "[fwd] load_data FAILED (xclbin=%zuB cc=%zu xadx=%zuB): %s\n",
                                 xclbin.bytes.size(), elf.control_code.size(), xadx.size(), m ? m : "?");
                    hrx_status_free_message(m);
                }
                hrx_status_ignore(s);
                exe = nullptr;
            }
        } catch (const std::exception& e) {
            if (dbg) std::fprintf(stderr, "[fwd] build_xadx threw: %s\n", e.what());
            exe = nullptr;
        }
        uint32_t ord = 0;
        if (exe) hrx_executable_lookup_export_by_name(exe, "MLIR_AIE", &ord);
        exe_cache[key] = {exe, ord};
        if (ord_out) *ord_out = ord;
        return exe;
    }

    // Preload the HRX executable at xrt::kernel construction time, when FLM has
    // already paired a hw_context/xclbin with a module/control ELF. This moves
    // executable lookup/build out of the dispatch hot path while preserving the
    // existing lazy fallback for unusual construction orders.
    void resolve_kernel_executable_locked(KernelImpl* kernel) {
        if (!enabled || !kernel || kernel->executable_resolved) return;
        if (!dev) ensure_device();
        if (!dev || !kernel->xclbin || !kernel->module || !kernel->module->elf)
            return;
        uint32_t ord = 0;
        kernel->executable = executable_for(*kernel->xclbin, *kernel->module->elf,
                                            /*patch_out=*/nullptr, &ord);
        kernel->executable_ordinal = ord;
        kernel->executable_resolved = true;
    }
};

// Monotonic microsecond clock for phase instrumentation (A0). Timing is always
// on; it is a couple of clock reads per dispatch and writes to relaxed atomics.
static inline uint64_t now_us() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static void graph_escape(std::ofstream& os, const std::string& value) {
    for (char c : value) {
        switch (c) {
            case '\\': os << "\\\\"; break;
            case '"': os << "\\\""; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default: os << c; break;
        }
    }
}

static uint64_t graph_new_node_locked(Forwarder& fwd) {
    return fwd.graph_next_node++;
}

static void graph_write_binding_locked(Forwarder& fwd, const Binding& binding,
                                       size_t bo_ordinal) {
    if (!binding.is_bo)
        return;
    BoImpl* bo = binding.bi;
    fwd.graph << "{\"slot\":" << binding.index << ",\"ordinal\":" << bo_ordinal
              << ",\"size\":" << binding.size;
    if (bo) {
        fwd.graph << ",\"bo\":" << bo->id
                  << ",\"host_dirty\":" << (bo->host_dirty ? "true" : "false")
                  << ",\"dev_dirty\":" << (bo->dev_dirty ? "true" : "false");
    }
    fwd.graph << "}";
}

static void graph_write_dispatch_locked(Forwarder& fwd, RunImpl* run,
                                        uint64_t runlist_node,
                                        int runlist_ordinal) {
    if (!fwd.graph_enabled || !fwd.graph || !run || !run->kernel)
        return;
    if (!run->graph_node)
        run->graph_node = graph_new_node_locked(fwd);
    KernelImpl* kernel = run->kernel.get();
    fwd.graph << "{\"record\":\"node\",\"node\":" << run->graph_node
              << ",\"kind\":\"dispatch\"";
    if (runlist_node)
        fwd.graph << ",\"runlist\":" << runlist_node
                  << ",\"runlist_ordinal\":" << runlist_ordinal;
    fwd.graph << ",\"kernel\":\"";
    graph_escape(fwd.graph, kernel->name);
    fwd.graph << "\",\"xclbin_hash\":"
              << (kernel->xclbin ? kernel->xclbin->content_hash : 0)
              << ",\"control_hash\":"
              << ((kernel->module && kernel->module->elf)
                      ? kernel->module->elf->control_hash
                      : 0)
              << ",\"bindings\":[";
    size_t bo_ordinal = 0;
    for (const auto& binding : run->args) {
        if (!binding.is_bo)
            continue;
        if (bo_ordinal)
            fwd.graph << ",";
        graph_write_binding_locked(fwd, binding, bo_ordinal++);
    }
    fwd.graph << "]}\n";
    ++fwd.graph_records;
    fwd.graph.flush();
}

static uint64_t graph_write_runlist_locked(
    Forwarder& fwd, std::vector<std::shared_ptr<RunImpl>>& runs) {
    if (!fwd.graph_enabled || !fwd.graph)
        return 0;
    uint64_t node = graph_new_node_locked(fwd);
    fwd.graph << "{\"record\":\"node\",\"node\":" << node
              << ",\"kind\":\"runlist\",\"runs\":[";
    for (size_t i = 0; i < runs.size(); ++i) {
        RunImpl* run = runs[i].get();
        if (run && !run->graph_node)
            run->graph_node = graph_new_node_locked(fwd);
        if (i)
            fwd.graph << ",";
        fwd.graph << (run ? run->graph_node : 0);
    }
    fwd.graph << "]}\n";
    ++fwd.graph_records;
    fwd.graph.flush();
    return node;
}

static void graph_write_bo_event_locked(Forwarder& fwd, const char* event,
                                        BoImpl* bo, size_t size,
                                        size_t offset) {
    if (!fwd.graph_enabled || !fwd.graph || !bo)
        return;
    fwd.graph << "{\"record\":\"event\",\"event\":\"" << event
              << "\",\"bo\":" << bo->id << ",\"size\":" << size
              << ",\"offset\":" << offset << "}\n";
    ++fwd.graph_records;
    fwd.graph.flush();
}

static size_t clamp_bo_range(const BoImpl* bo, size_t size, size_t offset) {
    if (!bo || offset >= bo->size)
        return 0;
    size_t available = bo->size - offset;
    if (size == 0 || size > available)
        return available;
    return size;
}

static bool covers_whole_bo(const BoImpl* bo, size_t size, size_t offset) {
    return bo && offset == 0 && size >= bo->size;
}

static bool flush_mapped_bo_to_device_locked(Forwarder& fwd, BoImpl* bo,
                                             size_t size, size_t offset) {
    size_t clamped_size = clamp_bo_range(bo, size, offset);
    if (!bo || !bo->hbuf || !bo->mapped || clamped_size == 0)
        return false;
    uint64_t t0 = now_us();
    hrx_status_t status = hrx_buffer_flush_range(bo->hbuf, offset, clamped_size);
    if (!hrx_status_is_ok(status)) {
        hrx_status_ignore(status);
        return false;
    }
    fwd.t_h2d += now_us() - t0;
    fwd.t_h2d_bytes += clamped_size;
    fwd.h2d_copies++;
    graph_write_bo_event_locked(fwd, "h2d_upload", bo, clamped_size, offset);
    return true;
}

static void graph_write_node_event_locked(Forwarder& fwd, const char* event,
                                          uint64_t node) {
    if (!fwd.graph_enabled || !fwd.graph || !node)
        return;
    fwd.graph << "{\"record\":\"event\",\"event\":\"" << event
              << "\",\"node\":" << node << "}\n";
    ++fwd.graph_records;
    fwd.graph.flush();
}

static void complete_outputs_locked(Forwarder& fwd,
                                    const std::vector<BoImpl*>& outputs) {
    uint64_t t0 = now_us();
    for (BoImpl* b : outputs) {
        if (!b || !b->hbuf) continue;
        b->host_dirty = false;
        if (b->mapped) {
            if (fwd.lazy_readback) {
                b->dev_dirty = true;
                fwd.d2h_deferred++;
            } else {
                hrx_buffer_invalidate_range(b->hbuf, 0, b->size);
            }
            continue;
        }
        void* p = nullptr;
        if (hrx_status_is_ok(hrx_buffer_map(b->hbuf, HRX_MAP_READ, 0, b->size,
                                            &p)) && p) {
            std::memcpy(b->data.data(), p, b->size);
            hrx_buffer_unmap(b->hbuf);
            fwd.t_d2h_bytes += b->size;
        }
    }
    fwd.t_d2h += now_us() - t0;
}

static hrx_stream_t stream_for_run_locked(Forwarder& fwd, RunImpl* run) {
    if (run && run->kernel && run->kernel->hwctx && run->kernel->hwctx->stream)
        return run->kernel->hwctx->stream;
    return fwd.stream;
}

static hrx_stream_t stream_for_runlist_locked(Forwarder& fwd,
                                              RunlistImpl* runlist) {
    if (runlist && runlist->hwctx && runlist->hwctx->stream)
        return runlist->hwctx->stream;
    return fwd.stream;
}

static void wait_pending_submission_locked(Forwarder& fwd,
                                           PendingSubmission& pending) {
    if (!pending.active)
        return;
    uint64_t t0 = now_us();
    hrx_status_t status = hrx_semaphore_wait(
        pending.completion.semaphore, pending.completion.value, UINT64_MAX);
    fwd.t_sync += now_us() - t0;
    if (hrx_status_is_ok(status)) {
        graph_write_node_event_locked(
            fwd, pending.is_chain ? "runlist_complete" : "dispatch_complete",
            pending.graph_node);
        complete_outputs_locked(fwd, pending.outputs);
    } else {
        hrx_status_ignore(status);
    }
    pending.outputs.clear();
    pending.active = false;
    pending.graph_node = 0;
    pending.is_chain = false;
    pending.completion = {};
}

// Dispatch one captured run through HRX. xrt::run::start submits and records a
// pending completion; xrt::run::wait performs the wait and output visibility
// work. Dirty tracking skips static-weight re-uploads.
static void forward_dispatch(RunImpl* r) {
    auto& fwd = Forwarder::get();
    if (!fwd.enabled || !r || !r->kernel) return;
    auto k = r->kernel;
    if (!k->xclbin || !k->module || !k->module->elf) return;
    std::lock_guard<std::mutex> lk(fwd.mu);
    if (!fwd.dev) return;
    wait_pending_submission_locked(fwd, r->pending);
    hrx_stream_t stream = stream_for_run_locked(fwd, r);
    uint32_t ord = 0;
    if (!k->executable_resolved) {
        fwd.resolve_kernel_executable_locked(k.get());
    }
    hrx_executable_t exe = k->executable;
    ord = k->executable_ordinal;
    if (!exe) { fwd.exe_fail++; return; }
    // context-switch tracking: did this dispatch change xclbin from the last one?
    if (k->xclbin.get() == fwd.last_xclbin_disp) fwd.n_same++;
    else { fwd.n_switch++; fwd.last_xclbin_disp = k->xclbin.get(); }
    std::vector<hrx_buffer_ref_t> binds;
    BoImpl* out = nullptr;
    uint64_t t0 = now_us();
    for (auto& a : r->args) {
        if (!a.is_bo) continue;
        if (!a.hbuf || !a.host) { fwd.skipped++; return; }
        if (!out) out = a.bi;  // binding 0 = arg3 = output
        // h2d only dirty inputs (static weights upload once via sync tracking).
        if (a.bi && !a.bi->host_dirty) { fwd.h2d_skips++; }
        else {
            if (a.bi && a.bi->mapped)
                hrx_buffer_flush_range(a.hbuf, 0, a.size);  // in-place, no copy
            else
                hrx_stream_copy_h2d(stream, a.host, a.hbuf, 0, a.size);
            if (a.bi) a.bi->host_dirty = false;
            fwd.h2d_copies++;
            fwd.t_h2d_bytes += a.size;
            graph_write_bo_event_locked(fwd, "h2d_upload", a.bi, a.size, 0);
        }
        binds.push_back({a.hbuf, 0, a.size});
    }
    if (binds.empty()) { fwd.skipped++; return; }
    graph_write_dispatch_locked(fwd, r, /*runlist_node=*/0,
                                /*runlist_ordinal=*/-1);
    fwd.t_h2d += now_us() - t0;
    hrx_dispatch_config_t cfg = {{1, 1, 1}, {1, 1, 1}, 0};
    uint64_t t1 = now_us();
    hrx_status_t s = hrx_stream_dispatch(stream, exe, ord, &cfg, nullptr, 0,
                                         binds.data(), binds.size(),
                                         HRX_DISPATCH_FLAG_NONE);
    fwd.t_disp += now_us() - t1;
    if (hrx_status_is_ok(s)) {
        uint64_t t2 = now_us();
        hrx_status_t flush_status = hrx_stream_flush(stream);
        fwd.t_disp += now_us() - t2;
        if (hrx_status_is_ok(flush_status)) {
            hrx_stream_get_timeline_position(stream, &r->pending.completion);
            r->pending.active = true;
            r->pending.graph_node = r->graph_node;
            r->pending.is_chain = false;
            r->pending.outputs.clear();
            if (out && out->hbuf)
                r->pending.outputs.push_back(out);
            graph_write_node_event_locked(fwd, "dispatch_submit", r->graph_node);
        } else {
            hrx_status_ignore(flush_status);
        }
    }
    if ((++fwd.dispatched % 20000) == 0) Forwarder::print_stats();
}

// Chained runlist (default; FLM_CHAIN=0 disables): batch a whole runlist into
// one HRX ERT_CMD_CHAIN. execute() submits and records a pending completion;
// wait() performs the wait and output visibility work. Four phases:
//   1. h2d for all dirty inputs,
//   2. record every dispatch (no sync) so HRX coalesces them into one chain,
//   3. ONE flush submits the whole chain,
//   4. wait/readback on xrt::runlist::wait.
// This amortizes per-dispatch submit/completion overhead across the runlist.
// Requires the HRX amdxdna command-chain support (ROCm/hrx-system#37).
// One-shot capture of a real runlist to $FLM_DUMP_RUNLIST_DIR (default
// /tmp/flm_runlist) when FLM_DUMP_RUNLIST=<min-runs> is set: writes xclbin.bin,
// each run's control ELF (run_NN.elf) and a manifest.txt of binding sizes:ids
// (ids reveal buffers shared across runs). Feeds the standalone HRX microbench
// in bench/ (flm_hrx_runlist_bench.cpp). Off unless FLM_DUMP_RUNLIST is set.
static void maybe_capture_runlist(std::vector<std::shared_ptr<RunImpl>>& runs) {
    static const char* min_env = std::getenv("FLM_DUMP_RUNLIST");
    if (!min_env) return;
    static bool dumped = false;
    if (dumped || (int)runs.size() < std::atoi(min_env)) return;
    dumped = true;
    const char* d = std::getenv("FLM_DUMP_RUNLIST_DIR");
    std::string dir = d ? d : "/tmp/flm_runlist";
    (void)system(("mkdir -p '" + dir + "'").c_str());
    std::ofstream man(dir + "/manifest.txt");
    man << "nruns " << runs.size() << "\n";
    bool wrote_xclbin = false;
    int ri = 0;
    for (auto& rp : runs) {
        RunImpl* r = rp.get();
        if (r && r->kernel) {
            auto k = r->kernel;
            if (!wrote_xclbin && k->xclbin) {
                std::ofstream xf(dir + "/xclbin.bin", std::ios::binary);
                xf.write((const char*)k->xclbin->bytes.data(), k->xclbin->bytes.size());
                wrote_xclbin = true;
            }
            if (k->module && k->module->elf) {
                char fn[64]; std::snprintf(fn, sizeof(fn), "/run_%02d.elf", ri);
                std::ofstream ef(dir + fn, std::ios::binary);
                ef.write((const char*)k->module->elf->bytes.data(), k->module->elf->bytes.size());
            }
            man << "run " << ri << " bindings";
            for (auto& a : r->args)
                if (a.is_bo) man << " " << a.size << ":" << (a.bi ? a.bi->id : 0);
            man << "\n";
        }
        ri++;
    }
    std::fprintf(stderr, "[capture] wrote %zu-run runlist to %s\n", runs.size(), dir.c_str());
}

static void forward_runlist(RunlistImpl* runlist) {
    auto& fwd = Forwarder::get();
    if (!fwd.enabled || !fwd.dev || !runlist) return;
    std::lock_guard<std::mutex> lk(fwd.mu);
    auto& runs = runlist->runs;
    wait_pending_submission_locked(fwd, runlist->pending);
    for (auto& rp : runs) {
        if (rp) wait_pending_submission_locked(fwd, rp->pending);
    }
    hrx_stream_t stream = stream_for_runlist_locked(fwd, runlist);
    maybe_capture_runlist(runs);
    uint64_t graph_runlist_node = graph_write_runlist_locked(fwd, runs);
    // phase 1: DMA h2d for all dirty inputs
    uint64_t t0 = now_us();
    for (auto& rp : runs) {
        RunImpl* r = rp.get(); if (!r || !r->kernel) continue;
        for (auto& a : r->args) {
            if (!a.is_bo || !a.hbuf || !a.host) continue;
            if (a.bi && !a.bi->host_dirty) { fwd.h2d_skips++; continue; }
            if (a.bi && a.bi->mapped)
                hrx_buffer_flush_range(a.hbuf, 0, a.size);  // in-place, no copy
            else
                hrx_stream_copy_h2d(stream, a.host, a.hbuf, 0, a.size);
            fwd.t_h2d_bytes += a.size;
            graph_write_bo_event_locked(fwd, "h2d_upload", a.bi, a.size, 0);
            if (a.bi) a.bi->host_dirty = false; fwd.h2d_copies++;
        }
    }
    fwd.t_h2d += now_us() - t0;
    // phase 2: record every dispatch (no sync -> HRX coalesces into one chain).
    // L6: reuse one bindings vector across runs (clear() keeps its capacity) so
    // the hot loop does not heap-allocate a fresh vector per dispatch.
    std::vector<BoImpl*> outs; int n = 0;
    std::vector<hrx_buffer_ref_t> binds;
    uint64_t t1 = now_us();
    int runlist_index = 0;
    for (auto& rp : runs) {
        RunImpl* r = rp.get(); if (!r || !r->kernel) { ++runlist_index; continue; }
        auto k = r->kernel; if (!k->xclbin || !k->module || !k->module->elf) { ++runlist_index; continue; }
        // context-switch tracking across the runlist (and prior dispatches).
        if (k->xclbin.get() == fwd.last_xclbin_disp) fwd.n_same++;
        else { fwd.n_switch++; fwd.last_xclbin_disp = k->xclbin.get(); }
        uint32_t ord = 0;
        if (!k->executable_resolved) {
            fwd.resolve_kernel_executable_locked(k.get());
        }
        hrx_executable_t exe = k->executable;
        ord = k->executable_ordinal;
        if (!exe) { fwd.exe_fail++; ++runlist_index; continue; }
        binds.clear(); BoImpl* out = nullptr;
        for (const auto& a : r->args) { if (!a.is_bo) continue; if (!a.hbuf) { out=nullptr; break; }
            if (!out) out = a.bi; binds.push_back({a.hbuf, 0, a.size}); }
        if (binds.empty()) { ++runlist_index; continue; }
        graph_write_dispatch_locked(fwd, r, graph_runlist_node, runlist_index);
        hrx_dispatch_config_t cfg = {{1,1,1},{1,1,1},0};
        if (hrx_status_is_ok(hrx_stream_dispatch(stream, exe, ord, &cfg, nullptr, 0,
                                                 binds.data(), binds.size(), HRX_DISPATCH_FLAG_NONE))) {
            if (out) outs.push_back(out); n++;
        }
        ++runlist_index;
    }
    fwd.t_disp += now_us() - t1;
    fwd.dispatched += n; fwd.n_chain++; fwd.n_chain_disp += n;
    // phase 3: ONE flush submits the whole chain.
    uint64_t t2 = now_us();
    hrx_status_t flush_status = hrx_stream_flush(stream);
    fwd.t_disp += now_us() - t2;
    if (hrx_status_is_ok(flush_status)) {
        hrx_stream_get_timeline_position(stream, &runlist->pending.completion);
        runlist->pending.active = true;
        runlist->pending.graph_node = graph_runlist_node;
        runlist->pending.is_chain = true;
        runlist->pending.outputs = std::move(outs);
        graph_write_node_event_locked(fwd, "runlist_submit", graph_runlist_node);
    } else {
        hrx_status_ignore(flush_status);
    }
}

// L2: pull a deferred runlist output to host on demand. Called from bo::map()/
// bo::sync(FROM_DEVICE) right before the host reads the buffer. Only acts on
// buffers carrying un-copied device output (dev_dirty); the producing runlist
// already synchronized the stream, so the device buffer is ready.
static void flush_readback(BoImpl* b) {
    auto& fwd = Forwarder::get();
    if (!fwd.enabled || !fwd.lazy_readback || !b || !b->dev_dirty || !b->hbuf)
        return;
    std::lock_guard<std::mutex> lk(fwd.mu);
    if (!b->dev_dirty) return;  // re-check under lock
    uint64_t t0 = now_us();
    if (b->mapped) {
        // direct-map: just invalidate the host cache (no copy).
        hrx_buffer_invalidate_range(b->hbuf, 0, b->size);
    } else {
        void* p = nullptr;
        if (hrx_status_is_ok(hrx_buffer_map(b->hbuf, HRX_MAP_READ, 0, b->size,
                                            &p)) && p) {
            std::memcpy(b->data.data(), p, b->size);
            hrx_buffer_unmap(b->hbuf);
            fwd.t_d2h_bytes += b->size;
        }
    }
    b->dev_dirty = false;
    fwd.t_d2h += now_us() - t0;
    fwd.d2h_flushed++;
    graph_write_bo_event_locked(fwd, "readback", b, b->size, 0);
}

}  // namespace cap

// ================= ABI-matched xrt declarations =================
namespace xrt {

class uuid {
public:
    unsigned char m_uuid[16];
    uuid() { std::memset(m_uuid, 0, 16); }
};
static_assert(sizeof(uuid) == 16, "uuid must be 16 bytes");

class xclbin {
public:
    std::shared_ptr<void> handle;
    explicit xclbin(const std::string& fnm);
    uuid get_uuid() const;
    class kernel {
    public:
        std::shared_ptr<void> handle;
        std::string get_name() const;
    };
    std::vector<kernel> get_kernels() const;
};

class device {
public:
    std::shared_ptr<void> handle;
    explicit device(unsigned int index);
    ~device();
    uuid register_xclbin(const xclbin& xclbin);
    void reset();
};

class hw_context {
public:
    std::shared_ptr<void> handle;
    enum class access_mode : uint8_t { exclusive = 0, shared = 1 };
    hw_context(const device& device, const uuid& xclbin_id, access_mode mode);
    ~hw_context();
};

class elf {
public:
    std::shared_ptr<void> handle;
    elf(const void* data, size_t size);
};

class module {
public:
    std::shared_ptr<void> handle;
    explicit module(const elf& elf);
};

class kernel {
public:
    std::shared_ptr<void> handle;
    ~kernel();
};

class bo {
public:
    std::shared_ptr<void> handle;
    ~bo();
    void* map();
    size_t size() const;
    void sync(xclBOSyncDirection dir, size_t size, size_t offset);
};

namespace ext {
class bo : public xrt::bo {
public:
    bo(const device& device, size_t sz);
};
class kernel : public xrt::kernel {
public:
    kernel(const hw_context& ctx, const module& mod, const std::string& name);
};
}  // namespace ext

class run {
public:
    std::shared_ptr<void> handle;
    explicit run(const kernel& krnl);
    ~run();
    void set_arg_at_index(int index, const void* value, size_t bytes);
    void set_arg_at_index(int index, const bo& boh);
    void start();
    ert_cmd_state wait(const std::chrono::milliseconds& timeout) const;
};

class runlist {
public:
    std::shared_ptr<void> handle;
    explicit runlist(const hw_context& hwctx);
    ~runlist();
    void add(const run& run);
    void add(run&& run);
    void execute();
    void reset();
    void wait(const std::chrono::milliseconds& timeout) const;
};

}  // namespace xrt

// ================= implementations =================
using namespace cap;

template <typename T>
static T* as(const std::shared_ptr<void>& h) {
    return static_cast<T*>(h.get());
}

namespace xrt {

// ---- xclbin ----
static bool dbg() { static bool d = std::getenv("XRT_CAPTURE_DEBUG") != nullptr; return d; }

xclbin::xclbin(const std::string& fnm) {
    auto x = std::make_shared<XclbinImpl>();
    x->path = fnm;
    if (dbg() || std::getenv("FLM_FORWARD_DEBUG"))
        std::fprintf(stderr, "[cap] xclbin(\"%s\") read %zu bytes\n", fnm.c_str(),
                     x->bytes.size());
    Registry::get().last_xclbin = x;  // for hw_context association by load order
    std::ifstream f(fnm, std::ios::binary | std::ios::ate);
    if (f) {
        std::streamsize n = f.tellg();
        f.seekg(0);
        x->bytes.resize((size_t)n);
        f.read(reinterpret_cast<char*>(x->bytes.data()), n);
        x->content_hash = std::hash<std::string_view>{}(
            std::string_view(reinterpret_cast<const char*>(x->bytes.data()),
                             x->bytes.size()));
    }
    handle = x;
}

static void log_uuid(const char* tag, const unsigned char* u) {
    if (!dbg()) return;
    std::fprintf(stderr, "[cap] %s uuid=", tag);
    for (int i = 0; i < 16; i++) std::fprintf(stderr, "%02x", u[i]);
    std::fprintf(stderr, "\n");
}

uuid xclbin::get_uuid() const {
    uuid u;
    auto x = std::static_pointer_cast<XclbinImpl>(handle);
    auto bytes = Registry::get().fresh_uuid(x);
    std::memcpy(u.m_uuid, bytes.data(), 16);
    log_uuid("get_uuid ->", u.m_uuid);
    return u;
}

std::vector<xclbin::kernel> xclbin::get_kernels() const {
    // The engine searches for a kernel whose name starts with "MLIR_AIE".
    // We don't dispatch, so a single placeholder kernel is sufficient.
    xclbin::kernel k;
    auto ki = std::make_shared<std::string>("MLIR_AIE");
    k.handle = ki;
    return {k};
}

std::string xclbin::kernel::get_name() const {
    if (handle) return *std::static_pointer_cast<std::string>(handle);
    return "MLIR_AIE";
}

// ---- device ----
device::device(unsigned int) { handle = std::make_shared<DeviceImpl>(); }
device::~device() = default;
void device::reset() {}
uuid device::register_xclbin(const xclbin& xc) {
    // Record; the binding uuid is produced by xclbin::get_uuid().
    uuid u;
    auto x = std::static_pointer_cast<XclbinImpl>(xc.handle);
    auto bytes = Registry::get().fresh_uuid(x);
    std::memcpy(u.m_uuid, bytes.data(), 16);
    return u;
}

// ---- hw_context ----
hw_context::hw_context(const device&, const uuid& id, access_mode) {
    auto h = std::make_shared<HwCtxImpl>();
    std::array<uint8_t, 16> key{};
    std::memcpy(key.data(), id.m_uuid, 16);
    log_uuid("hw_context recv", id.m_uuid);
    auto& reg = Registry::get();
    std::lock_guard<std::mutex> lk(reg.mu);
    auto it = reg.uuid_to_xclbin.find(key);
    if (it != reg.uuid_to_xclbin.end())
        h->xclbin = it->second;
    else
        h->xclbin = reg.last_xclbin.lock();  // xrt::uuid ABI is opaque; the
        // engine loads each xclbin immediately before its hw_context, so the
        // most-recently-constructed xclbin is the right association.
    if (dbg())
        std::fprintf(stderr, "[cap] hw_context -> xclbin %s\n",
                     h->xclbin ? h->xclbin->path.c_str() : "(none)");
    auto& fwd = Forwarder::get();
    if (fwd.enabled) {
        fwd.ensure_device();
        if (fwd.dev &&
            !hrx_status_is_ok(hrx_stream_create(fwd.dev, 0, &h->stream))) {
            std::fprintf(stderr, "[fwd] HRX hw_context stream create FAILED\n");
        }
    }
    handle = h;
}
hw_context::~hw_context() = default;

// ---- elf (CONTROL-CODE CAPTURE POINT) ----
elf::elf(const void* data, size_t size) {
    auto e = std::make_shared<ElfImpl>();
    e->bytes.assign(reinterpret_cast<const uint8_t*>(data),
                    reinterpret_cast<const uint8_t*>(data) + size);
    e->parse_ok =
        parse_control_elf(e->bytes, 3, &e->control_code, &e->patch_table);
    if (e->parse_ok && !e->control_code.empty()) {
        e->control_hash = std::hash<std::string_view>{}(
            std::string_view(reinterpret_cast<const char*>(e->control_code.data()),
                             e->control_code.size() * sizeof(uint32_t)));
    }
    handle = e;
}

// ---- module ----
module::module(const elf& e) {
    auto m = std::make_shared<ModuleImpl>();
    m->elf = std::static_pointer_cast<ElfImpl>(e.handle);
    handle = m;
}

// ---- kernel ----
kernel::~kernel() = default;

// ext::kernel
ext::kernel::kernel(const hw_context& ctx, const module& mod,
                    const std::string& name) {
    auto k = std::make_shared<KernelImpl>();
    k->module = std::static_pointer_cast<ModuleImpl>(mod.handle);
    if (ctx.handle) {
        k->hwctx = std::static_pointer_cast<HwCtxImpl>(ctx.handle);
        k->xclbin = k->hwctx->xclbin;
    }
    k->name = name;
    auto& fwd = Forwarder::get();
    if (fwd.enabled) {
        std::lock_guard<std::mutex> lk(fwd.mu);
        fwd.resolve_kernel_executable_locked(k.get());
    }
    if (dbg())
        std::fprintf(stderr, "[cap] ext::kernel(\"%s\") xclbin=%s\n", name.c_str(),
                     k->xclbin ? k->xclbin->path.c_str() : "(none)");
    handle = k;
}

// ---- bo ----
bo::~bo() = default;
void* bo::map() {
    if (!handle) return nullptr;
    auto b = as<BoImpl>(handle);
    flush_readback(b);  // L2: pull deferred runlist output before the host reads
    auto& fwd = Forwarder::get();
    if (fwd.enabled && fwd.graph_enabled) {
        std::lock_guard<std::mutex> lk(fwd.mu);
        graph_write_bo_event_locked(fwd, "map", b, b->size, 0);
    }
    return b->mapped ? b->mapped : b->data.data();  // direct map or host staging
}
size_t bo::size() const {
    return handle ? as<BoImpl>(handle)->size : 0;
}
void bo::sync(xclBOSyncDirection dir, size_t size, size_t offset) {
    auto b = handle ? as<BoImpl>(handle) : nullptr;
    auto& fwd = Forwarder::get();
    if (!b || !fwd.enabled) return;
    if (dir == XCL_BO_SYNC_BO_TO_DEVICE) {
        fwd.sync_to++;
        bool uploaded = false;
        size_t clamped_size = clamp_bo_range(b, size, offset);
        {
            std::lock_guard<std::mutex> lk(fwd.mu);
            graph_write_bo_event_locked(fwd, "sync_to", b, clamped_size, offset);
            uploaded = flush_mapped_bo_to_device_locked(fwd, b, clamped_size,
                                                        offset);
        }
        // Staging or partial-sync paths stay dirty so dispatch keeps the
        // conservative full-upload fallback.
        b->host_dirty = !(uploaded && covers_whole_bo(b, clamped_size, offset));
        b->dev_dirty = false;  // host is now the source of truth
    } else {
        fwd.sync_from++;
        flush_readback(b);  // L2: pull deferred runlist output before the host reads
        if (fwd.graph_enabled) {
            std::lock_guard<std::mutex> lk(fwd.mu);
            graph_write_bo_event_locked(fwd, "sync_from", b, size, offset);
        }
    }
}

// ext::bo
ext::bo::bo(const device&, size_t sz) {
    auto b = std::make_shared<BoImpl>();
    b->size = sz;
    b->id = Registry::get().bo_counter++;
    auto& fwd = Forwarder::get();
    if (fwd.enabled) {
        fwd.ensure_device();
        // Preferred path: a device-visible, host-coherent BO mapped ONCE and kept
        // mapped (persistent). The engine reads/writes this mapping directly, so
        // there is no host staging copy (matches native XRT's single shared BO).
        // These BOs are cached, not HW-coherent, so coherence is still maintained
        // with an explicit clflush (hrx_buffer_flush_range/invalidate_range) around
        // device work -- the same cost native XRT pays, minus the redundant memcpy.
        if (fwd.dev &&
            hrx_status_is_ok(hrx_buffer_allocate(
                fwd.stream, sz,
                HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
                HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
                &b->hbuf))) {
            void* p = nullptr;
            if (hrx_status_is_ok(hrx_buffer_map_persistent(
                    b->hbuf, HRX_MAP_READ | HRX_MAP_WRITE, &p)) &&
                p) {
                b->mapped = p;
                std::memset(p, 0, sz);  // flushed before first dispatch (host_dirty)
                fwd.alloc_ok++;
            } else {
                hrx_buffer_release(b->hbuf);
                b->hbuf = nullptr;
            }
        }
        // Fallback: persistent map unavailable -> scoped BO + host staging copy.
        if (!b->mapped && fwd.dev &&
            hrx_status_is_ok(hrx_buffer_allocate(
                fwd.stream, sz,
                HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
                HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED,
                &b->hbuf))) {
            fwd.alloc_ok++;
        } else if (!b->mapped) {
            b->hbuf = nullptr;
            fwd.alloc_fail++;
        }
    }
    if (!b->mapped) b->data.assign(sz, 0);  // host staging (legacy/fallback path)
    handle = b;
}

// ---- run ----
run::run(const kernel& krnl) {
    auto r = std::make_shared<RunImpl>();
    r->kernel = std::static_pointer_cast<KernelImpl>(krnl.handle);
    handle = r;
}
run::~run() = default;
void run::set_arg_at_index(int index, const void* value, size_t bytes) {
    auto r = as<RunImpl>(handle);
    Binding b;
    b.index = index;
    b.is_bo = false;
    b.size = bytes;
    if (value && bytes <= sizeof(uint64_t)) std::memcpy(&b.value, value, bytes);
    r->args.push_back(b);
}
void run::set_arg_at_index(int index, const bo& boh) {
    auto r = as<RunImpl>(handle);
    Binding b;
    b.index = index;
    b.is_bo = true;
    auto bi = boh.handle ? as<BoImpl>(boh.handle) : nullptr;
    b.size = bi ? bi->size : 0;
    b.hbuf = bi ? bi->hbuf : nullptr;
    b.host = bi ? (bi->mapped ? bi->mapped : bi->data.data()) : nullptr;
    b.bi = bi;
    r->args.push_back(b);
}
void run::start() {
    RunImpl* r = as<RunImpl>(handle);
    if (Forwarder::get().enabled) { Forwarder::get().n_run_start++; forward_dispatch(r); }
    else Registry::get().capture(r);
}
ert_cmd_state run::wait(const std::chrono::milliseconds&) const {
    auto& fwd = Forwarder::get();
    if (fwd.enabled) {
        fwd.n_run_wait++;
        RunImpl* r = as<RunImpl>(handle);
        std::lock_guard<std::mutex> lk(fwd.mu);
        wait_pending_submission_locked(fwd, r->pending);
    }
    return ERT_CMD_STATE_COMPLETED;
}

// ---- runlist ----
runlist::runlist(const hw_context& ctx) {
    auto runlist = std::make_shared<RunlistImpl>();
    if (ctx.handle)
        runlist->hwctx = std::static_pointer_cast<HwCtxImpl>(ctx.handle);
    handle = runlist;
}
runlist::~runlist() = default;
void runlist::add(const run& r) {
    as<RunlistImpl>(handle)->runs.push_back(
        std::static_pointer_cast<RunImpl>(r.handle));
}
void runlist::add(run&& r) {
    as<RunlistImpl>(handle)->runs.push_back(
        std::static_pointer_cast<RunImpl>(r.handle));
}
void runlist::execute() {
    auto& F = Forwarder::get();
    bool fwd = F.enabled;
    RunlistImpl* runlist = as<RunlistImpl>(handle);
    auto& runs = runlist->runs;
    if (fwd) {
        F.n_runlist_exec++;
        F.n_runlist_runs += runs.size();
        uint64_t cur = F.max_runlist.load();
        while (runs.size() > cur && !F.max_runlist.compare_exchange_weak(cur, runs.size())) {}
    }
    if (fwd && F.chain) { forward_runlist(runlist); return; }
    for (auto& r : runs) {
        if (fwd) forward_dispatch(r.get());
        else Registry::get().capture(r.get());
    }
}
void runlist::reset() {
    RunlistImpl* runlist = as<RunlistImpl>(handle);
    auto& fwd = Forwarder::get();
    if (fwd.enabled) {
        std::lock_guard<std::mutex> lk(fwd.mu);
        wait_pending_submission_locked(fwd, runlist->pending);
        for (auto& run : runlist->runs) {
            if (run) wait_pending_submission_locked(fwd, run->pending);
        }
    }
    runlist->runs.clear();
}
void runlist::wait(const std::chrono::milliseconds&) const {
    auto& fwd = Forwarder::get();
    if (fwd.enabled) {
        RunlistImpl* runlist = as<RunlistImpl>(handle);
        fwd.n_runlist_wait++;
        std::lock_guard<std::mutex> lk(fwd.mu);
        wait_pending_submission_locked(fwd, runlist->pending);
        for (auto& run : runlist->runs) {
            if (run) wait_pending_submission_locked(fwd, run->pending);
        }
    }
}

}  // namespace xrt
