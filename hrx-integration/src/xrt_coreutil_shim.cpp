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
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
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
};
struct DeviceImpl {};
struct HwCtxImpl {
    std::shared_ptr<XclbinImpl> xclbin;
};
struct ElfImpl {
    std::vector<uint8_t> bytes;
};
struct ModuleImpl {
    std::shared_ptr<ElfImpl> elf;
};
struct KernelImpl {
    std::shared_ptr<ModuleImpl> module;
    std::shared_ptr<XclbinImpl> xclbin;
    std::string name;
};
struct BoImpl {
    std::vector<uint8_t> data;  // host staging the engine maps + read/writes
    size_t size = 0;
    uint64_t id = 0;
    hrx_buffer_t hbuf = nullptr;  // forward-mode device buffer
    void* mapped = nullptr;       // unused (legacy)
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
struct RunImpl {
    std::shared_ptr<KernelImpl> kernel;
    std::vector<Binding> args;
};
struct RunlistImpl {
    std::vector<std::shared_ptr<RunImpl>> runs;
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
        static Forwarder f;
        return f;
    }
    bool enabled = false;
    hrx_device_t dev = nullptr;
    hrx_stream_t stream = nullptr;
    std::mutex mu;
    std::unordered_map<size_t, hrx_executable_t> exe_cache;  // control-hash -> exe
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
    // per-dispatch time split by xclbin-change (context switch) vs same context
    XclbinImpl* last_xclbin_disp = nullptr;
    std::atomic<uint64_t> t_same{0}, n_same{0}, t_switch{0}, n_switch{0};
    // chained-path (forward_runlist): #chains formed, #dispatches chained
    std::atomic<uint64_t> n_chain{0}, n_chain_disp{0};

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
                     "[fwd] time(ms): h2d=%llu (%lluMB) dispatch+sync=%llu "
                     "d2h=%llu (%lluMB) h2d_pre_sync=%llu\n",
                     (unsigned long long)f.t_h2d / 1000,
                     (unsigned long long)f.t_h2d_bytes / (1024 * 1024),
                     (unsigned long long)f.t_disp / 1000,
                     (unsigned long long)f.t_d2h / 1000,
                     (unsigned long long)f.t_d2h_bytes / (1024 * 1024),
                     (unsigned long long)f.t_sync / 1000);
        std::fprintf(stderr,
                     "[fwd] runlists: single-ctx=%llu multi-ctx=%llu  "
                     "intra-runlist ctx-switches=%llu over %llu runs\n",
                     (unsigned long long)f.n_same, (unsigned long long)f.n_switch,
                     (unsigned long long)f.t_switch, (unsigned long long)f.t_same);
        std::fprintf(stderr, "[fwd] chaining: chains=%llu chained_dispatches=%llu\n",
                     (unsigned long long)f.n_chain,
                     (unsigned long long)f.n_chain_disp);
    }

    bool chain = true;  // batch runlists into one ERT_CMD_CHAIN (FLM_CHAIN=0 off)
    Forwarder() {
        enabled = std::getenv("FLM_FORWARD") != nullptr;
        const char* c = std::getenv("FLM_CHAIN");
        chain = !c || c[0] != '0';  // chaining ON by default; FLM_CHAIN=0 disables
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

    // Build (or fetch cached) executable for this xclbin + control ELF.
    hrx_executable_t executable_for(const std::vector<uint8_t>& xclbin,
                                    const std::vector<uint8_t>& elf,
                                    std::vector<uint32_t>* patch_out) {
        static bool dbg = std::getenv("FLM_FORWARD_DEBUG") != nullptr;
        std::vector<uint32_t> cc;
        if (!parse_control_elf(elf, 3, &cc, patch_out) || cc.empty()) {
            if (dbg)
                std::fprintf(stderr,
                             "[fwd] executable_for: parse FAILED (elf=%zuB cc=%zu)\n",
                             elf.size(), cc.size());
            return nullptr;
        }
        size_t h = std::hash<std::string_view>{}(std::string_view(
            (const char*)cc.data(), cc.size() * 4));
        auto it = exe_cache.find(h);
        if (it != exe_cache.end()) return it->second;
        flm_hrx::XadxEntryPoint ep;
        ep.name = "MLIR_AIE";
        ep.pdi_index = 0;
        ep.xclbin_index = 0;
        flm_hrx::XadxRun run;
        run.control_code = cc;
        run.patch_table = *patch_out;
        ep.runs.push_back(run);
        hrx_executable_t exe = nullptr;
        try {
            std::vector<uint8_t> xadx = flm_hrx::build_xadx(xclbin, {ep});
            hrx_status_t s =
                hrx_executable_load_data(dev, xadx.data(), xadx.size(), "", &exe);
            if (!hrx_status_is_ok(s)) {
                if (dbg) {
                    char* m = nullptr; size_t n = 0;
                    hrx_status_to_string(s, &m, &n);
                    std::fprintf(stderr,
                                 "[fwd] load_data FAILED (xclbin=%zuB cc=%zu xadx=%zuB): %s\n",
                                 xclbin.size(), cc.size(), xadx.size(), m ? m : "?");
                    hrx_status_free_message(m);
                }
                hrx_status_ignore(s);
                exe = nullptr;
            }
        } catch (const std::exception& e) {
            if (dbg) std::fprintf(stderr, "[fwd] build_xadx threw: %s\n", e.what());
            exe = nullptr;
        }
        exe_cache[h] = exe;
        return exe;
    }
};

// Dispatch one captured run through HRX (per-dispatch synchronous path): h2d
// dirty inputs, dispatch, synchronize, then read the output back via map() (a
// cheap host-cache invalidate, NOT a per-buffer queue submit). Used for
// xrt::run::start singles; runlists go through forward_runlist (chaining, on by
// default unless FLM_CHAIN=0). Dirty tracking skips static-weight re-uploads.
static void forward_dispatch(RunImpl* r) {
    auto& fwd = Forwarder::get();
    if (!fwd.enabled || !r || !r->kernel) return;
    auto k = r->kernel;
    if (!k->xclbin || !k->module || !k->module->elf) return;
    std::lock_guard<std::mutex> lk(fwd.mu);
    if (!fwd.dev) return;
    std::vector<uint32_t> patch;
    hrx_executable_t exe =
        fwd.executable_for(k->xclbin->bytes, k->module->elf->bytes, &patch);
    if (!exe) { fwd.exe_fail++; return; }
    std::vector<hrx_buffer_ref_t> binds;
    BoImpl* out = nullptr;
    for (auto& a : r->args) {
        if (!a.is_bo) continue;
        if (!a.hbuf || !a.host) { fwd.skipped++; return; }
        if (!out) out = a.bi;  // binding 0 = arg3 = output
        // h2d only dirty inputs (static weights upload once via sync tracking).
        if (a.bi && !a.bi->host_dirty) { fwd.h2d_skips++; }
        else {
            hrx_stream_copy_h2d(fwd.stream, a.host, a.hbuf, 0, a.size);
            if (a.bi) a.bi->host_dirty = false;
            fwd.h2d_copies++;
        }
        binds.push_back({a.hbuf, 0, a.size});
    }
    if (binds.empty()) { fwd.skipped++; return; }
    uint32_t ord = 0;
    hrx_executable_lookup_export_by_name(exe, "MLIR_AIE", &ord);
    hrx_dispatch_config_t cfg = {{1, 1, 1}, {1, 1, 1}, 0};
    hrx_status_t s = hrx_stream_dispatch(fwd.stream, exe, ord, &cfg, nullptr, 0,
                                         binds.data(), binds.size(),
                                         HRX_DISPATCH_FLAG_NONE);
    if (hrx_status_is_ok(s)) {
        hrx_stream_synchronize(fwd.stream);
        // read the output back via map() (cheap host-cache invalidate, no submit).
        if (out && out->hbuf) {
            void* p = nullptr;
            if (hrx_status_is_ok(hrx_buffer_map(out->hbuf, HRX_MAP_READ, 0,
                                                out->size, &p)) && p) {
                std::memcpy(out->data.data(), p, out->size);
                hrx_buffer_unmap(out->hbuf);
            }
            out->host_dirty = false;
        }
    }
    if ((++fwd.dispatched % 20000) == 0) Forwarder::print_stats();
}

// Chained runlist (default; FLM_CHAIN=0 disables): batch a whole runlist into
// one HRX ERT_CMD_CHAIN instead of one synchronous dispatch per run. Four phases:
//   1. h2d for all dirty inputs,
//   2. record every dispatch (no sync) so HRX coalesces them into one chain,
//   3. ONE synchronize runs the whole chain,
//   4. map()-readback the outputs.
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

static void forward_runlist(std::vector<std::shared_ptr<RunImpl>>& runs) {
    auto& fwd = Forwarder::get();
    if (!fwd.enabled || !fwd.dev) return;
    std::lock_guard<std::mutex> lk(fwd.mu);
    maybe_capture_runlist(runs);
    // phase 1: DMA h2d for all dirty inputs
    for (auto& rp : runs) {
        RunImpl* r = rp.get(); if (!r || !r->kernel) continue;
        for (auto& a : r->args) {
            if (!a.is_bo || !a.hbuf || !a.host) continue;
            if (a.bi && !a.bi->host_dirty) { fwd.h2d_skips++; continue; }
            hrx_stream_copy_h2d(fwd.stream, a.host, a.hbuf, 0, a.size);
            fwd.t_h2d_bytes += a.size;
            if (a.bi) a.bi->host_dirty = false; fwd.h2d_copies++;
        }
    }
    // phase 2: record every dispatch (no sync -> HRX coalesces into one chain)
    std::vector<BoImpl*> outs; int n = 0;
    for (auto& rp : runs) {
        RunImpl* r = rp.get(); if (!r || !r->kernel) continue;
        auto k = r->kernel; if (!k->xclbin || !k->module || !k->module->elf) continue;
        std::vector<uint32_t> patch;
        hrx_executable_t exe = fwd.executable_for(k->xclbin->bytes, k->module->elf->bytes, &patch);
        if (!exe) { fwd.exe_fail++; continue; }
        std::vector<hrx_buffer_ref_t> binds; BoImpl* out = nullptr;
        for (const auto& a : r->args) { if (!a.is_bo) continue; if (!a.hbuf) { out=nullptr; break; }
            if (!out) out = a.bi; binds.push_back({a.hbuf, 0, a.size}); }
        if (binds.empty()) continue;
        uint32_t ord = 0; hrx_executable_lookup_export_by_name(exe, "MLIR_AIE", &ord);
        hrx_dispatch_config_t cfg = {{1,1,1},{1,1,1},0};
        if (hrx_status_is_ok(hrx_stream_dispatch(fwd.stream, exe, ord, &cfg, nullptr, 0,
                                                 binds.data(), binds.size(), HRX_DISPATCH_FLAG_NONE))) {
            if (out) outs.push_back(out); n++;
        }
    }
    fwd.dispatched += n; fwd.n_chain++; fwd.n_chain_disp += n;
    // phase 3: ONE synchronize runs the whole chain
    hrx_stream_synchronize(fwd.stream);
    // phase 4: map()-readback outputs (cheap host-cache invalidate, no submit)
    for (BoImpl* b : outs) {
        if (!b || !b->hbuf) continue;
        void* p = nullptr;
        if (hrx_status_is_ok(hrx_buffer_map(b->hbuf, HRX_MAP_READ, 0, b->size, &p)) && p) {
            std::memcpy(b->data.data(), p, b->size); hrx_buffer_unmap(b->hbuf);
            fwd.t_d2h_bytes += b->size;
        }
        b->host_dirty = false;
    }
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
    handle = h;
}
hw_context::~hw_context() = default;

// ---- elf (CONTROL-CODE CAPTURE POINT) ----
elf::elf(const void* data, size_t size) {
    auto e = std::make_shared<ElfImpl>();
    e->bytes.assign(reinterpret_cast<const uint8_t*>(data),
                    reinterpret_cast<const uint8_t*>(data) + size);
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
    if (ctx.handle) k->xclbin = as<HwCtxImpl>(ctx.handle)->xclbin;
    k->name = name;
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
    return as<BoImpl>(handle)->data.data();  // host staging
}
size_t bo::size() const {
    return handle ? as<BoImpl>(handle)->size : 0;
}
void bo::sync(xclBOSyncDirection dir, size_t, size_t) {
    auto b = handle ? as<BoImpl>(handle) : nullptr;
    auto& fwd = Forwarder::get();
    if (!b || !fwd.enabled) return;
    if (dir == XCL_BO_SYNC_BO_TO_DEVICE) {
        fwd.sync_to++;
        b->host_dirty = true;  // engine wrote host; needs re-upload before dispatch
    } else {
        fwd.sync_from++;
    }
}

// ext::bo
ext::bo::bo(const device&, size_t sz) {
    auto b = std::make_shared<BoImpl>();
    b->size = sz;
    b->id = Registry::get().bo_counter++;
    auto& fwd = Forwarder::get();
    b->data.assign(sz, 0);  // host staging the engine maps + read/writes
    if (fwd.enabled) {
        fwd.ensure_device();
        // Device buffer for dispatch; data is synced host<->device explicitly
        // around each dispatch (CALLER_SYNCS_BINDINGS; coherent map is unreliable).
        if (fwd.dev &&
            hrx_status_is_ok(hrx_buffer_allocate(
                fwd.stream, sz,
                HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
                HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED,
                &b->hbuf))) {
            fwd.alloc_ok++;
        } else {
            b->hbuf = nullptr;
            fwd.alloc_fail++;
        }
    }
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
    b.host = bi ? bi->data.data() : nullptr;
    b.bi = bi;
    r->args.push_back(b);
}
void run::start() {
    RunImpl* r = as<RunImpl>(handle);
    if (Forwarder::get().enabled) { Forwarder::get().n_run_start++; forward_dispatch(r); }
    else Registry::get().capture(r);
}
ert_cmd_state run::wait(const std::chrono::milliseconds&) const {
    if (Forwarder::get().enabled) { Forwarder::get().n_run_wait++; }
    return ERT_CMD_STATE_COMPLETED;
}

// ---- runlist ----
runlist::runlist(const hw_context&) {
    handle = std::make_shared<RunlistImpl>();
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
    auto& runs = as<RunlistImpl>(handle)->runs;
    if (fwd) {
        F.n_runlist_exec++;
        F.n_runlist_runs += runs.size();
        uint64_t cur = F.max_runlist.load();
        while (runs.size() > cur && !F.max_runlist.compare_exchange_weak(cur, runs.size())) {}
    }
    if (fwd && F.chain) { forward_runlist(runs); return; }
    for (auto& r : runs) {
        if (fwd) forward_dispatch(r.get());
        else Registry::get().capture(r.get());
    }
}
void runlist::reset() { as<RunlistImpl>(handle)->runs.clear(); }
void runlist::wait(const std::chrono::milliseconds&) const {
    if (Forwarder::get().enabled) { Forwarder::get().n_runlist_wait++; }
}

}  // namespace xrt
