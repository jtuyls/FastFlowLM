// flm_hrx_runlist_bench: HRX-only microbenchmark of the command-chain path.
//
// Loads ONE real heterogeneous runlist captured from FLM (e.g. 28 distinct
// kernels: xclbin + per-run control ELF + binding sizes/ids), and runs it two
// ways through pure HRX (no shim, no FLM runtime):
//   A) CHAIN     : N hrx_stream_dispatch (no sync) -> ONE hrx_stream_synchronize
//                  -> the amdxdna ERT_CMD_CHAIN path.
//   B) SEPARATE  : per run, hrx_stream_dispatch + hrx_stream_synchronize
//                  -> N individual START_NPU submits.
// Reports per-dispatch wall time for each and the chain/separate ratio.
//
// Capture a runlist, build, and run: see README.md. In short:
//   FLM_DUMP_RUNLIST=20 ../scripts/run.sh run qwen3:0.6b   # writes /tmp/flm_runlist
//   ./build.sh                                             # -> build/flm_hrx_runlist_bench
//   build/flm_hrx_runlist_bench /tmp/flm_runlist

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "hrx_runtime.h"
#include "hrx_xadx_builder.hpp"

static bool ok(const char* w, hrx_status_t s) {
  if (hrx_status_is_ok(s)) return true;
  char* m = nullptr; size_t n = 0; hrx_status_to_string(s, &m, &n);
  std::fprintf(stderr, "FAIL %s: %s\n", w, m ? m : "?"); hrx_status_free_message(m);
  hrx_status_ignore(s); return false;
}
static std::vector<uint8_t> slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) return {};
  std::streamsize n = f.tellg(); f.seekg(0);
  std::vector<uint8_t> b((size_t)n); f.read((char*)b.data(), n); return b;
}
template <typename T> static T rd(const std::vector<uint8_t>& b, size_t o) {
  T v; std::memcpy(&v, b.data() + o, sizeof(T)); return v;
}
// minimal ELF32 parse: .ctrltext -> control_code, .rela.dyn -> patch_table
static bool parse(const std::vector<uint8_t>& e, std::vector<uint32_t>* cc,
                  std::vector<uint32_t>* pt) {
  if (e.size() < 52 || e[0] != 0x7f) return false;
  uint32_t shoff = rd<uint32_t>(e, 0x20); uint16_t shent = rd<uint16_t>(e, 0x2e);
  uint16_t shnum = rd<uint16_t>(e, 0x30); uint16_t shstr = rd<uint16_t>(e, 0x32);
  auto sh = [&](int i, int f) { return shoff + (size_t)i * shent + f; };
  size_t stroff = rd<uint32_t>(e, sh(shstr, 0x10));
  auto nm = [&](int i) { return std::string((const char*)e.data() + stroff + rd<uint32_t>(e, sh(i, 0))); };
  int ct = -1, rl = -1, ds = -1, dstr = -1;
  for (int i = 0; i < shnum; i++) { auto n = nm(i);
    if (n == ".ctrltext") ct = i; else if (n == ".rela.dyn") rl = i;
    else if (n == ".dynsym") ds = i; else if (n == ".dynstr") dstr = i; }
  if (ct < 0) return false;
  uint32_t co = rd<uint32_t>(e, sh(ct, 0x10)), cs = rd<uint32_t>(e, sh(ct, 0x14));
  cc->resize(cs / 4); std::memcpy(cc->data(), e.data() + co, cs);
  if (rl >= 0 && ds >= 0 && dstr >= 0) {
    uint32_t ro = rd<uint32_t>(e, sh(rl, 0x10)), rs = rd<uint32_t>(e, sh(rl, 0x14));
    uint32_t so = rd<uint32_t>(e, sh(ds, 0x10)), se = rd<uint32_t>(e, sh(ds, 0x24));
    if (!se) se = 16; uint32_t st = rd<uint32_t>(e, sh(dstr, 0x10));
    for (uint32_t o = 0; o + 12 <= rs; o += 12) {
      uint32_t off = rd<uint32_t>(e, ro + o), info = rd<uint32_t>(e, ro + o + 4);
      int32_t add = rd<int32_t>(e, ro + o + 8);
      uint32_t stn = rd<uint32_t>(e, so + (info >> 8) * se);
      int arg = std::atoi((const char*)e.data() + st + stn) - 3;
      if (arg < 0) continue;
      pt->push_back(off); pt->push_back((uint32_t)arg); pt->push_back((uint32_t)add);
    }
  }
  return true;
}

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
  return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
}

struct Bind { uint64_t size; uint64_t id; };

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "/tmp/flm_runlist";
  // parse manifest
  std::ifstream man(dir + "/manifest.txt");
  if (!man) { std::fprintf(stderr, "no manifest in %s\n", dir.c_str()); return 1; }
  int nruns = 0;
  std::vector<std::vector<Bind>> run_binds;
  std::string line;
  while (std::getline(man, line)) {
    std::istringstream ss(line); std::string tok; ss >> tok;
    if (tok == "nruns") { ss >> nruns; }
    else if (tok == "run") {
      int idx; ss >> idx; std::string b; ss >> b;  // "bindings"
      std::vector<Bind> bs; std::string sz;
      while (ss >> sz) {
        auto c = sz.find(':');
        Bind bb{strtoull(sz.substr(0, c).c_str(), 0, 0),
                c == std::string::npos ? 0 : strtoull(sz.substr(c + 1).c_str(), 0, 0)};
        bs.push_back(bb);
      }
      run_binds.push_back(bs);
    }
  }
  nruns = (int)run_binds.size();
  if (nruns == 0) { std::fprintf(stderr, "no runs parsed\n"); return 1; }
  auto xclbin = slurp(dir + "/xclbin.bin");
  if (xclbin.empty()) { std::fprintf(stderr, "no xclbin.bin\n"); return 1; }
  std::printf("loaded runlist: %d runs\n", nruns);

  if (!ok("init", hrx_gpu_initialize(0))) return 1;
  hrx_device_t dev = nullptr; if (!ok("dev", hrx_gpu_device_get(0, &dev))) return 1;
  hrx_stream_t st = nullptr; if (!ok("stream", hrx_stream_create(dev, 0, &st))) return 1;

  // Build one executable per run (xclbin + that run's control code + patch table).
  std::vector<hrx_executable_t> exes(nruns, nullptr);
  std::vector<uint32_t> ords(nruns, 0);
  for (int i = 0; i < nruns; i++) {
    char fn[64]; std::snprintf(fn, sizeof(fn), "/run_%02d.elf", i);
    auto elf = slurp(dir + fn);
    std::vector<uint32_t> cc, pt;
    if (elf.empty() || !parse(elf, &cc, &pt)) { std::fprintf(stderr, "parse fail run %d\n", i); return 1; }
    flm_hrx::XadxEntryPoint ep; ep.name = "MLIR_AIE"; ep.pdi_index = 0; ep.xclbin_index = 0;
    flm_hrx::XadxRun run; run.control_code = cc; run.patch_table = pt; ep.runs.push_back(run);
    auto xadx = flm_hrx::build_xadx(xclbin, {ep});
    if (!ok("load", hrx_executable_load_data(dev, xadx.data(), xadx.size(), "", &exes[i]))) return 1;
    hrx_executable_lookup_export_by_name(exes[i], "MLIR_AIE", &ords[i]);
  }

  // Allocate one device buffer per unique binding id (shared ids reused).
  std::map<uint64_t, hrx_buffer_t> bufs;
  uint64_t total = 0;
  for (auto& bs : run_binds)
    for (auto& b : bs)
      if (!bufs.count(b.id)) {
        hrx_buffer_t buf = nullptr;
        if (!ok("alloc", hrx_buffer_allocate(st, b.size,
                HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
                HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &buf))) return 1;
        bufs[b.id] = buf; total += b.size;
      }
  std::printf("allocated %zu unique buffers, %.2f GB total\n", bufs.size(), total / 1e9);

  // Per-run binding ref lists.
  std::vector<std::vector<hrx_buffer_ref_t>> binds(nruns);
  for (int i = 0; i < nruns; i++)
    for (auto& b : run_binds[i])
      binds[i].push_back({bufs[b.id], 0, b.size});

  hrx_dispatch_config_t cfg = {{1, 1, 1}, {1, 1, 1}, 0};
  auto disp = [&](int i) {
    hrx_stream_dispatch(st, exes[i], ords[i], &cfg, nullptr, 0,
                        binds[i].data(), binds[i].size(), HRX_DISPATCH_FLAG_NONE);
  };

  const int WARM = 3, ITERS = 20;
  // warmup both paths
  for (int w = 0; w < WARM; w++) {
    for (int i = 0; i < nruns; i++) { disp(i); hrx_stream_synchronize(st); }
    for (int i = 0; i < nruns; i++) disp(i); hrx_stream_synchronize(st);
  }

  auto us = [](clk::time_point a, clk::time_point b) {
    return (double)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count(); };

  // B) SEPARATE: dispatch + sync each. Split dispatch(build) vs sync(submit+wait).
  double sep_disp = 0, sep_sync = 0;
  auto t0 = clk::now();
  for (int it = 0; it < ITERS; it++)
    for (int i = 0; i < nruns; i++) {
      auto a = clk::now(); disp(i); auto b = clk::now();
      hrx_stream_synchronize(st); auto c = clk::now();
      sep_disp += us(a, b); sep_sync += us(b, c);
    }
  double sep = ms(t0, clk::now());

  // A) CHAIN: N dispatch (build), one sync (prepare_chain + submit + wait).
  double chn_disp = 0, chn_sync = 0;
  auto t1 = clk::now();
  for (int it = 0; it < ITERS; it++) {
    auto a = clk::now();
    for (int i = 0; i < nruns; i++) disp(i);
    auto b = clk::now();
    hrx_stream_synchronize(st);
    auto c = clk::now();
    chn_disp += us(a, b); chn_sync += us(b, c);
  }
  double chn = ms(t1, clk::now());

  int total_disp = ITERS * nruns;
  std::printf("\n=== %d-run heterogeneous FLM runlist, %d iters ===\n", nruns, ITERS);
  std::printf("SEPARATE (dispatch+sync each): %.1f us/dispatch  (%.1f ms/runlist)\n",
              sep * 1000.0 / total_disp, sep / ITERS);
  std::printf("CHAIN    (N dispatch, 1 sync): %.1f us/dispatch  (%.1f ms/runlist)\n",
              chn * 1000.0 / total_disp, chn / ITERS);
  std::printf("ratio chain/separate = %.2fx  (%s)\n", chn / sep,
              chn < sep ? "chain FASTER" : "chain SLOWER -> regression");
  std::printf("  SEPARATE breakdown: dispatch(build)=%.1f us/disp  sync(submit+wait)=%.1f us/disp\n",
              sep_disp / total_disp, sep_sync / total_disp);
  std::printf("  CHAIN    breakdown: dispatch(build)=%.1f us/disp  sync(prep+submit+wait)=%.1f us/runlist (%.1f us/disp)\n",
              chn_disp / total_disp, chn_sync / ITERS, chn_sync / total_disp);

  for (auto& kv : bufs) hrx_buffer_release(kv.second);
  for (auto e : exes) hrx_executable_release(e);
  hrx_stream_release(st); hrx_gpu_shutdown();
  return 0;
}
