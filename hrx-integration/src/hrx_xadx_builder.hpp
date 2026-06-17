/// \file hrx_xadx_builder.hpp
/// \brief Build an HRX amdxdna XADX executable flatbuffer from an FLM xclbin +
///        a TXN control-code stream (the output of npu_sequence::dump()).
///
/// This is the keystone of the FLM-on-HRX port. FLM's npu_app today wraps a TXN
/// instruction stream into an ELF via aiebu and dispatches it with the public
/// xrt:: API. HRX cannot enumerate the NPU through full XRT (its /opt/xilinx/xrt
/// lacks libxrt_driver_xdna.so), but it *can* dispatch a XADX executable through
/// its public "direct executable" API:
///
///   hrx_executable_load_data(dev, xadx_blob, size, ""/*auto-infer*/, &exe);
///   hrx_stream_dispatch(stream, exe, /*ordinal=*/0, &cfg,
///                       /*constants=*/nullptr, 0, bindings, n, flags);
///
/// XADX schema (hrx/runtime/src/iree/schemas/amdxdna_xclbin_executable_def.fbs):
///   ExecutableDef { xclbins:[XclbinDef{xclbin:string}],
///                   entry_points:[EntryPointDef{ name, pdi_index, xclbin_index,
///                                                runs:[RunDef] }] }
///   RunDef { control_code:[u32], data_payload:[u32], patch_table:[u32] }
///
/// For a basic single dispatch only `control_code` is required (patch_table is
/// optional — BO addresses are applied from the TXN's own DDR-patch ops plus the
/// binding order; the explicit patch_table is only the ERT_CMD_CHAIN host-patch
/// optimization). See hrx executable.c::iree_hal_amdxdna_verify_run_list.
///
/// The flatcc calls below mirror BuildXadx() in
/// hrx/runtime/src/iree/hal/drivers/amdxdna/executable_xclbin_test.cc, which is
/// the verified producer for this format.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// flatcc builder + the generated XADX schema builder. These come from the HRX
// build tree; the FLM build wires the include dirs (see workspace build notes):
//   -I <hrx>/runtime/src
//   -I <hrx-build>/runtime/src            (generated *_builder.h live here)
#include "iree/base/internal/flatcc/building.h"
#include "iree/schemas/amdxdna_xclbin_executable_def_builder.h"

namespace flm_hrx {

/// One dispatchable entry point: a name plus its ordered runs. Each run carries
/// a TXN control-code stream (npu_sequence::dump()); data_payload/patch_table
/// stay empty for the basic dispatch path.
struct XadxRun {
    std::vector<uint32_t> control_code;
    std::vector<uint32_t> data_payload;  // optional control-packet reconfig
    std::vector<uint32_t> patch_table;   // optional ERT_CMD_CHAIN host-patch
};

struct XadxEntryPoint {
    std::string name;            // FLM uses the xclbin's "MLIR_AIE*" kernel name
    int32_t pdi_index = 0;       // index into the xclbin's AIE_PARTITION PDIs
    int32_t xclbin_index = 0;    // index into ExecutableDef.xclbins
    std::vector<XadxRun> runs;
};

inline flatbuffers_uint32_vec_ref_t create_u32_vec(
    flatbuffers_builder_t* b, const std::vector<uint32_t>& v) {
    return flatbuffers_uint32_vec_create(b, v.empty() ? nullptr : v.data(),
                                         v.size());
}

/// Build a XADX flatbuffer wrapping `xclbin` (raw .xclbin bytes) with the given
/// entry points. Returns the serialized flatbuffer bytes ready for
/// hrx_executable_load_data(format=""). Throws std::runtime_error on failure.
inline std::vector<uint8_t> build_xadx(
    const std::vector<uint8_t>& xclbin,
    const std::vector<XadxEntryPoint>& entry_points) {
    flatbuffers_builder_t builder;
    if (flatcc_builder_init(&builder) != 0) {
        throw std::runtime_error("flatcc_builder_init failed");
    }

    auto fail = [&](const char* msg) {
        flatcc_builder_clear(&builder);
        throw std::runtime_error(std::string("build_xadx: ") + msg);
    };

    if (flatbuffers_failed(
            iree_hal_amdxdna_xclbin_ExecutableDef_start_as_root(&builder))) {
        fail("start_as_root");
    }

    // xclbins: a single container holding FLM's raw xclbin bytes.
    flatbuffers_string_ref_t xclbin_str = flatbuffers_string_create(
        &builder, reinterpret_cast<const char*>(xclbin.data()), xclbin.size());
    iree_hal_amdxdna_xclbin_XclbinDef_ref_t xclbin_ref =
        iree_hal_amdxdna_xclbin_XclbinDef_create(&builder, xclbin_str);
    if (!xclbin_str || !xclbin_ref) fail("XclbinDef create");
    iree_hal_amdxdna_xclbin_XclbinDef_ref_t xclbin_arr[] = {xclbin_ref};
    iree_hal_amdxdna_xclbin_XclbinDef_vec_ref_t xclbins_ref =
        iree_hal_amdxdna_xclbin_XclbinDef_vec_create(&builder, xclbin_arr, 1);

    // entry points
    std::vector<iree_hal_amdxdna_xclbin_EntryPointDef_ref_t> entry_refs;
    for (const auto& ep : entry_points) {
        std::vector<iree_hal_amdxdna_xclbin_RunDef_ref_t> run_refs;
        for (const auto& run : ep.runs) {
            iree_hal_amdxdna_xclbin_RunDef_ref_t run_ref =
                iree_hal_amdxdna_xclbin_RunDef_create(
                    &builder, create_u32_vec(&builder, run.control_code),
                    create_u32_vec(&builder, run.data_payload),
                    create_u32_vec(&builder, run.patch_table));
            if (!run_ref) fail("RunDef create");
            run_refs.push_back(run_ref);
        }
        flatbuffers_string_ref_t name_ref =
            flatbuffers_string_create_str(&builder, ep.name.c_str());
        iree_hal_amdxdna_xclbin_RunDef_vec_ref_t runs_ref =
            iree_hal_amdxdna_xclbin_RunDef_vec_create(&builder, run_refs.data(),
                                                      run_refs.size());
        iree_hal_amdxdna_xclbin_EntryPointDef_ref_t ep_ref = 0;
        if (name_ref && runs_ref &&
            !flatbuffers_failed(
                iree_hal_amdxdna_xclbin_EntryPointDef_start(&builder)) &&
            !flatbuffers_failed(iree_hal_amdxdna_xclbin_EntryPointDef_name_add(
                &builder, name_ref)) &&
            !flatbuffers_failed(
                iree_hal_amdxdna_xclbin_EntryPointDef_pdi_index_add(
                    &builder, ep.pdi_index)) &&
            !flatbuffers_failed(
                iree_hal_amdxdna_xclbin_EntryPointDef_xclbin_index_add(
                    &builder, ep.xclbin_index)) &&
            !flatbuffers_failed(iree_hal_amdxdna_xclbin_EntryPointDef_runs_add(
                &builder, runs_ref))) {
            ep_ref = iree_hal_amdxdna_xclbin_EntryPointDef_end(&builder);
        }
        if (!ep_ref) fail("EntryPointDef create");
        entry_refs.push_back(ep_ref);
    }

    iree_hal_amdxdna_xclbin_EntryPointDef_vec_ref_t entries_ref =
        iree_hal_amdxdna_xclbin_EntryPointDef_vec_create(
            &builder, entry_refs.data(), entry_refs.size());
    if (!xclbins_ref || !entries_ref ||
        flatbuffers_failed(iree_hal_amdxdna_xclbin_ExecutableDef_xclbins_add(
            &builder, xclbins_ref)) ||
        flatbuffers_failed(
            iree_hal_amdxdna_xclbin_ExecutableDef_entry_points_add(
                &builder, entries_ref))) {
        fail("populate ExecutableDef");
    }
    if (!iree_hal_amdxdna_xclbin_ExecutableDef_end_as_root(&builder)) {
        fail("end_as_root");
    }

    size_t size = 0;
    void* buf = flatcc_builder_finalize_buffer(&builder, &size);
    if (!buf) fail("finalize_buffer");
    std::vector<uint8_t> out(static_cast<uint8_t*>(buf),
                             static_cast<uint8_t*>(buf) + size);
    free(buf);
    flatcc_builder_clear(&builder);
    return out;
}

}  // namespace flm_hrx
