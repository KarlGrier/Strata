// include/strata/prefill/moe_fused_iq.hpp - #136: the fused int8 prompt experts (moe_fused.hpp) for the native GGUF
// packs (tools/iq_pack.py: an expert is its raw GGUF slices [gate rows | up rows | down rows], formats per layer in
// expert_layout().fmt), opt-in with the same STRATA_PF_FUSED=1 (NVIDIA sm_80 and newer; sm_75 and HIP: MMQ).
//
// The grouping on the GPU (fused::group), the per-expert pointers (fused::Batch: a cache slot or a ring slot, no
// gather) and the launch shape are the Q2_0 path's.  What differs is the load stage: the i-quant blocks are decoded to
// int8 in shared memory before the mma (codebook and sign lookups, as llama.cpp's MMQ load_tiles does), with the
// block's scale per 32 (or per 16: IQ2_XS, IQ2_S, multiplied at m16n8k16) beside them.  The activations and H are
// int8 per 32 values in their natural order here (quantize_act_native), not the Q2_0 path's permuted one.
//
// Covered: gate/up IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS; down Q2_0, IQ4_NL - every layer of the IQ2_XS,
// IQ3_XXS and IQ3_S packs but the IQ1_M ones (which MMQ does not cover either).  Other layers keep MMQ.
#pragma once

#include "strata/prefill/moe_fused.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::prefill::fused {

/// One layer's native expert geometry (cpu::NativeFmt's fields the kernels read; n_embd 2560, n_ff 640).
struct NativeGeom {
    int gu_type = -1, d_type = -1;    ///< ggml types
    size_t gu_row = 0, d_row = 0;     ///< bytes per gate/up row and per down row
    size_t up_off = 0, down_off = 0;  ///< inside the blob
};

/// enabled() (STRATA_PF_FUSED=1 on sm_80+), and the kernels cover this gate/up and down pair on this device.
bool native_supported(int gu_type, int d_type);

/// x [rows][cols] FP32 -> `xa` (act_bytes(rows, cols)): int8 per 32 values in natural order, the native kernels' form.
void quantize_act_native(const float* x, int64_t rows, int64_t cols, void* xa, void* stream);

/// experts() for native blobs: b.blob[e - e0] is the native blob of expert e (gate at 0, up at g.up_off, down at
/// g.down_off; 2-byte aligned).  `xa` from quantize_act_native; H in `ha` (act_bytes(n, 640)); down into `dm` at the
/// rows of `group`.  `dm` 16-byte aligned (gfx12 stores float4; checked there - the prompt path's buffers are 256-aligned
/// through Alloc's 256-byte takes from a 256-aligned cache slot or cudaMalloc).
void experts_native(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
                    const int32_t* src, void* ha, float* dm, void* stream);
/// The NT = 256 kernels (STRATA_PF_FUSED_NT=256 or _F16=1) or the warp-specialized kernel (STRATA_PF_FUSED_WS=1) are
/// set up on the current device: experts_native takes a WideBatch (STRATA_PF_BATCH up to kBatchRoom experts a launch).
bool native_wide();
/// experts_native() of a WideBatch, on the NT = 256 kernels or the warp-specialized kernel; only when native_wide().
void experts_native(const WideBatch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
                    const int32_t* src, void* ha, float* dm, void* stream);

// ---- v2 (adapted from Basalt (MIT), src/prefill/moe_fused_iq.cu at d9915bbc; see moe_fused_iq.cu's v2 section)
//
// STRATA_PREFILL_V2 (read once, at the first use - after the patch guards, which may set it to 0): unset or 0 = off,
// experts_native() runs exactly the kernels above; 1 = IQ3_XXS gate/up and IQ4_NL down on the v2 kernels (raw weight
// bytes by cp.async, decoded straight into the mma fragments, a warp per 32 weight rows x up to 128 routed rows, an
// atomic item queue); 2 = gate/up only, 3 = down only (to tell the two apart in an A/B).  Every other format, and a
// device the v2 kernels do not run on (below sm_80, too little shared memory, no device code), keeps the kernels above.
// The same math as the kernels above, each 32-value term rounded in another order: see moe_fused_iq.cu.

/// The v2 kernels' compiled shapes: K of gate/up (n_embd) and of down (the experts' n_ff), and the dynamic shared memory
/// of the larger one (gate/up; down needs 1,024 bytes less).  The patch guard compares them with the model and the GPUs.
constexpr int kV2Embd = 2560, kV2FF = 640;
constexpr int kV2SharedBytes = 99856;
/// The bytes group_bytes() adds past the grouping tables for the v2 item counters, when v2 is asked for.  A buffer that
/// was sized while v2 was off and then runs it under native_v2_override() (the tests) adds them itself.
constexpr size_t kV2CounterBytes = 64;

/// The request: 0 off, 1 both, 2 gate/up only, 3 down only (native_v2_override() if set, else STRATA_PREFILL_V2).
int native_v2_mode();
/// native_v2_mode() != 0.  No device work.
bool native_v2_requested();
/// This build has the v2 kernels (a CUDA build; HIP and the stubs: false).
bool native_v2_built();
/// Requested, and the v2 kernels run on the current device (set up once per device; a device they do not run on gets
/// one log line and the kernels above).
bool native_v2_ready();
/// The v2 kernels would take this format for gate/up (`gate_up`) or down under the current request.
bool native_v2_takes(int type, bool gate_up);
/// Tests: 0..3 force a mode, -1 returns to STRATA_PREFILL_V2.
void native_v2_override(int mode);

}  // namespace strata::prefill::fused
