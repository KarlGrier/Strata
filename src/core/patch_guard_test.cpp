// src/core/patch_guard_test.cpp - round 13: the patch guards (patch_guard.hpp) on a real model header and on synthetic
// mismatches.  No GPU: the guards only read headers and set switches.
//
//     build/patch_guard_test MODEL-00001-of-0000N.gguf [MTP_DIR]
//
// 1. the real header: every guard passes (0 off) - this is today's pack;
// 2. one synthetic mismatch per guard (the facts of 1 with one value changed): that guard, and only that guard, turns
//    its patch off (its switch is then the off value).  (round 14: 6 guards - pr245 and ple_prefetch went with their patches)
#include "strata/core/patch_guard.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using strata::core::ModelFacts;

namespace {
const char* kSwitches[] = {"STRATA_SPEC_EXACT", "STRATA_SIDE_MIXER", "STRATA_ROUTER_Q8K",
                           "STRATA_QSA_SPLIT_ATTN", "STRATA_MMVQ_IL", "STRATA_GR_FORK"};

void reset() {
    for (const char* s : kSwitches) setenv(s, "1", 1);
}

std::string offs() {
    std::string r;
    for (const char* s : kSwitches) {
        const char* v = std::getenv(s);
        if (v && std::strcmp(v, "0") == 0) r += std::string(r.empty() ? "" : ",") + s;
    }
    return r;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: patch_guard_test MODEL-00001-of-0000N.gguf [MTP_DIR]\n");
        return 2;
    }
    std::vector<std::string> shards;
    try {
        ModelFacts real = strata::core::read_model_facts({argv[1]}, argc > 2 ? argv[2] : "");
        (void) real;
    } catch (...) {
    }
    // the shards of the split (the engine passes gguf_split_paths; here the test takes the first shard's siblings)
    std::string first = argv[1];
    const size_t at = first.rfind("-00001-of-");
    if (at != std::string::npos) {
        const int n = std::atoi(first.c_str() + at + 10);
        for (int i = 1; i <= n; ++i) {
            char buf[32];
            std::snprintf(buf, sizeof buf, "-%05d-of-", i);
            shards.push_back(first.substr(0, at) + buf + first.substr(at + 10));
        }
    } else {
        shards.push_back(first);
    }
    const ModelFacts real = strata::core::read_model_facts(shards, argc > 2 ? argv[2] : "");
    int fail = 0;
    reset();
    const int off0 = strata::core::run_patch_guards(real);
    std::printf("real header: %d off%s\n", off0, off0 ? "  FAIL" : "");
    fail += off0 != 0;

    struct Case {
        const char* what;
        std::function<void(ModelFacts&)> change;
        const char* want;   // the switch that must be off
    };
    const std::vector<Case> cases = {
        {"draft id outside the vocabulary", [](ModelFacts& m) { m.draft_max_id = m.n_vocab; m.draft_vocab = 1; },
         "STRATA_SPEC_EXACT"},
        {"no QSA indexer layers", [](ModelFacts& m) { m.qsa = false; }, "STRATA_SIDE_MIXER"},
        {"full attention every 6 layers", [](ModelFacts& m) { m.full_attention_interval = 6; }, "STRATA_SIDE_MIXER"},
        {"head size 128", [](ModelFacts& m) { m.key_length = 128; }, "STRATA_QSA_SPLIT_ATTN"},
        {"16 query heads per KV head", [](ModelFacts& m) { m.head_count = 32; }, "STRATA_QSA_SPLIT_ATTN"},
        {"a dense row of 2000 values in Q6_K", [](ModelFacts& m) { m.bad_dense = {"blk.0.attn_qkv.weight"}; },
         "STRATA_MMVQ_IL"},
        {"low rank 256", [](ModelFacts& m) { m.hc_lr = 256; }, "STRATA_GR_FORK"},
        {"hyper-connection weights in Q8_0", [](ModelFacts& m) { m.hc_bf16 = false; }, "STRATA_GR_FORK"},
        // a hidden size change hits every guard that depends on it: B.1 (compiled 2560), A.2 only when not a multiple
        // of 256
        {"n_embd 3072", [](ModelFacts& m) { m.n_embd = 3072; }, "STRATA_GR_FORK"},
        {"n_embd 2600 (not a multiple of 256)", [](ModelFacts& m) { m.n_embd = 2600; }, "STRATA_ROUTER_Q8K"},
    };
    for (const Case& c : cases) {
        ModelFacts m = real;
        c.change(m);
        reset();
        strata::core::run_patch_guards(m);
        const std::string got = offs();
        const bool ok = got.find(c.want) != std::string::npos;
        std::printf("  %-38s -> off: %s%s\n", c.what, got.empty() ? "-" : got.c_str(), ok ? "" : "  FAIL");
        fail += !ok;
    }
    // the forced guard (the engine-level negative test's switch)
    reset();
    setenv("STRATA_GUARD_FORCE", "all", 1);
    const int all = strata::core::run_patch_guards(real);
    unsetenv("STRATA_GUARD_FORCE");
    std::printf("STRATA_GUARD_FORCE=all: %d off%s\n", all, all == 6 ? "" : "  FAIL");
    fail += all != 6;
    std::printf("patch_guard_test: %d failures\n", fail);
    return fail ? 1 : 0;
}
