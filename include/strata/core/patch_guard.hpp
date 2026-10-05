// include/strata/core/patch_guard.hpp - round 13: the startup check of the local patches' assumptions about the model.
//
// Every local patch that assumes something about the model (a size, a tensor format, the vocabulary, the drafter's
// files) names it here, and `run_patch_guards` compares it with the model's own header once, before any kernel reads
// its switch.  A mismatch logs ONE line and turns that patch off (its switch, or the setter of a switch that is read
// at program start), so upstream's path runs; nothing else changes.  patches/LEDGER.md lists each guard's patch,
// switch and assumption.
//
//     STRATA_GUARD_FORCE=name[,name...|all]   fail these guards on purpose (the negative test: the engine must start
//                                             and compute what the patch's switch off computes)
//     STRATA_GUARD_LIST=1                     one line per guard, also when it passes
#pragma once

#include <string>
#include <vector>

namespace strata::core {

/// The facts the guards compare, read from the model's header (every shard) and the drafter's directory.
struct ModelFacts {
    std::string arch;
    long long n_embd = 0, n_layers = 0, nextn = 0, n_expert = 0, n_expert_used = 0, n_ff = 0, n_ff_shexp = 0;
    long long head_count = 0, head_count_kv = 0, key_length = 0, hc = 0, hc_lr = 0, full_attention_interval = 0;
    long long ssm_inner = 0, ssm_state = 0, ssm_groups = 0, indexer_heads = 0, indexer_key = 0;
    long long ple_dim = 0, ple_ngram = 0, ple_heads = 0, n_vocab = 0, draft_vocab = -1, draft_max_id = -1;
    bool ple_table = false, hc_bf16 = true, gdn = false, qsa = false;
    std::vector<std::string> bad_dense;    // 2-D dense weights in a B.3 format whose row length is not a multiple of 256
    std::vector<std::string> expert_types; // the routed experts' gate/up and down formats, "gate/down" per distinct pair
};

/// Reads the facts; `mtp_dir` may be empty (no drafter).  Throws only when the header cannot be read.
ModelFacts read_model_facts(const std::vector<std::string>& native_shards, const std::string& mtp_dir);

/// The guards: logs a summary line (and one line per mismatch), turns failed patches off.  Returns the number off.
int run_patch_guards(const ModelFacts& f);

}  // namespace strata::core
