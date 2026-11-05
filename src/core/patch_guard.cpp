// src/core/patch_guard.cpp - round 13: see include/strata/core/patch_guard.hpp and patches/LEDGER.md.
#include "strata/core/patch_guard.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace strata::core {
namespace {

long long meta_u(const GgufFile& g, const std::string& key, long long dflt = 0) {
    const MetaValue* v = g.get(key);
    return v != nullptr && v->is_num() ? (long long) v->u : dflt;
}

bool has_suffix(const std::string& s, const char* suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

std::string str(long long v) { return std::to_string(v); }

struct Guard {
    const char* name;      // the ledger's name
    const char* patch;     // what it is
    const char* off_how;   // how it is turned off (for the log line)
    std::function<std::string(const ModelFacts&)> check;   // "" = the model is what the patch assumes
    std::function<void()> off;
};

void setenv_off(const char* key, const char* value) { setenv(key, value, 1); }

bool forced(const char* name) {
    const char* v = std::getenv("STRATA_GUARD_FORCE");
    if (v == nullptr || *v == '\0') return false;
    std::stringstream ss(v);
    std::string item;
    while (std::getline(ss, item, ','))
        if (item == "all" || item == name) return true;
    return false;
}

}  // namespace

ModelFacts read_model_facts(const std::vector<std::string>& native_shards, const std::string& mtp_dir) {
    ModelFacts f;
    const GgufModel model(native_shards);
    const GgufFile& g = model.meta();
    if (const MetaValue* a = g.get("general.architecture")) f.arch = a->s;
    const std::string p = f.arch + ".";
    f.n_embd = meta_u(g, p + "embedding_length");
    f.nextn = meta_u(g, p + "nextn_predict_layers");
    f.n_layers = meta_u(g, p + "block_count") - f.nextn;
    f.n_expert = meta_u(g, p + "expert_count");
    f.n_expert_used = meta_u(g, p + "expert_used_count");
    f.n_ff = meta_u(g, p + "expert_feed_forward_length");
    f.n_ff_shexp = meta_u(g, p + "expert_shared_feed_forward_length");
    f.head_count = meta_u(g, p + "attention.head_count");
    f.head_count_kv = meta_u(g, p + "attention.head_count_kv");
    f.key_length = meta_u(g, p + "attention.key_length");
    f.hc = meta_u(g, p + "hyper_connection.count");
    f.hc_lr = meta_u(g, p + "hyper_connection.low_rank");
    f.full_attention_interval = meta_u(g, p + "full_attention_interval");
    f.ssm_inner = meta_u(g, p + "ssm.inner_size");
    f.ssm_state = meta_u(g, p + "ssm.state_size");
    f.ssm_groups = meta_u(g, p + "ssm.group_count");
    f.indexer_heads = meta_u(g, p + "attention.indexer.head_count");
    f.indexer_key = meta_u(g, p + "attention.indexer.key_length");
    f.ple_dim = meta_u(g, p + "embedding_length_per_layer_input");
    f.ple_ngram = meta_u(g, p + "ple.ngram_size");
    f.ple_heads = meta_u(g, p + "ple.heads_per_ngram");
    if (const MetaValue* t = g.get("tokenizer.ggml.tokens")) f.n_vocab = (long long) t->count;
    std::set<std::string> pairs;
    for (size_t s = 0; s < model.size(); ++s) {
        for (const TensorInfo& t : model.shard(s).tensors()) {
            const std::string& n = t.name;
            if (n == "per_layer_token_embd.weight") f.ple_table = true;
            if (has_suffix(n, ".ssm_alpha.weight")) f.gdn = true;
            if (has_suffix(n, ".indexer.q_proj.weight")) f.qsa = true;
            // (the hyper-connection weights reach the read from the pack, which stores them in BF16 whatever the GGUF
            // holds - iq_pack converts Q8_0: so their GGUF type is not an assumption of the fork read)
            if (has_suffix(n, ".ffn_gate_exps.weight")) {
                const std::string base = n.substr(0, n.size() - std::strlen("gate_exps.weight"));
                const TensorInfo* d = model.find(base + "down_exps.weight");
                pairs.insert(std::string(t.type_name()) + "/" + (d ? d->type_name() : "?"));
            }
            // B.3's formats: the dense projections it reads (2-D, not experts, not the embedding)
            const bool il = t.type == 23 || t.type == 12 || t.type == 13 || t.type == 14 || t.type == 8;
            if (il && t.shape.size() == 2 && n.find("_exps") == std::string::npos && n != "token_embd.weight" &&
                n != "per_layer_token_embd.weight") {
                const uint64_t blk = t.type == 8 ? 32 : 256;
                if (t.shape[0] % blk != 0) f.bad_dense.push_back(n);
            }
        }
    }
    f.expert_types.assign(pairs.begin(), pairs.end());
    if (!mtp_dir.empty()) {
        std::ifstream in(mtp_dir + "/draft_vocab.bin", std::ios::binary);
        if (in) {
            std::vector<char> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            f.draft_vocab = (long long) (buf.size() / 4);
            for (size_t i = 0; i + 4 <= buf.size(); i += 4) {
                int32_t id;
                std::memcpy(&id, buf.data() + i, 4);
                if (id > f.draft_max_id) f.draft_max_id = id;
            }
        }
    }
    return f;
}

int run_patch_guards(const ModelFacts& f) {
    namespace K = strata::kernels;
    int gr_n = 0, gr_hc = 0, gr_lr = 0, attn_hd = 0, attn_g = 0;
    K::fused_gr_fork_geometry(gr_n, gr_hc, gr_lr);
    K::qsa_split_attn_geometry(attn_hd, attn_g);
    const std::vector<Guard> guards = {
        // (round 14: the pr245 and ple_prefetch guards went with their patches: v0.1.40's own kernels / PLE path)
        {"spec_exact", "exact speculative sampling (the drafter's q lists over its draft vocabulary)", "STRATA_SPEC_EXACT=0",
         [](const ModelFacts& m) -> std::string {
             if (m.draft_vocab < 0) return "";   // no drafter: nothing to sample from (the patch is inactive)
             if (m.draft_vocab == 0) return "the drafter's draft_vocab.bin is empty";
             if (m.draft_max_id >= m.n_vocab)
                 return "draft id " + str(m.draft_max_id) + " is outside the model's vocabulary of " + str(m.n_vocab);
             return "";
         },
         [] { setenv_off("STRATA_SPEC_EXACT", "0"); }},
        {"side_mixer", "GDN gates / QSA indexer chain (and with 2 the gate projection / query chain) on side branches",
         "STRATA_SIDE_MIXER=0",
         [](const ModelFacts& m) -> std::string {
             if (!m.gdn || !m.qsa || m.full_attention_interval != 4)
                 return "layer pattern GDN " + std::string(m.gdn ? "yes" : "no") + ", QSA+indexer " + (m.qsa ? "yes" : "no") +
                        ", full attention every " + str(m.full_attention_interval) + " (0.1.39's window: GDN x3 + QSA, every 4)";
             return "";
         },
         [] { setenv_off("STRATA_SIDE_MIXER", "0"); }},
        {"router_q8k", "A.2: the doorbell writes the pool's Q8_K rows", "STRATA_ROUTER_Q8K=0",
         [](const ModelFacts& m) -> std::string {
             if (m.n_embd % 256 != 0) return "n_embd " + str(m.n_embd) + " is not a multiple of a Q8_K superblock (256)";
             return "";
         },
         [] { setenv_off("STRATA_ROUTER_Q8K", "0"); }},
        {"qsa_split_attn", "B.2: the decode attention through the int8 tensor-core kernel, split", "STRATA_QSA_SPLIT_ATTN=0",
         [attn_hd, attn_g](const ModelFacts& m) -> std::string {
             const long long g = m.head_count_kv ? m.head_count / m.head_count_kv : 0;
             if (m.key_length != attn_hd || g != attn_g)
                 return "head size " + str(m.key_length) + ", " + str(g) + " query heads per KV head; compiled for " +
                        str(attn_hd) + " / " + str(attn_g);
             return "";
         },
         [] { setenv_off("STRATA_QSA_SPLIT_ATTN", "0"); }},
        {"gr_fork", "B.1: fork Eddoursul's hyper-connection read", "STRATA_GR_FORK=0",
         [gr_n, gr_hc, gr_lr](const ModelFacts& m) -> std::string {
             if (m.n_embd != gr_n || m.hc != gr_hc || m.hc_lr != gr_lr)
                 return "n_embd " + str(m.n_embd) + ", " + str(m.hc) + " streams, low rank " + str(m.hc_lr) +
                        "; compiled for " + str(gr_n) + " / " + str(gr_hc) + " / " + str(gr_lr);
             return "";
         },
         [] { setenv_off("STRATA_GR_FORK", "0"); }},
    };
    const bool list = std::getenv("STRATA_GUARD_LIST") != nullptr && std::atoi(std::getenv("STRATA_GUARD_LIST")) != 0;
    int off = 0;
    std::string names;
    for (const Guard& g : guards) {
        std::string why = g.check(f);
        if (why.empty() && forced(g.name)) why = "forced by STRATA_GUARD_FORCE (test)";
        if (!why.empty()) {
            g.off();
            ++off;
            std::printf("strata patch guard: %s OFF (%s; upstream's path): %s\n", g.name, g.off_how, why.c_str());
        } else if (list) {
            std::printf("strata patch guard: %s on - %s\n", g.name, g.patch);
        }
        names += (names.empty() ? "" : " ") + std::string(g.name) + (why.empty() ? "" : "(off)");
    }
    std::string types;
    for (const std::string& t : f.expert_types) types += (types.empty() ? "" : " ") + t;
    std::printf("strata patch guards: %zu checked, %d off [%s]; model %s %lld x %lld layers (+%lld nextn), %lld experts "
                "of %lld, vocab %lld, draft vocab %lld; experts %s\n",
                guards.size(), off, names.c_str(), f.arch.c_str(), f.n_embd, f.n_layers, f.nextn, f.n_expert, f.n_ff,
                f.n_vocab, f.draft_vocab, types.c_str());
    return off;
}

}  // namespace strata::core
