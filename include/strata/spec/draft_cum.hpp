// include/strata/spec/draft_cum.hpp - a cumulative, per-request adaptive draft length (STRATA_DRAFT_CUM=1, off by
// default).
//
// Adapted from Basalt (MIT), include/basalt/core/draft_cum.hpp at fb0944c7: the acceptance table, the keep rule, the
// pack factor and the per-request threshold (Basalt commits 241b5b4c, be7147f3, 32e5522f, 47f6192f, 5ec431db,
// 7af4a2c6; their call sites in Basalt's src/program/generate.cpp and src/core/mtp.cpp).  Basalt's notice is at the
// end of this file.
//
// The rule.  With --spec-min-p a window stops at the first draft whose raw top probability is below min_p.  This rule
// keeps draft j while the estimated chance that drafts 0..j are ALL accepted stays at or above a threshold: the product
// over i <= j of acc(p_i), acc = Basalt's acceptance rate of a draft by its probability bin.  A middling first draft is
// cheap to verify and still often right, so it stays; a long run of merely high probabilities stops once the joint
// chance decays.  Each request starts at the base threshold and lowers it toward the floor while its drafts are
// accepted more often than the table predicts (decayed counts of accepted vs expected drafts, a linear ramp from a
// ratio of 0.9 to 1.1).  Both are scaled by the pack factor: 1.0 up to 1.1 MB of RAM-served expert bytes per expert
// slot, rising linearly to 1.5 at 3.5 MB (Basalt: each extra verified row makes the CPU read more distinct experts).
// Our IQ3_S pack (46.84 GiB of experts, 48 x 512 slots, all of it in RAM), estimated from the mean blob (2.05 MB):
// the 5080 + the 4060 Ti helper (4,585 + 3,300 slots on the GPUs) leave about 1.39 MB per slot = factor 1.06
// (threshold 0.371 -> 0.265); the 5080 alone (4,433 slots) about 1.68 MB = factor 1.12 (0.392 -> 0.280).  The engine
// computes it from the real placement and blob sizes once the tiers are final, and prints it at start.
//
// With the cost gate (STRATA_COST_GATE) the gate prices the draft steps its table covers (CostGate::steps(), 3 in
// configs/r12-cost-gate.txt) and its vetoes stand; this rule decides the steps past them, where the gate has no cost
// data - and every step when no gate is loaded.  Greedy and sampled windows alike; a lookup window's own length stays
// the draft policy's (as in Basalt: the policy compares the lookup with the MTP window this rule sized).
//
// Only what is proposed changes.  The length depends on the drafter's raw top probabilities (read before any draw)
// and on a threshold fixed before the window's drafts are made.  Greedy verification still emits the target's argmax
// (windows regroup, which can move rounding, as the cost gate's do: class (b)); exact speculative sampling
// (spec_exact.hpp) stays distribution-preserving (class (c)).  The drafter's chain
// (mtp.cpp) and the window size (generate.cpp) evaluate the same products in the same order, so the window never asks
// for a draft the chain did not make.
//
//     STRATA_DRAFT_CUM=1          on (0, empty or unset: off)
//     STRATA_DRAFT_CUM_BASE=V     the threshold a request starts from, before the pack factor (Basalt 0.35)
//     STRATA_DRAFT_CUM_FLOOR=V    the lowest it adapts to, before the pack factor (Basalt 0.25)
//     STRATA_DRAFT_CUM_RATE=V     the adaptation rate: the counts decay by 1 - V per window (Basalt 0.03: decay 0.97)
//     STRATA_DRAFT_CUM_SCALE=V    the pack factor itself, instead of the one computed from the placement
#pragma once

#include "strata/spec/cost_gate.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::spec {

/// Basalt's acceptance rate of a sampled draft by its raw probability bin (its measurement on its own machine).
inline float draft_cum_acc(float p) {
    return p < 0.30f ? 0.12f : p < 0.50f ? 0.26f : p < 0.70f ? 0.43f : p < 0.80f ? 0.51f
         : p < 0.90f ? 0.59f : p < 0.95f ? 0.66f : p < 0.99f ? 0.79f : 0.96f;
}

/// STRATA_DRAFT_CUM and its constants.
struct DraftCumConfig {
    bool on = false;
    float base = 0.35f;    ///< Basalt kDraftCumBase0
    float floor = 0.25f;   ///< Basalt kLo (before the pack factor)
    double decay = 0.97;   ///< Basalt kDecay = 1 - the adaptation rate
    float scale = -1.0f;   ///< the pack factor; < 0: computed from the placement (draft_cum_cpu_mb)
};

/// The config from the variables' values (null = unset); out-of-range values keep the default.
inline DraftCumConfig draft_cum_config_from(const char* on_s, const char* base_s, const char* floor_s, const char* rate_s,
                                            const char* scale_s) {
    DraftCumConfig c;
    c.on = on_s != nullptr && *on_s != '\0' && std::strcmp(on_s, "0") != 0;
    auto num = [](const char* s, double lo, double hi, double& out) {
        if (s == nullptr || *s == '\0') return;
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        if (end != s && v >= lo && v <= hi) out = v;
    };
    double b = c.base, f = c.floor, r = -1.0, k = c.scale;
    num(base_s, 1e-4, 1.0, b);
    num(floor_s, 0.0, 1.0, f);
    num(rate_s, 1e-4, 1.0, r);
    num(scale_s, 0.1, 10.0, k);
    c.base = (float) b;
    c.floor = (float) f;
    if (r > 0.0) c.decay = 1.0 - r;   // unset or out of range: Basalt's 0.97 exactly
    c.scale = (float) k;
    return c;
}

/// STRATA_DRAFT_CUM's config, read once (after the patch guards, which may turn it off).
inline const DraftCumConfig& draft_cum_config() {
    static const DraftCumConfig c = draft_cum_config_from(
        std::getenv("STRATA_DRAFT_CUM"), std::getenv("STRATA_DRAFT_CUM_BASE"), std::getenv("STRATA_DRAFT_CUM_FLOOR"),
        std::getenv("STRATA_DRAFT_CUM_RATE"), std::getenv("STRATA_DRAFT_CUM_SCALE"));
    return c;
}

/// The RAM-served expert bytes per expert slot, in MB (Basalt's draft_cum_cpu_mb), set once the tiers are final.
inline float& draft_cum_cpu_mb() {
    static float v = 0.0f;
    return v;
}

/// Basalt's pack factor for `cpu_mb`: 1.0 up to 1.1 MB, 1.5 from 3.5 MB, linear between.
inline float draft_cum_pack_scale(float cpu_mb) {
    constexpr float kLoMb = 1.1f, kHiMb = 3.5f, kMax = 1.5f;
    const float t = (cpu_mb - kLoMb) / (kHiMb - kLoMb);
    return 1.0f + (kMax - 1.0f) * (t < 0.0f ? 0.0f : t > 1.0f ? 1.0f : t);
}

/// The factor in force: STRATA_DRAFT_CUM_SCALE, else the placement's.
inline float draft_cum_scale() {
    const DraftCumConfig& c = draft_cum_config();
    return c.scale > 0.0f ? c.scale : draft_cum_pack_scale(draft_cum_cpu_mb());
}

/// One window's length rule, fixed before its drafts are made.
struct DraftCumWindow {
    bool on = false;                 ///< false: the cost gate / --spec-min-p decide, as without the switch
    float th = 0.0f;                 ///< the request's threshold for this window
    const CostGate* gate = nullptr;  ///< the cost gate when one is loaded: it prices its table's steps
};

/// The rule at threshold `th`, with `gate` (or none) pricing its table's steps.
inline DraftCumWindow draft_cum_window(float th, const CostGate* gate) {
    DraftCumWindow w;
    w.on = true;
    w.th = th;
    w.gate = gate != nullptr && gate->on && gate->steps() > 0 ? gate : nullptr;
    return w;
}

/// The rule at threshold `th` with STRATA_COST_GATE's table (what the serve and generate loops and the drafter use).
inline DraftCumWindow draft_cum_window(float th) { return draft_cum_window(th, &cost_gate()); }

/// Whether draft j stays in the window, given that drafts 0..j-1 stay: probs[0..j] are the drafts' raw top
/// probabilities.  Recomputed from probs[0] each time, in one fixed order, so the chain and the window agree exactly.
inline bool draft_cum_keep(const float* probs, int j, const DraftCumWindow& w) {
    for (int i = 0; i <= j; ++i)
        if (!(probs[i] > 0.0f)) return false;   // a step the drafter did not run (it writes 0), or a NaN
    if (w.gate != nullptr) {   // the steps the cost gate prices: its rule, its products (as cost_gate_T)
        const int G = w.gate->steps();
        double P = 1.0;
        for (int i = 0; i <= j && i < G; ++i) {
            P *= w.gate->a(i, probs[i]);
            if (P < w.gate->th(i)) return false;
        }
        if (j < G) return true;
    }
    float c = 1.0f;   // Basalt's draft_keep: the same products in the same order
    for (int i = 0; i <= j; ++i) c = c * draft_cum_acc(probs[i]);
    return c >= w.th;
}

/// The window size: 1 + the leading drafts that stay, at most S rows (probs[0 .. S-2] readable).
inline int draft_cum_T(const float* probs, int S, const DraftCumWindow& w) {
    int T = 1;
    while (T < S && draft_cum_keep(probs, T - 1, w)) ++T;
    return T;
}

/// A request's threshold (Basalt's DraftCumAdapt): it starts at the base and falls toward the floor as the request's
/// MTP drafts are accepted more often than the table predicts.  The observed and expected accepted counts decay per
/// window; their ratio sets the threshold along a linear ramp (0.9 -> base, 1.1 -> floor).
struct DraftCumAdapt {
    float base = 0.35f, lo = 0.25f, cur = 0.35f;
    double decay = 0.97, obs = 0.0, expect = 0.0;   ///< Basalt's obs / exp

    void start(const DraftCumConfig& c, float scale) {
        base = c.base * scale;
        lo = c.floor * scale;
        decay = c.decay;
        obs = expect = 0.0;
        cur = base;
    }
    float th() const { return cur; }

    /// One verified MTP window: `n` drafts were offered with raw probabilities probs[0..n), `acc` of them accepted.
    void observe(const float* probs, int n, int acc) {
        if (n <= 0) return;
        constexpr double kRlo = 0.9, kRhi = 1.1;
        double c = 1.0, e = 0.0;
        for (int j = 0; j < n; ++j) {
            c *= (double) draft_cum_acc(probs[j]);
            e += c;
        }
        obs = obs * decay + (double) acc;
        expect = expect * decay + e;
        if (expect < 0.5) return;
        const double t = std::fmin(std::fmax((obs / expect - kRlo) / (kRhi - kRlo), 0.0), 1.0);
        cur = base + (float) t * (std::fmin(lo, base) - base);
    }
};

}  // namespace strata::spec

// Basalt's notice (the code above adapts parts of include/basalt/core/draft_cum.hpp):
//
// MIT License
//
// Copyright (c) 2026 Jesús Gulfo
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
