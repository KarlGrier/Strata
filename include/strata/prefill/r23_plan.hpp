// include/strata/prefill/r23_plan.hpp - the rules of r23's prompt-path switches (prefill.cpp, all off by default) as
// pure functions, each with the reason it does not apply: prefill.cpp takes its numbers from here and prints the
// reason, tests/prefill/r23_plan_test.cpp checks them (no GPU, no model), and the gate's rules
// (work/r23/basalt_fix_rules.py) compute the same formulas.
//
// Why the reasons are printed (gap 11-05, results/r23/basalt-prompt/FIX-1105.md): the yardstick engine has one cache
// slot, so `--prefill auto:16384` there reads in 1024-token chunks (nothing lendable: generate.cpp), and the switches
// did nothing in the gate's prompt sets without a line saying so.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::prefill::r23 {

// ---- STRATA_PF_AHEAD=1 (adapted from Basalt 31aacda2, MIT) ----
// A staged chunk (below the stream-all size) copies the next layer's likely experts ahead into staging slots the
// layout already holds past the staged walk's own `stage_grp`: `layout_slots` (the layout's stage_slots()), never past
// `ring_cap`, split over the two layer parities, at most `max_per_parity` each.  A layout laid out for a chunk below the
// stream-all size holds only the staged walk's slots, so mode 1 copies nothing there (mode 2 adds Basalt's slots; that
// grows the loan).  The rest of a prompt longer than the chunk is read in the long chunk's stream-all layout: there it
// copies ahead.
struct AheadSlots {
    int per_parity = 0;          // 0: none
    const char* why = nullptr;   // set when per_parity == 0
};
inline AheadSlots ahead_slots(int layout_slots, int ring_cap, int stage_grp, int max_per_parity) {
    const int room = std::min(layout_slots, ring_cap);
    AheadSlots a;
    if (room <= stage_grp) {
        a.why = "the layout holds no staging slots past the staged walk's (it was laid out for a chunk below the stream-all size)";
        return a;
    }
    a.per_parity = std::min(max_per_parity, (room - stage_grp) / 2);
    if (a.per_parity <= 0) a.why = "the layout holds fewer than two staging slots past the staged walk's";
    return a;
}

// ---- STRATA_PF_TAIL_SPLIT (after Basalt's equal chunks, MIT) ----
// n tokens in chunks of C (the last the rest), stream-all from smin tokens, the switch's minimum rest tmin (> 0): when the
// last chunk is a rest of at least tmin tokens but below smin (a staged chunk), it and the chunk before it become two
// chunks of a = ceil((C + rest) / 2) rounded up to 256 (at most C) and C + rest - a, both at least smin, starting at
// `at`.  Every earlier chunk stays C.
struct TailSplit {
    bool on = false;
    bool staged_rest = false;    // the last chunk is below smin (what the switch is about)
    int64_t at = -1, a = 0, b = 0, last = 0;
    const char* why = nullptr;   // set when !on
};
inline TailSplit tail_split(int64_t n, int64_t C, int64_t smin, int64_t tmin) {
    TailSplit t;
    if (tmin <= 0) { t.why = "off"; return t; }
    if (C <= 0 || n <= C) { t.why = "the prompt is one chunk"; return t; }
    const int64_t k = (n + C - 1) / C, last = n - (k - 1) * C, r = C + last;
    const int64_t a = std::min<int64_t>(C, ((r + 1) / 2 + 255) / 256 * 256);
    t.last = last;
    t.a = a;
    t.b = r - a;
    t.at = (k - 2) * C;
    t.staged_rest = last < smin;
    if (!t.staged_rest) { t.why = "the last chunk streams every expert (not a staged rest)"; return t; }
    if (last < tmin) { t.why = "the rest is shorter than the switch's minimum"; return t; }
    if (a < smin || r - a < smin) { t.why = "the two chunks would be below the stream-all size (the chunk is too small)"; return t; }
    t.on = true;
    return t;
}

// ---- STRATA_PF_FIRST_CHUNK (after Basalt e8f3b84b, MIT) ----
// n tokens with a chunk C above the first chunk f0: the first f0 tokens as one chunk, the rest in k = ceil(rest / C)
// chunks of e = ceil(rest / k) rounded up to 256 (at most C; the last the rest), when that leaves no staged chunk
// (k > 1, or the rest at least smin).
struct FirstChunk {
    bool on = false;
    int64_t f0 = 0, k = 0, e = 0, rest = 0;
    const char* why = nullptr;   // set when !on
};
inline FirstChunk first_chunk(int64_t n, int64_t C, int64_t f0, int64_t smin) {
    FirstChunk f;
    f.f0 = f0;
    if (f0 <= 0) { f.why = "off"; return f; }
    if (C <= f0) { f.why = "the chunk is not above the first chunk"; return f; }
    if (n <= C) { f.why = "the prompt is one chunk"; return f; }
    f.rest = n - f0;
    f.k = (f.rest + C - 1) / C;
    f.e = std::min<int64_t>(C, ((f.rest + f.k - 1) / f.k + 255) / 256 * 256);
    if (!(f.k > 1 || f.rest >= smin)) { f.why = "the rest would be one staged chunk"; return f; }
    f.on = true;
    return f;
}

}  // namespace strata::prefill::r23
