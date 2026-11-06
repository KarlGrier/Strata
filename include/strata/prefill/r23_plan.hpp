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

}  // namespace strata::prefill::r23
