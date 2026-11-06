// r23's prompt-path switches' rules (include/strata/prefill/r23_plan.hpp): no GPU, no model.  The numbers are this
// PC's (Qwen3.8-Flash-Next IQ3_S native pack: ring_cap 512, STAGE_GRP 32, STRATA_PF_AHEAD_MAX 112; the served chunk
// 16384, the stream-all size 1024 without the CPU share and 1792 with it) and the gate's of 11-05, whose yardstick
// engine laid out 1024-token chunks.
#include <cstdio>
#include <cstring>

#include "strata/prefill/r23_plan.hpp"

using namespace strata::prefill::r23;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // ---- STRATA_PF_AHEAD=1: the slots per layer parity ----
    // a stream-all layout (the served 16384 chunk, or the yardstick's 1024 chunk without the share: 1024 >= 1024): its
    // 512-slot ring holds 240 per parity past the staged walk's 32, capped at 112
    CHECK(ahead_slots(512, 512, 32, 112).per_parity == 112);
    CHECK(ahead_slots(512, 512, 32, 112).why == nullptr);
    CHECK(ahead_slots(512, 512, 32, 1000).per_parity == 240);
    CHECK(ahead_slots(1024, 512, 32, 1000).per_parity == 240);   // never past ring_cap
    CHECK(ahead_slots(96, 512, 32, 112).per_parity == 32);       // a small ring: (96 - 32) / 2
    // a staged layout (the yardstick's 1024 chunk WITH the served share: stream-all from 1792, so the layout holds only
    // stage_slots = STAGE_GRP = 32): nothing - the gate's A16s
    CHECK(ahead_slots(32, 512, 32, 112).per_parity == 0);
    CHECK(ahead_slots(32, 512, 32, 112).why != nullptr);
    CHECK(std::strstr(ahead_slots(32, 512, 32, 112).why, "below the stream-all size") != nullptr);
    CHECK(ahead_slots(8, 512, 32, 112).per_parity == 0);
    CHECK(ahead_slots(33, 512, 32, 112).per_parity == 0);        // one slot past: not two parities' worth
    CHECK(ahead_slots(33, 512, 32, 112).why != nullptr);
    CHECK(ahead_slots(34, 512, 32, 112).per_parity == 1);

    if (failed) {
        std::fprintf(stderr, "r23_plan_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("r23_plan_test: all checks passed\n");
    return 0;
}
