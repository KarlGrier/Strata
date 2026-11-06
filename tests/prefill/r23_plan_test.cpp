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

    // ---- STRATA_PF_TAIL_SPLIT=1 (rests of 256 tokens or more) ----
    // the served chunk: Basalt's pp16k fixture (16,669 read in a batch: 16384 + 285) -> 8448 + 8221, as the gate's profile
    // printed; the tail24 cut 16684 (the yardstick reads 16683: the last token goes through the first window) -> 8448 + 8235
    {
        const TailSplit t = tail_split(16669, 16384, 1792, 256);
        CHECK(t.on && t.at == 0 && t.a == 8448 && t.b == 8221 && t.last == 285 && t.why == nullptr);
        const TailSplit u = tail_split(16683, 16384, 1024, 256);
        CHECK(u.on && u.at == 0 && u.a == 8448 && u.b == 8235);
        const TailSplit w = tail_split(66609, 16384, 1792, 256);   // pp64k: 4 x 16384 + 1073 -> 3 x 16384 + 8960 + 8497
        CHECK(w.on && w.at == 3 * 16384 && w.a == 8960 && w.b == 8497);
        const TailSplit x = tail_split(33416, 16384, 1792, 256);   // the screen's R4 (33,417): 16384 + 8704 + 8328
        CHECK(x.on && x.at == 16384 && x.a == 8704 && x.b == 8328);
    }
    // tail24's cuts 16640 and 49408 leave a 255-token rest once the last token is taken off: below 256, left as it is
    {
        const TailSplit t = tail_split(16639, 16384, 1024, 256);
        CHECK(!t.on && t.staged_rest && t.last == 255 && std::strstr(t.why, "shorter than the switch's minimum") != nullptr);
        CHECK(tail_split(16639, 16384, 1024, 255).on);             // STRATA_PF_TAIL_SPLIT=255 takes it
    }
    // the yardstick's 1024-token chunk (gap 11-05): 1024 + rest is under 2 x 1024, so two stream-all halves never exist
    {
        const TailSplit t = tail_split(16683, 1024, 1024, 256);
        CHECK(!t.on && t.staged_rest && t.last == 299 && t.a == 768 && t.b == 555);
        CHECK(std::strstr(t.why, "below the stream-all size") != nullptr);
        for (int64_t n = 1025; n < 70000; n += 97) CHECK(!tail_split(n, 1024, 1024, 256).on);
    }
    // a last chunk that streams every expert, a one-chunk prompt, the switch off
    CHECK(!tail_split(16384 + 1792, 16384, 1792, 256).on && !tail_split(16384 + 1792, 16384, 1792, 256).staged_rest);
    CHECK(tail_split(16384 + 1791, 16384, 1792, 256).on);
    CHECK(!tail_split(16384, 16384, 1792, 256).on && !tail_split(16384, 16384, 1792, 256).staged_rest);
    CHECK(!tail_split(16683, 16384, 1792, 0).on);
    // whenever it applies: the chunks still cover n, both halves stream, none is above C
    for (int64_t C : {8192, 16384, 21504}) {
        for (int64_t n = C + 1; n < 4 * C; n += 37) {
            const TailSplit t = tail_split(n, C, 1792, 256);
            if (!t.on) continue;
            CHECK(t.at + t.a + t.b == n && t.a <= C && t.b <= C && t.a >= 1792 && t.b >= 1792 && t.at % C == 0);
        }
    }

    // ---- STRATA_PF_FIRST_CHUNK=1 (a first chunk of 16384) ----
    // the served chunk 16384, and the gate's cap arm at --prefill auto:24576 (its ceiling was 16384): nothing, said why
    {
        const FirstChunk f = first_chunk(66609, 16384, 16384, 1792);
        CHECK(!f.on && std::strstr(f.why, "not above the first chunk") != nullptr);
        CHECK(!first_chunk(66609, 8192, 16384, 1792).on);
        CHECK(!first_chunk(66609, 1024, 16384, 1024).on);   // the yardstick's 1024 chunk
    }
    // a chunk above it (the commit's example, 66,617 at C 20480 -> 16384 + 3 x 16,896 + the rest; 66,616 read in a batch)
    {
        const FirstChunk f = first_chunk(66616, 20480, 16384, 1792);
        CHECK(f.on && f.f0 == 16384 && f.rest == 50232 && f.k == 3 && f.e == 16896 && f.why == nullptr);
        const FirstChunk g = first_chunk(33187, 21504, 16384, 1024);   // a tail24 cut at a 21504 chunk: 16384 + 16803
        CHECK(g.on && g.k == 1 && g.e == 16896 && g.rest == 16803);
        const FirstChunk h = first_chunk(17000, 16640, 16384, 1792);   // the rest would be one staged chunk (616 < 1792)
        CHECK(!h.on && h.k == 1 && std::strstr(h.why, "staged") != nullptr);
        CHECK(first_chunk(17000, 16640, 16384, 512).on);                // ... not below a 512 stream-all size
        CHECK(!first_chunk(20000, 21504, 16384, 1024).on);              // one chunk
        CHECK(!first_chunk(66616, 20480, 0, 1792).on);                  // off
    }
    // whenever it applies: the chunks cover n, none is above C, only the last may be shorter than e
    for (int64_t C : {16640, 20480, 21504, 24576, 32768}) {
        for (int64_t n = C + 1; n < 140000; n += 211) {
            const FirstChunk f = first_chunk(n, C, 16384, 1792);
            if (!f.on) continue;
            CHECK(f.e <= C && f.e % 256 == 0 && (f.k - 1) * f.e < f.rest && f.k * f.e >= f.rest);
            CHECK(f.k > 1 || f.rest >= 1792);
        }
    }

    if (failed) {
        std::fprintf(stderr, "r23_plan_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("r23_plan_test: all checks passed\n");
    return 0;
}
