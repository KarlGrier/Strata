// src/spec/draft_cum_test.cpp - the cumulative draft length's arithmetic (strata/spec/draft_cum.hpp, STRATA_DRAFT_CUM):
// Basalt's table and keep rule, the cost gate pricing its table's steps, the window never past the chain's drafts, the
// per-request threshold, the pack factor and the variables.  Header-only: no model, no GPU.
#include "strata/spec/draft_cum.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>

using namespace strata::spec;

namespace {

bool close_to(double a, double b, double tol = 1e-4) { return std::fabs(a - b) <= tol; }

// configs/r12-cost-gate.txt's shape (as cost_gate_test): 20 bins, 3 steps
CostGate table(double th = -1.0) {
    CostGate g;
    g.on = true;
    g.bins = 20;
    g.theta = {0.32295, 0.30841, 0.39299};
    if (th > 0.0) g.theta = {th, th, th};
    for (int d = 0; d < 3; ++d) {
        for (int b = 2; b <= 21; ++b) g.alpha[{d, b}] = 0.2 + 0.8 * (b - 2) / 19.0 - 0.05 * d;
        g.def[d] = 0.74 - 0.05 * d;
    }
    return g;
}

}  // namespace

int main() {
    int fail = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) { std::printf("FAIL: %s\n", what); ++fail; }
    };

    // Basalt's bins
    check(draft_cum_acc(0.29f) == 0.12f && draft_cum_acc(0.30f) == 0.26f && draft_cum_acc(0.50f) == 0.43f &&
              draft_cum_acc(0.70f) == 0.51f && draft_cum_acc(0.80f) == 0.59f && draft_cum_acc(0.90f) == 0.66f &&
              draft_cum_acc(0.95f) == 0.79f && draft_cum_acc(0.99f) == 0.96f && draft_cum_acc(1.0f) == 0.96f,
          "the acceptance table's bins");

    // the rule alone (no cost gate)
    {
        const DraftCumWindow w = draft_cum_window(0.35f, nullptr);
        const float a[7] = {0.99f, 0.99f, 0.95f, 0.6f, 0.99f, 0.99f, 0.0f};   // 0.96, 0.92, 0.73, 0.31 < 0.35
        check(draft_cum_T(a, 7, w) == 4, "the joint chance falls below the threshold at the 4th draft");
        const float b[7] = {0.6f, 0.99f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        check(draft_cum_keep(b, 0, w), "a middling first draft (0.43) stays");
        const float c[7] = {0.45f, 0.99f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        check(draft_cum_T(c, 7, w) == 1, "a first draft of 0.26 goes");
        const float d[7] = {0.97f, 0.97f, 0.97f, 0.97f, 0.97f, 0.97f, 0.0f};   // 0.79^k: 0.39 at 4, 0.31 at 5
        check(draft_cum_T(d, 7, w) == 5, "a long run of merely high drafts stops at threshold 0.35");
        check(draft_cum_T(d, 7, draft_cum_window(0.25f, nullptr)) == 6, "... and one later at the floor 0.25");
        const float e[7] = {0.995f, 0.995f, 0.995f, 0.995f, 0.995f, 0.995f, 0.0f};
        check(draft_cum_T(e, 7, w) == 7, "confident drafts fill --spec 7");
        check(draft_cum_T(e, 4, w) == 4, "and never more than S rows");
        const float z[7] = {0.99f, 0.99f, 0.0f, 0.99f, 0.99f, 0.99f, 0.0f};
        check(draft_cum_T(z, 7, draft_cum_window(0.0f, nullptr)) == 3, "a step the drafter did not run never stays");
        const float n[7] = {0.99f, std::nanf(""), 0.99f, 0.99f, 0.99f, 0.99f, 0.0f};
        check(draft_cum_T(n, 7, w) == 2, "a NaN probability never stays");
    }

    // with the cost gate: it decides the steps its table prices, the rule the steps past them
    {
        const CostGate g = table();
        const DraftCumWindow w = draft_cum_window(0.35f, &g);
        const float a[7] = {0.99f, 0.99f, 0.99f, 0.6f, 0.6f, 0.6f, 0.0f};   // rule: 0.96^3 x 0.43 = 0.380, then 0.163
        check(draft_cum_T(a, 7, w) == 5, "past the table the rule's product over all drafts decides");
        const CostGate strict = table(0.9);
        const float b[7] = {0.97f, 0.97f, 0.97f, 0.97f, 0.97f, 0.0f, 0.0f};
        check(draft_cum_T(b, 7, draft_cum_window(0.35f, &strict)) == 2, "a veto inside the table stands");
        check(draft_cum_T(b, 7, draft_cum_window(0.35f, nullptr)) == 5, "(the rule alone keeps 4 there)");
        const float c[7] = {0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        check(draft_cum_T(c, 7, w) == 2 && draft_cum_T(c, 7, draft_cum_window(0.35f, nullptr)) == 1,
              "inside the table the gate keeps a draft the rule alone would drop");
        CostGate off = table();
        off.on = false;
        check(draft_cum_window(0.35f, &off).gate == nullptr, "a gate that is off prices nothing");

        // --spec 4 (3 drafts, all inside the table): the rule changes nothing beside the gate; any S: the window never
        // asks for a draft the chain did not make (the chain in mtp.cpp runs step j while the window keeps j - 1)
        uint64_t s = 0x2545f4914f6cdd1dull;
        auto rnd = [&]() {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            return (float) ((s >> 40) & 0xffffff) / 16777216.0f;
        };
        int same4 = 0, bad = 0, n_cases = 0;
        for (int it = 0; it < 50000; ++it) {
            float q[8];
            for (float& x : q) x = it % 3 == 0 ? 0.9f + 0.1f * rnd() : rnd();
            const float th = 0.25f * rnd() + (it % 5 == 0 ? 0.0f : 0.1f);
            same4 += draft_cum_T(q, 4, draft_cum_window(th, &g)) == cost_gate_T(g, q, 4, 0.5f);
            for (const CostGate* gp : {(const CostGate*) nullptr, &g}) {
                const DraftCumWindow w2 = draft_cum_window(th, gp);
                for (int S = 2; S <= 8; ++S) {
                    const int max_steps = S - 1;
                    float cpr[8] = {}, dprob[8] = {};
                    cpr[0] = dprob[0] = q[0];
                    int nd = 1;
                    for (int j = 1; j < max_steps && draft_cum_keep(cpr, j - 1, w2); ++j) {
                        cpr[j] = dprob[j] = q[j];
                        ++nd;
                    }
                    const int T = draft_cum_T(dprob, S, w2);
                    bad += T - 1 > nd || T > S;
                    ++n_cases;
                }
            }
        }
        check(same4 == 50000, "--spec 4 with the gate: the same size as the gate's");
        check(bad == 0 && n_cases == 50000 * 2 * 7, "the window never asks for a draft the chain did not make");
    }

    // the per-request threshold
    {
        DraftCumConfig c;
        DraftCumAdapt a;
        a.start(c, 1.0f);
        check(close_to(a.th(), 0.35), "a request starts at the base");
        const float hi[3] = {0.99f, 0.99f, 0.99f};   // expected 0.96 + 0.9216 + 0.8847 = 2.766
        a.observe(hi, 3, 3);                          // 3 / 2.766 = 1.085 -> t 0.923
        check(close_to(a.th(), 0.35 - 0.1 * (3.0 / (0.96 + 0.96 * 0.96 + 0.96 * 0.96 * 0.96) - 0.9) / 0.2, 1e-5),
              "drafts beating the table lower the threshold along the ramp");
        a.observe(hi, 3, 0);
        a.observe(hi, 3, 0);
        check(a.th() > 0.33f, "rejected drafts raise it back toward the base");
        DraftCumAdapt b;
        b.start(c, 1.0f);
        const float mid[3] = {0.6f, 0.6f, 0.6f};      // expected 0.694: 3 accepted is far above it
        b.observe(mid, 3, 3);
        check(close_to(b.th(), 0.25), "far above the table: the floor");
        DraftCumAdapt d;
        d.start(c, 1.0f);
        const float lo[1] = {0.2f};
        d.observe(lo, 1, 1);                          // expected 0.12 < 0.5: too little to judge
        check(close_to(d.th(), 0.35), "too few expected drafts: unchanged");
        d.observe(lo, 0, 0);
        check(close_to(d.th(), 0.35), "an empty window: unchanged");
        DraftCumAdapt k;
        k.start(c, 1.06f);
        k.observe(mid, 3, 3);
        check(close_to(k.base, 0.371) && close_to(k.th(), 0.265), "the pack factor scales base and floor");
        DraftCumConfig f = c;
        f.floor = 0.5f;   // a floor above the base: no adaptation
        DraftCumAdapt m;
        m.start(f, 1.0f);
        m.observe(mid, 3, 3);
        check(close_to(m.th(), 0.35), "a floor above the base keeps the base");
    }

    // Basalt's pack factor: 1.0 to 1.1 MB, 1.5 from 3.5 MB; our pack: 1.39 MB (two cards), 1.68 MB (the 5080 alone)
    check(close_to(draft_cum_pack_scale(0.0f), 1.0) && close_to(draft_cum_pack_scale(1.1f), 1.0) &&
              close_to(draft_cum_pack_scale(3.5f), 1.5) && close_to(draft_cum_pack_scale(9.0f), 1.5),
          "the pack factor's range");
    check(close_to(draft_cum_pack_scale(1.393f), 1.061, 1e-3) && close_to(draft_cum_pack_scale(1.677f), 1.120, 1e-3),
          "the pack factor for our IQ3_S placements");

    // the variables
    {
        const DraftCumConfig d = draft_cum_config_from(nullptr, nullptr, nullptr, nullptr, nullptr);
        check(!d.on && d.base == 0.35f && d.floor == 0.25f && d.decay == 0.97 && d.scale < 0.0f, "unset: off, Basalt's");
        check(!draft_cum_config_from("0", nullptr, nullptr, nullptr, nullptr).on, "0: off");
        check(!draft_cum_config_from("", nullptr, nullptr, nullptr, nullptr).on, "empty: off");
        check(draft_cum_config_from("1", nullptr, nullptr, nullptr, nullptr).on, "1: on");
        const DraftCumConfig v = draft_cum_config_from("1", "0.4", "0.2", "0.05", "1.25");
        check(close_to(v.base, 0.4) && close_to(v.floor, 0.2) && close_to(v.decay, 0.95, 1e-12) && close_to(v.scale, 1.25),
              "the overrides");
        const DraftCumConfig x = draft_cum_config_from("1", "abc", "-1", "0", "100");
        check(x.base == 0.35f && x.floor == 0.25f && x.decay == 0.97 && x.scale < 0.0f, "bad values keep the defaults");
    }

    std::printf("draft_cum tests: %s\n", fail ? "FAILED" : "OK");
    return fail ? 1 : 0;
}
