// src/spec/cost_gate_test.cpp - the cost gate's window size (strata/spec/cost_gate.hpp, STRATA_COST_GATE): within the
// steps its table prices it is the rule it always was; past them the window goes on by --spec-min-p instead of
// stopping (which capped every gated window at 3 drafts above --spec 4).  A synthetic table: no model, no GPU.
#include "strata/spec/cost_gate.hpp"

#include <cstdint>
#include <cstdio>

using strata::spec::CostGate;

namespace {

// the window size before 2026-10-21: the table's rule, and no draft past its steps (theta 1e9)
int old_T(const CostGate& g, const float* dprob, int S) {
    int T = 1;
    double P = 1.0;
    while (T < S) {
        P *= g.a(T - 1, dprob[T - 1]);
        if (P < g.th(T - 1)) break;
        ++T;
    }
    return T;
}

// configs/r12-cost-gate.txt's shape: 20 bins, 3 steps, acceptance rising with p
CostGate table() {
    CostGate g;
    g.on = true;
    g.bins = 20;
    g.theta = {0.32295, 0.30841, 0.39299};
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
    const CostGate g = table();
    check(g.steps() == 3, "the table prices 3 steps");

    // --spec 4 (S = 4: at most 3 drafts, all inside the table): the same size as before, whatever the probabilities
    uint64_t s = 0x9e3779b97f4a7c15ull;
    auto rnd = [&]() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (float) ((s >> 40) & 0xffffff) / 16777216.0f;
    };
    int same4 = 0;
    for (int it = 0; it < 100000; ++it) {
        float p[8];
        for (float& x : p) x = rnd();
        if (it % 7 == 0) p[it % 3] = 0.0f;   // a step the drafter did not run
        for (const float mp : {0.3f, 0.5f, 0.7f}) same4 += strata::spec::cost_gate_T(g, p, 4, mp) == old_T(g, p, 4);
    }
    check(same4 == 300000, "--spec 4: the window size is unchanged");

    // --spec 7: confident drafts go past the table's 3 steps as far as --spec-min-p allows
    {
        const float p[7] = {0.99f, 0.99f, 0.99f, 0.99f, 0.99f, 0.99f, 0.0f};
        check(old_T(g, p, 7) == 4, "before: 3 drafts at most (the cap)");
        check(strata::spec::cost_gate_T(g, p, 7, 0.5f) == 7, "--spec 7: 6 confident drafts all stay");
    }
    {
        const float p[7] = {0.99f, 0.99f, 0.99f, 0.97f, 0.40f, 0.99f, 0.0f};
        check(strata::spec::cost_gate_T(g, p, 7, 0.5f) == 5, "past the table: the first draft below min_p ends it");
        check(strata::spec::cost_gate_T(g, p, 7, 0.3f) == 7, "past the table: min_p 0.3 keeps the 0.40 draft");
    }
    {
        const float p[7] = {0.99f, 0.99f, 0.99f, 0.99f, 0.0f, 0.0f, 0.0f};   // the drafter made 4 drafts
        check(strata::spec::cost_gate_T(g, p, 7, 0.5f) == 5, "never past the drafts the chain made");
    }
    {
        const float p[7] = {0.99f, 0.06f, 0.99f, 0.99f, 0.99f, 0.99f, 0.0f};   // the table stops it at step 1
        check(strata::spec::cost_gate_T(g, p, 7, 0.01f) == 2, "a veto inside the table stands");
    }
    std::printf("cost_gate tests: %s\n", fail ? "FAILED" : "OK");
    return fail ? 1 : 0;
}
