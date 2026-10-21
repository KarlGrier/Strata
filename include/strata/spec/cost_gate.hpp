// include/strata/spec/cost_gate.hpp - round 12: a COST-AWARE draft length (STRATA_COST_GATE=FILE, off by default).
//
// Today a window drafts while the draft head's raw top probability stays >= --spec-min-p (T = 1; while T < S &&
// dprob[T-1] >= floor: ++T).  With this switch, draft d (0-based) joins the window while the PREDICTED probability that
// the window reaches and accepts it - the product over d' <= d of alpha(d', dprob[d']) - is at least theta(d): the tokens
// it is expected to add pay for the time it adds.  alpha(d, p) = the mean exact-sampling acceptance of a draft at step d
// whose raw top probability falls in p's bin (FILE: measured on recorded research traffic), theta(d) = the decode rate
// times the extra time of a window with one more row (W(d+2) - W(d+1), measured).
// The length still depends only on the drafter's raw probabilities, read before any draw, so exact speculative
// sampling stays distribution-preserving (spec_exact.hpp).  The gate sizes every MTP window, greedy ones too (it
// regroups them: patches/LEDGER.md row 26); lookup windows are unchanged.
// The table prices the steps its theta list covers (steps(): 3 in configs/r12-cost-gate.txt, fitted at --spec 4).  Past
// them the gate has no cost data, and the window goes on by the --spec-min-p rule, as it would without the gate.  (Until
// 2026-10-21 the gate stopped there - theta 1e9 - which capped every window at steps() drafts above --spec 4: at
// --spec 7 our greedy struct run drafted 3.00 per window where stock v0.2.1 drafts 5.91.)  At --spec <= steps() + 1
// nothing changed.
// FILE format (text): "bins N" (edges i/N, i = 0..N; bin = the number of edges <= p, numpy's digitize), "theta t0 t1 t2 ..."
// then lines "alpha d bin value"; a missing bin uses "default d value".
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace strata::spec {

struct CostGate {
    bool on = false;
    int bins = 20;
    std::vector<double> theta;
    std::map<std::pair<int, int>, double> alpha;
    std::map<int, double> def;

    double a(int d, float p) const {
        int b = 0;
        for (int i = 0; i <= bins; ++i)
            if ((double) p >= (double) i / bins) b = i + 1;
        auto it = alpha.find({d, b});
        if (it != alpha.end()) return it->second;
        auto jt = def.find(d);
        return jt != def.end() ? jt->second : 0.0;
    }
    double th(int d) const { return d < (int) theta.size() ? theta[(size_t) d] : 1e9; }
    /// The draft steps the table prices (its theta list); the steps past them go by --spec-min-p.
    int steps() const { return (int) theta.size(); }
};

/// STRATA_COST_GATE=FILE, read once.
inline const CostGate& cost_gate() {
    static const CostGate g = [] {
        CostGate c;
        const char* p = std::getenv("STRATA_COST_GATE");
        if (p == nullptr || !*p) return c;
        std::FILE* f = std::fopen(p, "r");
        if (f == nullptr) { std::fprintf(stderr, "strata: STRATA_COST_GATE: cannot read %s (off)\n", p); return c; }
        char key[32];
        while (std::fscanf(f, "%31s", key) == 1) {
            if (!std::strcmp(key, "bins")) { if (std::fscanf(f, "%d", &c.bins) != 1) break; }
            else if (!std::strcmp(key, "theta")) {
                double v;
                int n = 0;
                if (std::fscanf(f, "%d", &n) != 1) break;
                for (int i = 0; i < n && std::fscanf(f, "%lf", &v) == 1; ++i) c.theta.push_back(v);
            } else if (!std::strcmp(key, "alpha")) {
                int d, b;
                double v;
                if (std::fscanf(f, "%d %d %lf", &d, &b, &v) != 3) break;
                c.alpha[{d, b}] = v;
            } else if (!std::strcmp(key, "default")) {
                int d;
                double v;
                if (std::fscanf(f, "%d %lf", &d, &v) != 2) break;
                c.def[d] = v;
            }
        }
        std::fclose(f);
        c.on = !c.theta.empty() && !c.alpha.empty();
        std::fprintf(stderr, "strata: cost-aware draft length %s (%s: %zu bins, theta", c.on ? "ON" : "off", p, c.alpha.size());
        for (double t : c.theta) std::fprintf(stderr, " %.3f", t);
        std::fprintf(stderr, ")\n");
        return c;
    }();
    return g;
}

/// The window size the gate `g` gives for drafts with raw top probabilities dprob[0 .. S-2] (S = the most rows): the
/// steps its table prices by its rule, the steps past them by --spec-min-p (`min_p`).
inline int cost_gate_T(const CostGate& g, const float* dprob, int S, float min_p) {
    int T = 1;
    double P = 1.0;
    while (T < S) {
        if (T - 1 >= g.steps()) {   // past the table: the --spec-min-p rule (a step the drafter did not run has 0)
            if (!(dprob[T - 1] >= min_p)) break;
            ++T;
            continue;
        }
        P *= g.a(T - 1, dprob[T - 1]);
        if (P < g.th(T - 1)) break;
        ++T;
    }
    return T;
}
/// The same with STRATA_COST_GATE's table.
inline int cost_gate_T(const float* dprob, int S, float min_p) { return cost_gate_T(cost_gate(), dprob, S, min_p); }

}  // namespace strata::spec
