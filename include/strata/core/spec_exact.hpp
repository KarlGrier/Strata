// include/strata/core/spec_exact.hpp - EXACT SPECULATIVE SAMPLING (STRATA_SPEC_EXACT=1, off by default).
//
// Today a sampled request verifies by SAMPLE-AND-MATCH: row t of a window at pos0 samples the target's chain with
// Philox(seed, pos0 + t), and a draft is kept only when it EQUALS that pick.  The output is then a function of the seed
// alone; a draft x is accepted with probability p(x) (p = the target's final distribution of that row).
//
// With this switch a sampled request runs the speculative sampling of Leviathan et al. / Chen et al. (2023):
//   * the drafter SAMPLES draft j from q_j, its own chain's final distribution (the request's penalties / top_k /
//     top_p / min_p / temperature over the draft head's logits), with a PROPOSAL stream of its own, and hands q_j to the
//     verify head as an exact integer list (counts out of 2^24: the distribution its 24-bit draw really has);
//   * the verify head accepts x_j with probability min(1, p_j(x_j) / q_j(x_j)) (ACCEPTANCE stream); a rejected row
//     draws from normalize((p_j - q_j)+) over the target's list (RESIDUAL stream) - tokens the draft head cannot
//     propose (outside its vocabulary subset) keep their whole mass there; a row with no draft (the bonus) draws
//     exactly as before (the request's stream, the same uniform);
//   * a prompt-lookup (suffix) draft is a point mass, q = 1 at the draft: accepted with probability p(x), else the
//     residual = p without x - the same distribution sample-and-match gives;
//   * WHETHER to draft position j is decided BEFORE x_j is drawn: the draft length gates on the draft head's raw top
//     probability (today's `--spec-min-p` rule, computed from the logits before the draw), and the lookup-vs-MTP choice
//     compares the lookup's first token with the draft head's raw argmax, never with the sampled draft.
// Accepted rows emit the draft and a rejected row's residual draw is never the draft, so the host's accept rule
// (window[t + 1] == out[t]) is unchanged.  The emitted text is distributed exactly as the target's sampler (each row's
// p), but it is no longer the same function of the seed as sample-and-match: same distribution, other draws.
// Lossless class (c): distribution-preserving.  Tests: work/r9/ess_test.cu (tiny vocabularies, chi-square).
#pragma once

#include <cstdlib>
#include <cstring>

namespace strata::core {

/// STRATA_SPEC_EXACT=1 (any value but empty or "0"): read once.
inline bool spec_exact_env() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_SPEC_EXACT");
        return e != nullptr && *e != '\0' && std::strcmp(e, "0") != 0;
    }();
    return on;
}

}  // namespace strata::core
