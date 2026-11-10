# Branches of this fork

This fork holds the local patch series that run Strata on one PC (an RTX 5080 16 GB with an RTX 4060 Ti 8 GB helper card, 96 GB RAM). `main` follows upstream [Niko1221/Strata](https://github.com/Niko1221/Strata) and adds only this file.

## Served now

| Model | Tag | Branch | Engine sha256 (first 12) |
|---|---|---|---|
| Qwen3.8-Flash-Next IQ3_S (port 8080) | `served-2026-11-09-afetch2` | `r23-adaptfetch` | 153ce47d3ef0 |
| DeepSeek-V4.1-Flash (the DeepSeek daily) | `served-2026-10-27-ds41-daily` | `ds41-v026-daily` | 575de3a8df3f |

The Qwen build runs with `STRATA_ADAPT_FETCH=2` and `STRATA_HC_UPMIX=1`. Its server is this branch's `serve/`.

## Branches

| Branch | Base | Our commits | What it is |
|---|---|---|---|
| `r23-adaptfetch` | v0.2.8 | 31 | **Served.** `r22-port028x` plus the PCIe share's fetch admission (upstream 24c6c2f68, PR 2964 by sergqwer, cherry-picked) and our two-GPU follow-up. |
| `r22-port028x` | v0.2.8 | 29 | P028x: our patch series on v0.2.8. Served 10-30 to 11-09; the last commit is the model-mismatch guard (11-09). |
| `r23-port0213` | v0.2.13 | 36 | The v0.2.13 port, with the fetch patch. Passed the switch checks on 11-10; the baseline gate is next. Not served. |
| `r23-port0211` | v0.2.11 | 29 | The v0.2.11 port. Its baseline check passed under the 11-07 amended rule. The base the v0.2.13 port carries over. Not served. |
| `r23-basalt-prompt` | v0.2.10 | 34 | Basalt's prompt-path pieces ported onto v0.2.10 (MIT; the notices are kept in the files). Measured, not served. |
| `ds41-v026-daily` | v0.2.6 | 3 | **Served** as the DeepSeek daily. v0.2.6 plus the model-mismatch guard (server), a port of Project Maya PR #343 (prompt skip on a layer split's last group) and a slot RAM cap (`STRATA_DS4_SLOT_RAM_GB`). |
| `r22-port021-cont2` | v0.2.1 | 30 | P021m. Served 10-23 to 10-30; the rollback build. |
| `r22-port021-next` | v0.2.1 | 29 | P021n. Served 10-22 to 10-23. |
| `r22-port021` | v0.2.1 | 26 | P021. Served 10-20 to 10-22. |
| `r21-think-fence` | v0.1.44.1 | 32 | C*18t's server tree: C*18 plus the think-tag fix and the web-search fallback. Served 10-16 to 10-20. |
| `r20-trial-v04441` | v0.1.44.1 | 29 | C*18: the engine C*18t ran. |

## Tags: every build that was served

| Tag | Served | Engine sha256 (first 12) | Built from (old local hash) |
|---|---|---|---|
| `served-2026-11-09-afetch2` | 2026-11-09 20:06 to now | 153ce47d3ef0 | f381bfdf5 |
| `served-2026-11-09-p028x-upmix-guard` | 2026-11-09 11:39 to 20:06 | f4c4f654fb24 (unchanged) | 019b04dd2 |
| `served-2026-10-30-p028x` | 2026-10-30 to 11-09 | f4c4f654fb24 | 8f7a48aab |
| `served-2026-10-27-ds41-daily` | 2026-10-27 to now (DeepSeek) | 575de3a8df3f | v0.2.6 + local diff |
| `served-2026-10-23-p021m` | 2026-10-23 to 10-30 | 7b38796237f5 | a653ae6c3 |
| `served-2026-10-22-p021n` | 2026-10-22 to 10-23 | 68befe4e03bf | 785e45dfd |
| `served-2026-10-20-p021` | 2026-10-20 to 10-22 | c9fe314316c4 | a8a5ee601 |
| `served-2026-10-16-cstar18t` | 2026-10-16 to 10-20 | e428a826c5fb | a2ec5ac1d (engine: 113e4f0f7) |

Each tag's message names its config, engine and dates.

## Why the hashes differ from the local clone

Before publishing, our own commits were rewritten in one pass. The local working clone keeps the original hashes, which the engines' `SOURCE` files record (the last column above).
- The author and committer emails were normalized to the GitHub noreply address. The cherry-picked fetch admission got its original author, sergqwer, back.
- Links to private work sessions (`Claude-Session:` lines) were removed from the commit messages. The `Co-Authored-By` trailers are kept.
- One hard-coded home path in `serve/websearch.py` (the paid-search fallback's import path) now reads `/home/USER/...`; set it to your own home before using that fallback.

No other file content changed. Upstream's commits and tags are untouched.

## Credits

Everything here is MIT licensed, like Strata.
- **Upstream Strata:** Niko1221 and the Strata contributors. Cherry-picked upstream commits keep their authors.
- **Basalt (MIT, Copyright (c) 2026 Jesús Gulfo):**
  - the v2 prompt kernels (`STRATA_PREFILL_V2`) and the cumulative draft-length rule (`STRATA_DRAFT_CUM`), in the P02x series and the ports;
  - `PF_AHEAD`, `TAIL_SPLIT` and `FIRST_CHUNK` in `r23-basalt-prompt`.

  Basalt's MIT notice is kept in the ported files, and the porting commits credit it.
- **Project Maya #343** (Maya commit 7644743, v1.1.26; MIT): the prompt-skip change in `ds41-v026-daily`. Strata's `third_party/project-maya/` carries Maya's license and notice.
- **sergqwer:** the PCIe share's fetch admission (PR 2964, upstream 24c6c2f68), authored by sergqwer in `r23-adaptfetch`.
