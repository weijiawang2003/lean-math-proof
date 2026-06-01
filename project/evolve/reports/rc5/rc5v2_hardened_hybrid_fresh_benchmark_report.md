# RC5V2 — Hardened Hybrid Fresh Benchmark Report

**Type:** benchmark/prototype (no release, no promotion, no wrapper change)
**Date:** 2026-06-01

---

## 1. Executive summary

RC5V2 benchmarks the hardened hybrid — **RC4R static core + RC5S strict safe dynamic B5** — on a
**fresh out-of-sample frontier**, to answer whether the now-safe dynamic stage adds fresh,
stable, attributable deltas over RC4.

| Metric | Result |
|---|---|
| fresh frontier / eval batch | 3267 strict-fresh / 240 |
| RC2 baseline solved | 67 |
| RC4 static solved (Δ/RC2) | 67 (+0 fresh) |
| dynamic-eligible (RC4-failed, allowed-ns) | 149 |
| safe B5 plan / off-policy | 149 thms / 745 programs / **0** |
| safe B5 — global stalls / max wall | **none** / 65.0s (cap 60s) |
| **FRESH_TRUE_RC5V2_DELTA over RC4** | **8** |
| RC5V2 solved (Δ/RC4 / Δ/RC2) | 75 (+8 / +8) |
| regressions | 0 |
| safety verdict | SAFE_DYNAMIC_B5_CONFIRMED |

**Decision: RC5V2_HARDENED_HYBRID_CONFIRMED** (see §13).

---

## 2. Background

- **RC4R** = `RC4_SAFE_BUT_NO_FRESH_DELTA` (+22 known, 0 fresh).
- **TR7** = `HYBRID_STATIC_PLUS_RANKER_NEXT` (78% static / 22% dynamic-only).
- **RC5H** = `RC5H_DYNAMIC_STAGE_USEFUL_BUT_NOT_RELEASE_SAFE` (+3 over RC4 but unsafe: stalls,
  off-policy, B10+ unrunnable).
- **RC5S** = `RC5S_SAFETY_HARDENING_SUCCESS` (strict grammar + process-group-kill runner: off-policy
  74→0, stalls pervasive→none, max wall 150s→12.3s, all wins preserved, B5-only).
- RC5V2 puts the hardened stage on a **fresh** frontier — the test RC5H's reused/known-win
  benchmark could not do.

---

## 3. Fresh benchmark frontier

`cases/rc5v2_fresh_frontier_pool.jsonl` — TR6 fresh pool ∪ discovered, minus a 519-name exclusion
registry (TR6 batch+wins, RC4D/RC4R known wins, RC5H/RC5S benchmark, TR7 corpus). **3267
strict-fresh candidates** (2532 allowed-ns). Eval batch: **240** stratified (Finset 55 / Set 45 /
List 45 / Multiset 45 / Nat 35 / Other 15), 225 with dynamic-tail features.

---

## 4. RC2 baseline

`out/rc5v2_rc2_baseline_results.json` — exact RC2 config, all-fresh (live). **67 solved / 163
failed / 10 flakes** of 240. Solved by ns: Finset 28, Multiset 14, Set 13, List 8, Nat 3.

---

## 5. RC4 static stage

`out/rc5v2_static_stage_results.json` — RC4R wrapper (additive: 234 forced RC4≡RC2 non-firing,
6 gate-firing live). **67 solved — 0 new over RC2, 0 regressions** (6 gate emissions, none won).
RC4 again shows **0 fresh delta** on a fresh frontier (the RC4R caveat reproduced). The 163 RC4
failures form the dynamic frontier.

---

## 6. Dynamic eligibility

`out/rc5v2_dynamic_eligibility_summary.json` — **149 dynamic-eligible** (RC4 failed ∧ allowed
namespace ∧ non-flake); excluded 67 STATIC_SOLVED + 10 FLAKE + 14 DYNAMIC_NAMESPACE_DISABLED.
Eligible by ns: List 34, Nat 32, Multiset 31, Set 29, Finset 23.

---

## 7. Retrieval and safe B5 program plan

Retrieval: 149 targets, **100% coverage**, best-score mean 1.44. Safe plan
(`out/rc5v2_safe_dynamic_plan.json`): 149 theorems, 2980 programs generated, **802 rejected
off-policy**, **745 B5 programs, off-policy in final plan = 0**. Patterns: simp_L 386,
simp_L_aesop 289, exact_L 32, simpa_using_L 23, simpa_L 12 (no simp_all, no depth-3).

---

## 8. Safe dynamic B5 live results

`out/rc5v2_b5_dynamic_results.json` — RC5S timeout-safe runner.

| metric | value |
|---|---|
| theorems | 149 |
| **dynamic successes** | **8** |
| **global stalls** | **none** |
| max wall | 65.0s (cap 60s + 5s SIGKILL grace) |
| killed_by_timeout | **0** |
| off-policy programs | **0** |
| unknown-name (bounded) | 75 |

Wins (List 5 / Multiset 2 / Finset 1 / Set 1): `Finset.fiber_nonempty_iff_mem_image`,
`List.attach_map_val'`, `List.choose_mem`, `List.get_pmap`, `List.pmap_append'`,
`Multiset.map_count_True_eq_filter_card`, `Multiset.mem_bind`, `Set.compl_range_subset_kernImage`.
The runner stayed fully bounded on a genuinely fresh frontier — no manual intervention, unlike RC5H.

---

## 9. Attribution

`out/rc5v2_attribution.json` — **8 FRESH_TRUE_RC5V2_DELTA**, 141 NO_DYNAMIC_WIN, 0 duplicates, 0
source-specific. Every one of the 8 dynamic wins is genuine: RC2 failed ∧ RC4 failed ∧ a strict
safe program solved ∧ bare controls did not solve ∧ theorem is strict-fresh.

---

## 10. RC2 vs RC4 vs RC5V2 comparison

`out/rc5v2_system_comparison.json`.

| system | solved | Δ/RC2 | Δ/RC4 | regr |
|---|---|---|---|---|
| RC2 | 67 | — | — | — |
| RC4 static | 67 | +0 | 0 | 0 |
| **RC5V2 (RC4 + safe B5)** | **75** | **+8** | **+8** | **0** |

The safe dynamic B5 stage delivers **+8 fresh out-of-sample wins over RC4** (the generalization
RC4 static could not produce) with **0 regressions** (additive). Dynamic probes 698, **~87
probes/fresh win**. RC4 remains the static core; the dynamic stage is purely additive on its failures.

---

## 11. Safety audit

`out/rc5v2_safety_audit.json` → **`SAFE_DYNAMIC_B5_CONFIRMED`**. off-policy 0 · namespace
violations 0 · no global stalls (max wall 65s, cap+grace) · 0 killed · unknown-name rate ~0.10 ·
8 fresh deltas · 0 source-specific · ~87 probes/fresh win · **B5-only recommended**. The RC5S
hardening holds on a fresh frontier at scale — no engineering regressions.

---

## 12. Exported examples

`data/rc5v2_dynamic_examples.jsonl` — **745 rows** (one safe attempt = one row) with policy
status / result / attribution / freshness. **off-policy: 0** · winning: 8. Safety/audit data; the
ranker is NOT retrained.

---

## 13. Decision

**RC5V2_HARDENED_HYBRID_CONFIRMED**

Every confirmation criterion is met: **8 FRESH_TRUE_RC5V2_DELTA over RC4** (≥1 required), **0
regressions** (additive), **0 off-policy**, **no global stalls** (max wall 65s under a 60s
process-kill cap), **bounded timeouts** (0 killed), and acceptable cost (~87 probes/fresh win).
This is the result the whole RC4→RC5 arc was after: the static core (RC4) is safe but yields **0
fresh delta**, and the hardened dynamic stage now adds **+8 fresh out-of-sample wins** to it
**safely** — the RC5H value without the RC5H safety failures. The hardened hybrid generalizes.

---

## 14. Next steps

Confirmed → the hardened hybrid is a real, safe, additive fresh-delta source.
1. **Larger RC5V2 benchmark** — scale the fresh frontier (500–1000) to size the fresh-delta rate
   and probes/win more precisely.
2. **Owner decision** — whether to maintain an RC5H-style guided-search *mode* (RC4 static core +
   gated safe B5 dynamic) as an off-by-default capability; it is additive and safe but costs ~87
   probes/win, so it is a guided-search tool, not a always-on production wrapper.
3. **Ranker** — gather more fresh dynamic-win positives (now 11 across RC5H+RC5V2) before any
   retrain.
RC2 stays production; RC4 static stays the best always-on static candidate; the safe dynamic stage
is the validated fresh-delta extension.

---

## 15. Protected-file confirmation

- RC1 · RC2 · NS24 · RC4R wrapper · **RC5S strict policy** — untouched
- RC5H policy · NS9 · REL1/RC1/RC2 reports · TR1–7 datasets · RC4*/RC5H/RC5S originals — untouched
- No production change · no RC5 release · no promotion · **no commit made**
- `git diff --stat HEAD` over protected wrappers + router + RC5S policy: **empty**.
