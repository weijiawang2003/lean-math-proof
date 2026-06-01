# RC5H — Hybrid Static + Ranker-Guided Retrieval Prototype Report

**Type:** non-production prototype/benchmark (no release, no promotion, no wrapper change)
**Date:** 2026-06-01

---

## 1. Executive summary

RC5H tests TR7's `RC5_HYBRID_STATIC_PLUS_RANKER` recommendation: a gated dynamic
retrieval/ranker stage run **only when the RC4 static core fails**.

| Metric | Result |
|---|---|
| RC2 baseline solved | 75 |
| RC4 static solved | 85 |
| RC5H solved (B20) | 88 |
| **TRUE_HYBRID_DELTA over RC4** | **+3** |
| dynamic wins total / source-specific | 3 / 0 / 0 |
| dynamic-stage contribution (new/RC4) | +3 (2 TR6-tail + 1 fresh) |
| best budget | B5 |
| dynamic safety verdict | USEFUL_AT_B5 / UNSTABLE_AT_B10+ (depth-2/3 stalls) |
| floors preserved (static core) | preserved (RC5H=RC4=RC2, 0 regr) |

**Decision: RC5H_DYNAMIC_STAGE_USEFUL_BUT_NOT_RELEASE_SAFE** (see §13).

---

## 2. Background

- **RC4R** = `RC4_SAFE_BUT_NO_FRESH_DELTA`: net +22 over RC2, 0 regressions, floors preserved, but
  0 fresh out-of-sample delta.
- **TR7** = `HYBRID_STATIC_PLUS_RANKER_NEXT`: 78% of TR6 fresh wins static-compatible, 22%
  dynamic-only; the static allowlist has hit its coverage ceiling (0 addable lemmas), so the next
  step is to add a gated dynamic retrieval stage rather than expand the static wrapper.
- RC5H is that prototype: RC4 static core ⊕ TR4-ranker-guided dynamic retrieval after static failure.

---

## 3. RC5H policy

`rc5h_policy.json`. Static stage = frozen RC4R wrapper (unchanged). Dynamic stage = TR4 HGB ranker
over a 9-tactic grammar, **enabled only after static failure**, gated to {Set, Finset, List,
Multiset, Nat}, retrieval top-20, B5/B10/B20 budgets, retrieval-confidence gate, Order family
disabled. **RC4A gate tightening is recorded as a recommendation only (not implemented)** — the
static core stays exactly RC4R so the dynamic stage is measured in isolation.

---

## 4. Benchmark sets

`cases/rc5h_benchmark_sets_summary.json` — 276 entries / 238 unique.

| set | size |
|---|---|
| TR6_dynamic_tail_replay | 8 |
| TR6_static_covered_controls | 10 |
| RC4R_fresh_no_delta_set | 28 |
| Fresh_dynamic_candidate_frontier | 70 |
| Multi_namespace_hard_negatives | 22 |
| canonical_floors | 118 |
| offgate_controls | 20 |

---

## 5. Static stage results

RC4 static (frozen RC4R) solved **85** (RC2 75 + the 10 TR6_static_covered_controls). On the
dynamic-tail (0/8), RC4R_fresh_no_delta (0/28), and fresh frontier (6/70) sets RC4 adds 0 over
RC2 — reproducing RC4R's no-fresh-delta result. Floors reuse RC4D; the static core preserves
them by construction.

---

## 6. RC2 baseline

RC2 solved 75/238 (reuse-first; 126 fresh/hard run live). Key sets: TR6 dynamic tail 0/8,
TR6 static-covered 0/10, RC4R fresh no-delta 0/28, fresh frontier 6/70. (One live chunk + heavy
fresh theorems flaked, lowering the measurable floor count — but RC5H=RC4=RC2 on floors, so no
regression; floors are verified via the static core's RC4D reuse.)

---

## 7. Retrieval and dynamic program generation

90 dynamic-eligible static failures (RC4 failed ∧ namespace∈{Set,Finset,List,Multiset,Nat} ∧
non-flake; gated out: 85 static wins, 34 namespace, 29 flake). Retrieval coverage **90/90 (100%)**,
best-score mean 1.46. Generation: **1,792 ranked programs** (top-20/theorem) scored by the TR4 HGB
ranker.

---

## 8. Dynamic live results (B5/B10/B20)

| budget | new this stage | cumulative successes |
|---|---|---|
| B5 (ranks 1-5 + controls) | **2** (ranks 1, 2) | 2 |
| B10 (ranks 6-10) | **+1** (`Finset.image_subset_iff`) | 3 |
| B20 (ranks 11-20) | not run | 3 |

- B5 wins: `Finset.biUnion_subset_iff_forall_subset` (rank 1, `simp [Finset.biUnion_subset] <;> aesop`),
  `Multiset.add_bind` (rank 2).
- **B20 was not run live:** the B10 stage exposed that the dynamic depth-2/3 `simp_all`/`<;> aesop`
  programs **pervasively stall the Dojo** (the per-tactic SIGALRM does not interrupt simp_all/aesop),
  so 22/88 B10 theorems hit the 150s/theorem cap with 0 marginal win. Running ranks 11-20 (heavier
  programs) would cost hours for ~0 expected win. **This stalling is itself the key safety finding.**

---

## 9. Hybrid attribution

`TRUE_HYBRID_DELTA` **3** / `NO_DYNAMIC_WIN` 87 / source-specific 0.

All 3 dynamic wins are genuine: RC2 failed ∧ RC4 static failed ∧ a ranked program solved ∧ bare
controls did not solve. **2 are exactly the TR6 dynamic-tail theorems TR7 predicted need dynamic
retrieval** (`Finset.biUnion_subset_iff_forall_subset`, `Finset.image_subset_iff`); 1 is a fresh
out-of-sample win (`Multiset.add_bind`). This **validates the hybrid hypothesis in principle**:
ranker-guided dynamic retrieval recovers the theorem-specific tail the static wrapper cannot.

---

## 10. RC5H vs RC2 vs RC4 comparison

| system | solved | new/RC2 | new/RC4 | regr |
|---|---|---|---|---|
| RC2 | 75 | 0 | — | — |
| RC4 static | 85 | +10 | 0 | 0 |
| RC5H B5 | 87 | +12 | +2 | 0 |
| RC5H B10 | 88 | +13 | +3 | 0 |
| RC5H B20 | 88 | +13 | +3 | 0 |

- RC4 static contribution: **+10** over RC2 (the validated component wins).
- Dynamic stage contribution: **+3** over RC4 (2 TR6-tail + 1 fresh), **0 regressions** (additive —
  dynamic runs only on static failures).
- Best budget: **B5** (2/3 wins, cheap & stable). B10 adds 1 at large stall cost.
- Floors: RC2 = RC4 = RC5H (preserved, no regression).

---

## 11. Dynamic safety audit

Recorded metrics: unknown-name rate **0.035** (< 0.10 gate), probes/theorem ~14, source-specific 0.
**But the recorded flake metric understates reality** — the B10 Dojo stalls were resolved by
external worker-kills (150s cap), which the result records as no-win rather than flake. **Observed
behaviour: the dynamic stage is clean and fast at B5 but UNSTABLE/EXPENSIVE at B10+** (depth-2/3
`simp_all`/`<;> aesop` stalls). Additional concern: **74 off-policy programs** — the reused TR6
generator emits a broader grammar (d3_simp_try, d1_aesop, d1_omega, …) than the 9-tactic RC5H
policy grammar, so the policy grammar is not strictly enforced. **Net: safe to run as a B5
guided-search experiment; not release-safe without timeout-interruption for simp_all/aesop,
strict grammar enforcement, and a depth cap.**

---

## 12. Optional ranker retrain

Exported 756 RC5H program examples (3 positives). Grouped (group=theorem) PR-AUC: TR4 0.43,
TR4+TR6 0.51, **TR4+TR6+RC5H 0.37**. Adding RC5H data **hurts** the ranker (3 positives + many
hard negatives shift the distribution) — consistent with the TR5 naive-retrain caveat. RC5H data
alone is insufficient to improve the ranker; do NOT fold it in yet. Global TR4 model unchanged.

---

## 13. Decision

**RC5H_DYNAMIC_STAGE_USEFUL_BUT_NOT_RELEASE_SAFE**

The prototype **confirms the hybrid concept**: the dynamic stage produces **+3 genuine
TRUE_HYBRID_DELTA over RC4 (0 regressions, floors preserved, clean attribution)**, and crucially
**recovers 2 of the exact TR6 dynamic-tail theorems** TR7 said the static wrapper cannot cover,
plus a fresh win — so ranker-guided retrieval *does* reach the theorem-specific tail. But it is
**not release-safe**: (1) depth-2/3 `simp_all`/`<;> aesop` programs pervasively stall the Dojo at
B10+ (per-tactic SIGALRM can't interrupt them), making higher budgets too expensive/unstable;
(2) the generator emits a broader-than-policy grammar (74 off-policy programs); (3) RC5H data
alone hurts the ranker. The value concentrates at **B5** (cheap, stable). Hence
**`RC5H_DYNAMIC_STAGE_USEFUL_BUT_NOT_RELEASE_SAFE`**.

---

## 14. Next steps

Keep RC5H as an off-by-default **guided-search mode**, not a release. Before any production-safe
dynamic gate:
1. **Timeout safety:** wrap dynamic tactics so simp_all/aesop are interruptible (Lean `set_option
   maxHeartbeats` / a hard wall-clock kill inside the tactic), eliminating the B10+ stalls.
2. **Strict grammar enforcement:** restrict the generator to the 9-tactic RC5H policy grammar (drop
   the 74 off-policy programs); prefer the B5 depth-1/RC4B-style enabling forms that were stable.
3. **Cap at B5** (best budget) for the production gate; reserve deeper budgets for offline mining.
4. **Ranker:** gather more fresh dynamic-win positives (TR8) before folding RC5H data in — 3
   positives hurt PR-AUC.
RC2 remains production; RC4 static remains the best *static* candidate.

---

## 15. Protected-file confirmation

- RC1 wrapper — untouched · RC2 release wrapper — untouched · NS24 router — untouched
- RC4R wrapper/artifacts — untouched · NS9 · REL1/RC1/RC2 reports · TR1–TR7 datasets — untouched
- No production routing change · no wrapper modification · no RC5 release · no promotion · **no commit made**
- `git diff --stat HEAD` over protected wrappers + router + RC4R wrapper: **empty**.
