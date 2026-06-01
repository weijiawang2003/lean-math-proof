# TR7 — Fresh Delta Gap Analysis: TR6 Ranker Search vs RC4 Static Wrapper

**Type:** diagnostic (no candidate, no promotion, no wrapper change)
**Date:** 2026-06-01

---

## 1. Executive summary

TR6 (ranker-guided live search) found **18 fresh TRUE_DELTA** wins; RC4R (the static RC4
wrapper) showed **0 fresh out-of-sample delta**. TR7 explains the gap. It has **three causes**,
in order of magnitude:

1. **Cohort / selection artifact (dominant).** **14 of the 18 TR6 wins were folded into RC4D
   validation** as RC4A/B/C evidence; RC4R's fresh frontier **explicitly excluded every
   RC4D-used theorem**, so **0/18 TR6 wins could appear in the RC4R fresh set by construction**.
   RC4 actually **covers 10–11 of the 18 TR6 wins** — they are its *known wins* (part of the +22).
   The headline "0 fresh delta" therefore massively understates RC4's real coverage of the TR6 wins.
2. **Static-abstraction limit (real).** Of the 8 TR6 wins RC4 does not cover: **4 miss the gate**
   (winning lemma not in the 14-lemma allowlist, and they do **not** cluster into an addable
   family), **3 fail because RC4's static `simp [disjoint_left]` is the wrong tactic** where TR6
   won via a theorem-specific `tauto`/`rw`, and **1 is a wrapper-search representation gap**
   (`Set.disjoint_sUnion_right`: the action solves single-shot but the best-first search does not).
3. **RC4A broad-gate dilution.** RC4R's fresh frontier **over-sampled RC4A mono cases** (RC4A gate
   fires 71× on fresh, precision **0.09**) and under-sampled the disjoint/subset families TR6 won
   on — so the fresh benchmark mostly exercised the one component that rarely closes.

**Static-compatible: 78% (14/18)** of TR6 wins (10 already in RC4, 4 reachable with
allowlist/gate/schema work); **dynamic-only: 22% (4/18)** need theorem-specific retrieved lemmas.

**Decision: `HYBRID_STATIC_PLUS_RANKER_NEXT`** — keep RC4 static as the safe deterministic core,
add a gated ranker-guided dynamic-retrieval stage for the dynamic tail (RC5H). Secondary:
`GATE_REFINEMENT_NEXT` (tighten RC4A) and `NEED_MORE_FRESH_DATA` (TR8 for 3 allowlist-expansion
candidates).

---

## 2. Background

- **TR6** (commit 2c0dc72): ranker-guided live search over a fresh 3,616-pool frontier (200
  batch → 137 RC2 failures), 2,458 ranked programs probed → **18 fresh TRUE_DELTA, 13 non-Set**.
  Decision `RANKER_GENERALIZES_TO_FRESH_FRONTIER`.
- **RC4R** (just completed): static RC2⊕RC4A⊕RC4B⊕RC4C_residue wrapper benchmark → net **+22**
  (all known reproduction), 0 regressions, 0 off-gate, floors preserved, deterministic, but
  **0 fresh out-of-sample delta** → `RC4_SAFE_BUT_NO_FRESH_DELTA`.
- The gap matters because it decides RC5's shape: does the next step expand the static allowlist,
  or does fresh generalization fundamentally require the dynamic retrieval TR6 used?

---

## 3. Comparison corpus

`cases/tr7_comparison_corpus.jsonl` (`out/tr7_comparison_corpus_summary.json`) — **373 rows**
(TR6 137 searched ∪ RC4R 271 benchmark). 18 TR6 fresh wins. **Key joins:**

- TR6 fresh wins in **RC4R fresh frontier: 0/18** (the cohort artifact).
- TR6 fresh wins in **RC4R known wins: 11/18**.
- RC4 static gate fires on **14/18** TR6 wins.

---

## 4. Distribution mismatch

`out/tr7_distribution_mismatch.json` → **`PARTIAL_DISTRIBUTION_MISMATCH`**.

- 0/18 TR6 wins present in the RC4R fresh set (excluded by construction).
- RC4R fresh gate firing: **RC4A 71 vs RC4B 2 + RC4C_residue 7** — a ~8× skew toward the loose
  RC4A def-unfold gate, while TR6's fresh wins were **disjoint-shaped** (14/18 have `has_disjoint`).
- Namespaces broadly overlap (so not a *total* mismatch), but the **winnable-family composition**
  is skewed: RC4R fresh over-tested RC4A mono (low precision) and under-tested the
  Multiset/Set-disjoint + subset-pair patterns TR6 actually won on.
- **Conclusion: the 0 fresh delta is partly a benchmark selection artifact.**

---

## 5. Static coverage audit (core diagnostic)

`out/tr7_static_coverage_audit.json` — per TR6 win, would the RC4 static wrapper cover it?

| class | n | meaning |
|---|---|---|
| STATIC_COVERED_AND_SHOULD_SOLVE | 10 | RC4 already solves (known wins) |
| ALLOWLIST_MISS | 3 | winning lemma not in the 14-lemma allowlist (`Finset.subset_iff`, `Nat.le_sqrt`, `Set.MapsTo`) |
| DYNAMIC_RETRIEVAL_REQUIRED | 3 | won via theorem-specific `tauto`/`rw` (Multiset disjoint symm/comm, add_eq_union) |
| WRAPPER_REPRESENTATION_MISS | 1 | `Set.disjoint_sUnion_right` — action solves single-shot, search fails |
| RC4C_RESIDUE_EXCLUDED | 1 | `Finset.biUnion_subset` (deliberately dropped depth-1 simp_only) |

**RC4 static would cover 10/18; 8 missing** (4 allowlist incl. excluded, 0 pure gate-miss,
3 dynamic, 1 schema).

---

## 6. Live replay on TR6 wins

`out/tr7_rc4_replay_on_tr6_wins.json` — 18 wins replayed (7 live RC4 searches + exact TR6
programs single-shot; 11 RC4 results reused from the RC4R benchmark).

| class | n |
|---|---|
| RC4_REPRODUCES_TR6_WIN | 10 |
| RC4_MISSES_GATE | 4 |
| RC4_ACTION_FAILS_BUT_TR6_PROGRAM_WORKS | 4 |

- **The exact TR6 winning program reproduces on ALL 18** — TR6's dynamic wins are real and stable.
- RC4 reproduces **10/18** (its known wins minus the 1 search-depth gap).
- **4 RC4_MISSES_GATE**: no RC4 action fires (`Finset.biUnion_subset_iff_forall_subset`,
  `Finset.image_subset_iff`, `Nat.sqrt_pos`, `Set.mapsTo_singleton`).
- **4 RC4_ACTION_FAILS_BUT_TR6_PROGRAM_WORKS**: RC4's `simp [disjoint_left]` action fires but
  fails while TR6's theorem-specific tactic works (`Multiset.Disjoint.symm` & `disjoint_comm` via
  `tauto`; `Multiset.add_eq_union_right_of_le` via `rw`); and `Set.disjoint_sUnion_right` where
  the action *does* solve single-shot (`rc4_action=True`) but the **search** does not — a clean
  schema-representation gap.

This is the crisp localization: where RC4 misses, it is because its **fixed action set is too
small (gate miss) or the wrong tactic (action fails)**, while the TR6 **ranker retrieves the
right per-theorem program every time**.

---

## 7. Missing allowlist analysis

`out/tr7_missing_allowlist_analysis.json`.

- ALREADY_IN_ALLOWLIST: 10 · **ADD_TO_STATIC_ALLOWLIST: 0** · KEEP_DYNAMIC_ONLY: 5 ·
  NEED_MORE_EVIDENCE: 3 (`Finset.biUnion_subset`, `Finset.subset_iff`, `Set.MapsTo`).
- **No missing lemma recurs or is namespace-parametric enough to add now.** The dynamic tail
  (`tauto`/`rw`/`exact`) is theorem-specific by nature. **Static allowlist expansion alone will
  not close the gap** — the residual wins do not form a new family.

---

## 8. Gate refinement analysis

`out/tr7_gate_refinement_analysis.json`.

| component | fired | closed | precision | change |
|---|---|---|---|---|
| RC4A | 76 | 7 | **0.09** | **TIGHTEN** |
| RC4B | 18 | 15 | 0.83 | keep |
| RC4C_residue | 20 | 12 | 0.60 | keep |

- **RC4A** def-unfold gate is broad: fires on every monotone/antitone theorem, closes ~9%.
  Recommend tightening to the iff-unfold shape (require `_iff_` in the name) — additive/safe today
  so low urgency. Proposals: **RC4A_TIGHTEN_MONO_GATE, RC4B_KEEP, RC4C_RESIDUE_KEEP,
  DYNAMIC_RETRIEVAL_REQUIRED.**

---

## 9. Dynamic vs static classification

`out/tr7_dynamic_vs_static_classification.json`.

| class | n |
|---|---|
| STATIC_WRAPPER_COMPATIBLE_NOW | 10 |
| STATIC_WRAPPER_COMPATIBLE_WITH_ALLOWLIST_EXPANSION | 3 |
| STATIC_WRAPPER_COMPATIBLE_WITH_SCHEMA_FIX | 1 |
| DYNAMIC_RETRIEVAL_PREFERRED | 4 |

**78% static-compatible** (10 now + 4 with work) · **22% dynamic-only** → **hybrid RC5**.

---

## 10. RC5 recommendations

`out/tr7_rc5_recommendations.json` — primary **`RC5_HYBRID_STATIC_PLUS_RANKER`**.

- **Rationale:** most TR6 wins are static-compatible (they *became* RC4) but a real ~22% tail
  needs theorem-specific retrieved lemmas a fixed allowlist cannot hold. Keep RC4 static as the
  deterministic, safe core; add a gated ranker-guided dynamic-retrieval stage for the tail.
- **Benefit:** recovers fresh generalization without losing RC4's safety/determinism.
- **Risk:** dynamic retrieval reintroduces nondeterminism + probe cost (gate + owner-bill it).
- **Validation:** RC5 hybrid benchmark — RC4 static floors/known-wins preserved + dynamic stage
  measured on a fresh frontier with SX4 attribution; determinism scoped to the static core.
- **Secondaries:** RC4A_TIGHTEN_MONO_GATE (precision 0.09); TR8_MORE_FRONTIER_DATA for the 3
  allowlist-expansion candidates (gather recurrence before adding).

---

## 11. Diagnostic dataset

`data/tr7_fresh_delta_gap_examples.jsonl` (18 rows) + `data/tr7_diagnostic_summary.json`. Each
row carries the static-coverage class, replay class, dynamic/static class, and a concrete
`recommended_next_action`. **Use:** scope RC5H (which wins the hybrid must capture dynamically vs
statically) and seed TR8's recurrence search. **Not** a prover training set; no prior dataset
overwritten.

---

## 12. Decision

**Primary: `HYBRID_STATIC_PLUS_RANKER_NEXT`.**
Secondary: `GATE_REFINEMENT_NEXT` (RC4A tighten) · `NEED_MORE_FRESH_DATA` (TR8 for the 3
allowlist-expansion candidates).

Rejected: `STATIC_ALLOWLIST_EXPANSION_NEXT` (0 addable lemmas now) · `DYNAMIC_RETRIEVAL_RC5_NEXT`
alone (78% is static-compatible — pure dynamic would discard RC4's validated safe core) ·
`ORDER_STRUCTURAL_BATTERY_NEXT` (Order-family did not dominate the residual).

---

## 13. Next concrete task

**RC5H — Hybrid static+ranker wrapper prototype & fresh-frontier benchmark.** Compose RC4 static
(deterministic core, floors/known-wins preserved) with a gated TR5/TR6-style ranker-guided
dynamic-retrieval stage; benchmark on a fresh out-of-sample frontier with SX4 attribution; verify
the static core stays deterministic and floor-safe while the dynamic stage recovers fresh delta.
Why: TR7 shows the static abstraction has reached its coverage ceiling on this evidence (0 addable
lemmas), and the only mechanism that produced fresh wins is dynamic retrieval — but RC4 static is
too valuable (safe +22) to discard, so the next step is hybridization, not replacement.

---

## 14. Protected-file confirmation

- RC1 wrapper — untouched · RC2 release wrapper — untouched · NS24 router — untouched
- RC4A/B/C/D + RC4R artifacts — untouched · NS9/REL1/RC1/RC2 reports · TR1–TR6 datasets — untouched
- No production routing change · no wrapper modification · no RC5 release · no promotion · **no commit made**
- `git diff --stat HEAD` over protected wrappers + router + RC4R wrapper: **empty**.
