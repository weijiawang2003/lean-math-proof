# TR6 — Ranker-Guided Fresh Multi-Namespace Frontier Sweep

**Decisions:** `RANKER_GENERALIZES_TO_FRESH_FRONTIER` · `FRESH_TRUE_DELTA_FOUND` ·
`NONSET_POSITIVES_FOUND` · `RC4B_READY_FOR_LITERAL_VALIDATION_WITH_FRESH_SUPPORT` ·
`RC4C_READY_FOR_LITERAL_VALIDATION_WITH_FRESH_SUPPORT`.

Exploratory; ranker/candidates **not promoted**, no production change, no RC4 release, no commit.

---

## 1. Executive summary

- **Fresh frontier:** 3,616 candidates source-scanned from the traced Mathlib tree across
  ~25 namespaces, minus a 158-name exclusion registry (zero leakage).
- **Eval batch:** 200 theorems, stratified (Set 30 / Finset 35 / List 35 / Nat 35 /
  Multiset 25 / Order 20 / Other 20), **0 overlap** with prior work.
- **Fresh RC2 failures:** **137 / 200** confirmed live (59 RC2-solved, 4 flakes) — far
  above the 50 target; failures span Finset 30, List 29, Multiset 22, Set 21, Nat 14,
  plus Order-family (AntitoneOn/MonotoneOn/IsGLB/IsLUB), Option, Equiv.
- **Live programs:** B5 654 + B10 622 + B20 1,182 = **2,458 ranked programs** (+540 controls).
- **Fresh true deltas:** **18 FRESH_TRUE_DELTA** (3 baseline-dup, 2 flakes, 114 no-win).
- **Non-Set positives:** **13** — Multiset 9, Finset 2, List 1, **Nat 1** (TR4 had **0**
  Multiset/Nat positives).
- **Ranker decision:** `RANKER_GENERALIZES_TO_FRESH_FRONTIER`.
- **RC4B / RC4C:** 8 / 9 fresh wins → both `READY_FOR_LITERAL_VALIDATION_WITH_FRESH_SUPPORT`.
- **Headline:** unlike TR5 (0 fresh wins), TR6 finds **18 genuinely new wins in new
  namespaces**, and discovers that the `disjoint_left` bridge **generalizes** —
  `Multiset.disjoint_left` is a fresh analogue of `Set.disjoint_left`.

## 2. Motivation

TR5 proved the ranker recovers TR3's wins ~5× more efficiently but found **0 fresh wins**
and left the by-namespace generalization gap open (Nat/Multiset had 0 positives). TR6 tests
whether ranker-guided search produces **new coverage** on a fresh multi-namespace frontier.

## 3. Exclusion registry

`scripts/tr6_build_exclusion_registry.py` — **158 unique full_names** from TR1 (57) / SF4
(40) / SF5 (20) / TR3 (150) / TR4 (96) / TR5 (92) / RC4A known-wins (5) (heavy overlap;
all draw from the same discovered-theorems pool). Guarantees zero in-sample evaluation.

## 4. Fresh frontier construction

`scripts/tr6_build_fresh_frontier.py` source-scans a curated 26-file multi-namespace list
from the traced cache (cx1-style regex with proper namespace/section tracking; a smoke test
confirmed List/Order/Multiset/Finset theorems **open in LeanDojo**) and merges
discovered_theorems.json, excluding the registry and keeping proof-search-relevant
theorem/lemma decls (non-private, real file_path). **3,616 fresh candidates**: Finset 824,
Set 627, Multiset 524, Nat 402, List 381, Int 105, Option 73, plus a full Order family
(BddAbove/IsGLB/IsLUB/Monotone/Preorder/…). The live RC2 step is the final availability filter.

## 5. Eval batch

`scripts/tr6_select_eval_batch.py` — 200 theorems, stratified quotas met exactly, prioritised
within stratum by proof-search features (disjoint/iff/subset rank high for RC4B/RC4C support),
48 RC4B/RC4C candidates tagged, **overlap with exclusion registry = 0**.

## 6. Literal RC2 confirmation (live, fresh)

`scripts/tr6_confirm_rc2.py` — live RC2 (rc2_release wrapper, ns24, hybrid_evolved, top-k 8,
max-steps 8) on all 200. **137 CONFIRMED_RC2_FAILURE**, 59 RC2_SOLVED, 4 OPEN_FLAKE. The
~68 % fresh-failure rate shows RC2 does **not** already cover this frontier — selection was
not too easy. Failures by namespace: Finset 30, List 29, Multiset 22, Set 21, Nat 14, ''
(root/Order) 10, Option 3, AntitoneOn/MonotoneOn 2 each, IsGLB/IsLUB/PLift/Equiv 1 each.

## 7. Retrieval and ranked program generation

`scripts/tr6_retrieve_lemmas.py` retrieves top-20 lemmas per failure from the TR3∪SF5 index
(10,790+ decls; reuses the TR3/SF5 scorer; statement_text as the query goal) — coverage
137/137, avg best score 1.38. `scripts/tr6_generate_ranked_programs.py` builds the TR3/TR5
grammar (+RC4B `simp [Set.disjoint_left]` probes), scores **7,090 programs** with the TR4
HGB ranker, keeps top-20/theorem, assigns B5/B10/B20.

## 8. Live B5/B10/B20 search

`scripts/tr6_run_ranked_live_search.py` (serialized, one Dojo/theorem, hard-timeout workers,
per-theorem checkpoint, controls at B5).

| stage | theorems run | programs | new wins | cumulative wins |
|---|---|---|---|---|
| B5 (ranks 1–5) | 137 | 654 (+540 controls) | 9 | 9 |
| B10 (ranks 6–10) | 128 | 622 | 3 | 12 |
| B20 (ranks 11–20) | 125 | 1,182 | 9 | **21** |

21 live successes; mean first-success rank **8.28** (much higher than TR5's 1.54 — fresh
wins sit deeper in the ranking, so **B20 was essential here**, unlike TR5 where it was
useless). 42 unknown-name failures encountered live (retrieval mis-fires on fresh names).
2 transient flakes → NEEDS_REVIEW.

## 9. Attribution

`scripts/tr6_apply_attribution.py` (SX4; controls run at B5; registry guarantees freshness):

| class | count |
|---|---|
| NO_WIN_UNDER_BUDGET | 114 |
| **FRESH_TRUE_DELTA** | **18** |
| BASELINE_DUPLICATE | 3 |
| NEEDS_REVIEW (flake) | 2 |

**18 credited fresh true deltas** — Set 5, Multiset 9, Finset 2, List 1, Nat 1. 3 baseline
dups (Multiset.coe_disjoint, Multiset.disjoint_left, mem_lowerBounds_iff_subset_Ici — a bare
control solves them). Winning families: d2_simp_aesop 9, d1_simp_lemma 3, d1_tauto 3,
d2_rw_aesop 1, d1_exact 1, def_unfold_simp 1.

**Key discovery — the disjoint_left bridge generalizes:** `simp [Set.disjoint_left] <;> aesop`
closes 4 fresh Set goals, and the analogous **`simp [Multiset.disjoint_left] [<;> aesop]`**
closes 5 fresh Multiset goals (disjoint_add_left/cons_left/add_right/singleton/zero). The
RC4B mechanism is namespace-parametric, not Set-specific.

## 10. Ranker fresh-frontier performance

`scripts/tr6_analyze_ranker_fresh_performance.py` → **`RANKER_GENERALIZES_TO_FRESH_FRONTIER`**.

- credited wins by budget: B5 9, B10 12, B20 **18**; success/probe **0.0073** (TR5 ref 0.0161
  — lower because fresh wins are deeper and the no-win frontier is large).
- no-win rate 0.83; mean first-success rank 8.28.
- **by namespace (searched → credited):** Multiset 22→9, Set 21→5, Finset 30→2, List 29→1,
  Nat 14→1; Order-family (AntitoneOn/MonotoneOn/IsGLB/IsLUB) 6→0; '' (root) 10→0.

The ranker surfaces real wins in **five** namespaces including two (Multiset, Nat) where TR4
had zero positives — it does not merely reproduce Set. Order-family and root-level lemmas
yielded 0 (the next frontier).

## 11. RC4B / RC4C fresh evidence

`scripts/tr6_rc4b_rc4c_fresh_evidence.py`.

- **RC4B (`*.disjoint_left` bridge) → `READY_FOR_LITERAL_VALIDATION_WITH_FRESH_SUPPORT`:**
  **8 fresh wins** across Set (4) and Multiset (4, via `Multiset.disjoint_left`). Off-gate
  risk low (single named-lemma rewrite on disjoint-shaped goals). The fresh holdouts make
  this materially stronger than TR5's reproduction-only evidence; validation should test both
  the `Set.` and `Multiset.` bridge variants.
- **RC4C (`d2_simp_aesop`) → `READY_FOR_LITERAL_VALIDATION_WITH_FRESH_SUPPORT`:** **9 fresh
  wins** (Set 4, Multiset 3, Finset 1, List 1); large overlap with RC4B (the disjoint wins are
  `simp [L] <;> aesop`). Source-specific risk medium (depends on the retrieved L); SX4
  PRODUCTION_SUBSUMED guard applied.

Both are **fresh-supported**, not reproduction-only — a real upgrade over TR5. No RC4B/RC4C
artifact created (per instructions).

## 12. Training export

`scripts/tr6_export_training_data.py` — **2,458 program-level examples** (TR4 schema; TR4/TR5
not overwritten). 21 success / **18 credit** positives. **New positive namespaces vs TR4
(Set/Finset/List): Multiset and Nat** — the first credited positives in those namespaces,
directly targeting the by-namespace gap.

## 13. Optional retrain (exploratory)

`scripts/tr6_retrain_ranker_with_fresh_data.py` (TR4 model NOT replaced):

| set | n | pos | pos-ns | PR-AUC(thm) | PR-AUC(ns) | top-5 |
|---|---|---|---|---|---|---|
| TR4 only | 4,737 | 23 | 4 | **0.522** | 0.0079 | 1.00 |
| TR4 + TR5 | 5,542 | 36 | 4 | 0.300 | 0.0055 | 0.77 |
| **TR4 + TR6** | 7,195 | 44 | **7** | **0.519** | **0.0095** | 0.85 |
| TR4 + TR5 + TR6 | 8,000 | 57 | 7 | 0.372 | 0.0100 | 0.91 |

Two findings: (1) unlike the TR5 union (which **dropped** by-theorem PR-AUC to 0.30 by
re-probing the same 92 theorems), the **TR6 union preserves it (0.519 ≈ 0.522)** because TR6
adds genuinely new theorems/namespaces; (2) **by-namespace PR-AUC rises 0.0079 → 0.0095** —
the first improvement of the generalization metric (TR5 lowered it). It is still tiny: 18
fresh-namespace positives is a start, not a cure. Exploratory; model not promoted.

## 14. Decision

- **`RANKER_GENERALIZES_TO_FRESH_FRONTIER`** — 18 fresh true deltas across 5 namespaces from
  ranker-guided search on a never-seen frontier.
- **`FRESH_TRUE_DELTA_FOUND`** (18) and **`NONSET_POSITIVES_FOUND`** (13: Multiset 9, Finset
  2, List 1, Nat 1).
- **`RC4B_READY_FOR_LITERAL_VALIDATION_WITH_FRESH_SUPPORT`** (8 fresh, incl. the new
  `Multiset.disjoint_left` bridge) and **`RC4C_READY_FOR_LITERAL_VALIDATION_WITH_FRESH_SUPPORT`**
  (9 fresh).

**Bottleneck answer (Q5):** the next bottleneck is **candidate validation**, not retrieval or
frontier size — there is now ample fresh RC4B/RC4C evidence to validate, and a large no-win
residual (Order-family, root lemmas, deep Nat) that retrieval-aware depth does not yet crack.

**Caveats:** fresh wins are deep (mean rank 8.28 — B20 needed); Order-family/root namespaces
yielded 0; by-namespace ranker transfer is still weak (PR-AUC 0.0095) — improved but not
solved; 42 unknown-name retrieval mis-fires indicate retrieval index scope is the second lever.

## 15. Next steps

- **Validate RC4B** (`Set.disjoint_left` + `Multiset.disjoint_left`) and **RC4C**
  (`d2_simp_aesop`) with the RC4A literal-RC2⊕candidate harness on these fresh holdouts +
  further disjoint holdouts — this is the now-unblocked path to an RC4 candidate.
- **Improve ranker namespace generalization** by folding the 18 fresh positives in and mining
  more Multiset/Nat/Order positives (the gap is data, and TR6 shows fresh-namespace data helps).
- **Order-family / root lemmas** need a different lever (these had retrieval coverage but 0
  wins) — likely a structural/order tactic battery, not retrieval-aware simp.

## 16. Protected-file confirmation

- RC1 wrapper, RC2-release wrapper, NS24 router — **untouched** (`git diff --stat HEAD` empty).
  NS9 / REL / RC reports / TR1–TR5 / SF / RC4A datasets unchanged.
- **No production routing change, no RC4/RC4B/RC4C release, ranker & candidates not promoted,
  no README update, no commit.** Artifacts under `project/evolve/experiments/tr6/` &
  `project/evolve/reports/tr6/`, scripts `scripts/tr6_*.py`. Live LeanDojo used for RC2
  confirmation + B5/B10/B20.
