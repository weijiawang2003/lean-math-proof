# TR5 — Ranker-Guided Live Search and RC4B/RC4C Evidence Collection

**Decisions:** `RANKER_LIVE_CONFIRMED` · `RC4B_READY_FOR_VALIDATION` ·
`RC4C_READY_FOR_VALIDATION`.

Exploratory; **ranker not promoted**, no production change, no RC4 release, no commit.

---

## 1. Executive summary

- **Target pool:** 92 confirmed-RC2-failure theorems (TR3's frontier), tagged A–G, all
  with a TR4 ranker score; RC4B (Set.disjoint_left) and RC4C (d2_simp_aesop) targets
  flagged.
- **RC2 confirmation:** 92/92 `CONFIRMED_RC2_FAILURE`, all **reused** from TR3's
  identical-config run (0 fresh literal-RC2 runs needed).
- **Live probes:** **B5 407 programs + 364 controls; B10 393 programs; B20 skipped**
  (0 yield by construction — see §6). Total **800 ranked programs** live.
- **Wins:** **13 live successes** — 12 at B5 (all rank 1), 1 at B10 (Finset.mem_disjUnion,
  rank 8). **100 % recovery of the 13 known TR3 successes.**
- **True deltas:** **7 TRUE_RANKER_DELTA + 5 TRUE_RC4A_REPRODUCTION = 12 credited**,
  1 BASELINE_DUPLICATE (Prop.compl_singleton), 79 NO_WIN_UNDER_BUDGET — **identical
  credited set to TR3** (12/13).
- **vs TR3 full battery:** TR3 ran **3,926** programs for 13 wins; TR5 ran **800** —
  **79.5 % probe reduction**, success/probe **0.0163 vs 0.0033 (≈5×)**, mean live
  first-success rank **1.54**, **467 unknown-name failures avoided**.
- **RC4B (Set.disjoint_left):** 3 live true wins → `READY_FOR_RC4B_VALIDATION`.
- **RC4C (d2_simp_aesop):** 3 live true wins → `READY_FOR_RC4C_VALIDATION`.
- **Training export:** 805 program-level examples (13 success / 12 credit positives).
- **Optional retrain:** naive TR4+TR5 union **lowers** by-theorem PR-AUC (0.52→0.30) —
  exploratory; fresh-namespace data, not re-probes, is the lever.

**The TR4 offline probe-reduction result survives live execution.** The honest limit: TR5
found **0 fresh wins** — it confirms the ranker recovers TR3's wins ~5× more efficiently,
not that it expands theorem coverage.

## 2. Motivation

TR4 showed offline that the HGB ranker recovers 100 % of TR3 successes at top-5/theorem
(88.8 % probe reduction), leakage-free by theorem but collapsing by namespace. TR5 tests
that claim under real LeanDojo execution and collects live evidence for the next two RC4
candidate families.

## 3. Target pool

92 deduped theorems (`scripts/tr5_build_target_pool.py`):

- **By category:** A_tr3_winner 13, D_rc4c_d2aesop 54, E_high_confidence 8,
  F_high_uncertainty 5, G_underrep_namespace 12.
- **By namespace:** Set 31, Nat 28, Finset 18, List 8, Multiset 4, Prop/Eq/Function 1 each.
- **Known/fresh:** 13 known TR3 winners (incl. the 5 RC4A def-unfold wins) + 79 open
  (no-win) theorems. RC4B targets: 3; RC4C-tagged: 57.

The pool is TR3's confirmed-failure frontier; TR2/TR3 already showed no *fresh* failures
remain on these sources, so TR5 is a recovery/efficiency test, not a new-frontier sweep.

## 4. RC2 confirmation

`scripts/tr5_confirm_rc2.py` — **92/92 CONFIRMED_RC2_FAILURE**, 100 % reused from TR3's
identical-config literal-RC2 run (rc2_release wrapper, ns24 router, hybrid_evolved,
top-k 8, max-steps 8). 0 RC2_SOLVED, 0 flakes, 0 path errors, **0 live runs**. Only
confirmed failures are TRUE_DELTA-eligible.

## 5. Ranked program plan

`scripts/tr5_build_ranked_program_plan.py` — for each target, the TR3-grammar candidate
programs (4,377 total) were scored with the **full-data TR4 HGB ranker** (featurized
identically to `tr4_featurize_programs.py`) plus the heuristic, and ranked. Budgets
B1/B3/B5/B10/B20 = 92/276/460/920/1840 program-slots. Sanity: the known winning tactic
ranks **rank 1 for 12/13** theorems and rank 8 for Finset.mem_disjUnion → predicts B5
recovers 12, B10 recovers 13. Families incl. def_unfold_simp, d1_simp_lemma,
d2_simp_aesop (RC4C), and the Set.disjoint_left bridge (RC4B).

## 6. Live B5 / B10 / B20 search

`scripts/tr5_run_ranked_live_search.py` (serialized; one Dojo/theorem; hard-timeout
worker; 4 bare controls then top-B ranked programs; stop after first success;
per-theorem checkpoint).

| budget | theorems run | ranked programs | controls | new wins | notes |
|---|---|---|---|---|---|
| **B5** | 92 | 407 | 364 | **12** (all rank 1) | 1 transient worker crash, re-run → clean NO_WIN |
| **B10** | 80 unsolved | 393 | 0 | **+1** (Finset.mem_disjUnion, rank 8) | |
| **B20** | — | **0 (skipped)** | — | 0 | see note |

- **First-success ranks:** {1: 12, 8: 1} → mean **1.54**.
- **B20 skipped — not a silent cap:** the ranked programs are a strict subset/re-ordering
  of TR3's full battery, and all 13 TR3 wins are already recovered by B10. Ranks 11–20
  therefore cannot yield a credited win TR3 didn't already find; running ~790 more probes
  for a provable 0 yield was not worth the compute. Documented here explicitly.
- **One flake:** `Set.ssubset_singleton_iff` first returned "worker no output (rc=1)"; a
  single re-run was live and clean (controls + all 5 ranked programs fail) → genuine
  `NO_WIN_UNDER_BUDGET`, patched into the results.

## 7. Attribution

`scripts/tr5_apply_attribution.py` (SX4 discipline — every win beats literal RC2; controls
guard BASELINE_DUPLICATE):

| class | count |
|---|---|
| NO_WIN_UNDER_BUDGET | 79 |
| **TRUE_RANKER_DELTA** | **7** |
| **TRUE_RC4A_REPRODUCTION** | **5** |
| BASELINE_DUPLICATE | 1 |

- **Credited total: 12** (7 + 5) — **exactly TR3's credited set**.
- **TRUE_RANKER_DELTA (7):** List.toFinset.ext_iff, List.toFinset_eq, Set.compl_union_self
  (d1_simp_lemma); Set.disjoint_singleton_left (d1, Set.disjoint_left);
  Set.Nonempty.subset_pair_iff_eq, Set.disjoint_iff_forall_ne, Set.disjoint_right
  (d2_simp_aesop).
- **TRUE_RC4A_REPRODUCTION (5):** Set.{mono,antitone,strictMono,strictAnti}On_iff_*
  (`simp [Def, DefOn]`) + Finset.mem_disjUnion (`simp [Finset.disjUnion]`).
- **BASELINE_DUPLICATE (1):** Prop.compl_singleton — bare `aesop` solves it (matches TR3).
- **No false credit:** controls were run in-worker for every win; the 12 credited all had
  failing controls and confirmed literal-RC2 failures.

## 8. Comparison to TR3 full battery

`scripts/tr5_compare_to_tr3.py` → **`RANKER_LIVE_CONFIRMED`**.

| metric | TR3 full battery | TR5 ranker-guided |
|---|---|---|
| programs attempted | 3,926 | **800** (B5 407 + B10 393) |
| controls | (in-battery) | 364 |
| known successes recovered | 13/13 | **B5 12/13, B10 13/13** |
| credited wins | 12 | **12** |
| missed known wins | — | **0** |
| new wins | — | 0 |
| success / probe | 0.0033 | **0.0163 (≈5×)** |
| mean live first-success rank | 16.46 (orig order) | **1.54** |
| unknown-name failures | 467 | 0 (in top-5) → **467 avoided** |

- **Probe reduction: 79.5 %** at **100 % recovery** (B10). Criteria met: B5 ≥70 % (92 %),
  B10 ≥85 % (100 %), probe reduction ≥60 %, no false-credit.
- **Honest caveat — rank-1 false positives = 80.** On the 80 theorems with *no* win in the
  whole TR3 battery, the ranker's rank-1 program necessarily fails. This is the base rate
  of unsolvable theorems in the pool, not a ranking error: where a win exists, the ranker
  put it at rank 1 in 12/13 cases. Rank-1 precision is high *conditional on solvability*,
  but the pool is mostly PROOF_DEPTH_GAP.

## 9. RC4B / RC4C evidence

`scripts/tr5_candidate_evidence_analysis.py`.

**RC4B — `Set.disjoint_left` bridge → `READY_FOR_RC4B_VALIDATION`.** 3 live true wins
(Set.disjoint_singleton_left `simp [Set.disjoint_left]`; Set.disjoint_iff_forall_ne &
Set.disjoint_right `simp [Set.disjoint_left] <;> aesop`). All 3 reproduce TR3; **0 fresh**.
Off-gate risk **low** — single named-lemma rewrite, fires only when the lemma is retrieved.
Suggested candidate: narrow allowlist adding `simp [Set.disjoint_left]` (+ the d2 variant)
to the Set route battery, off-by-default, additive over RC2 (SET_ITE_SIMP / RC4A pattern).

**RC4C — `d2_simp_aesop` (`simp [L] <;> aesop`) → `READY_FOR_RC4C_VALIDATION`.** 3 live
true wins (Set.Nonempty.subset_pair_iff_eq, Set.disjoint_iff_forall_ne, Set.disjoint_right).
All reproduce TR3; **0 fresh**. Source-specific risk **medium**: credit is the `simp [L]`
enabling step (SX4 PRODUCTION_SUBSUMED guard already applied — RC2's best-first search does
not reach the simp[L]-advanced state). Note RC4B/RC4C **overlap** on disjoint_iff_forall_ne
and disjoint_right (both are `simp [Set.disjoint_left] <;> aesop`).

**Both READY decisions mean "enough live-verified wins to warrant a separate
literal-RC2⊕candidate validation"** — NOT that the candidates are validated. Because all
wins are TR3 reproductions (0 fresh), the validation itself must source fresh
Disjoint/Set holdouts (as SET_ITE_SIMP and RC4A did). No RC4B/RC4C artifact created here.

## 10. Training export

`scripts/tr5_export_training_data.py` — **805 program-level examples** (one attempted
program = one row, TR4 schema; TR4 data not overwritten). 13 success / **12 credit**
positives; 80 RANKER_FALSE_POSITIVE (rank-1 fails), 712 NO_WIN. Labels:
TRUE_RC4A_REPRODUCTION 5, TRUE_RC4B_EVIDENCE 3, TRUE_RANKER_DELTA 3, TRUE_RC4C_EVIDENCE 1,
BASELINE_DUPLICATE 1. The 12 live-verified credit positives are the scarce class (TR4 had
22); they confirm rather than expand the label set (same theorems/namespaces).

## 11. Optional retrain

`scripts/tr5_retrain_ranker_with_live_data.py` (EXPLORATORY — TR4 model not replaced):

| set | n | pos | PR-AUC(thm) | PR-AUC(ns) | PR-AUC(cluster) | top5 rec |
|---|---|---|---|---|---|---|
| TR4 only | 4,737 | 23 | **0.522** | 0.008 | 0.477 | 1.00 |
| TR4 + TR5 | 5,542 | 36 | 0.300 | 0.006 | 0.516 | 0.77 |

Naive union **drops** by-theorem PR-AUC (Δ **−0.222**) and top-5 recovery, nudges
by-cluster up (+0.04), leaves by-namespace collapsed. Reason: TR5 rows re-probe the **same
92 theorems** with a re-ordered subset + many rank-1 failures — redundant/conflicting
within-group signal, not new coverage. **Conclusion: don't naively union; fresh-namespace
positives are the lever** (reinforces TR4).

## 12. Decision

- **`RANKER_LIVE_CONFIRMED`** — B5 recovered 12/13 known successes (all rank 1), B10 13/13,
  at 79.5 % probe reduction and ≈5× success/probe vs the TR3 full battery, with 0
  false-credit. The TR4 offline result holds live, **within seen namespaces**.
- **`RC4B_READY_FOR_VALIDATION`** — 3 live Set.disjoint_left wins, low off-gate risk.
- **`RC4C_READY_FOR_VALIDATION`** — 3 live d2_simp_aesop wins, SX4-clean.

**Scope caveats:** 0 fresh wins (efficiency, not coverage); all RC4B/RC4C evidence is TR3
reproduction; Nat (28 targets) yielded 0 (consistent with TR4's namespace gap); by-namespace
transfer still unestablished.

## 13. Next steps

- **RC4B validation:** run the RC4A-style literal-RC2⊕candidate harness on a narrow
  `simp [Set.disjoint_left]` allowlist with **fresh** Disjoint/Set holdouts (the 3 wins are
  in-sample).
- **RC4C validation:** same harness for `simp [L] <;> aesop`, sourcing fresh d2 holdouts;
  watch the source-specific (retrieved-L) dependence.
- **Ranker-guided frontier expansion:** the confirmed ~80 % probe saving makes a *fresh
  multi-namespace* discovery sweep affordable — that is where new wins (and the
  by-namespace generalization data the ranker lacks) will come from, not re-probing these 92.
- **Do not** naively retrain on TR5; collect fresh-namespace positives first.

## 14. Protected-file confirmation

- RC1 wrapper, RC2-release wrapper, NS24 router — **untouched** (`git diff --stat HEAD`
  empty). NS9 / REL / RC reports / TR1·TR2·SF·TR source datasets unchanged.
- **No production routing change, no RC4 release, no RC4B/RC4C artifact, ranker not
  promoted, no README update, no commit.** All TR5 artifacts under
  `project/evolve/experiments/tr5/` & `project/evolve/reports/tr5/`, scripts
  `scripts/tr5_*.py`. Live LeanDojo used for B5/B10 only.
