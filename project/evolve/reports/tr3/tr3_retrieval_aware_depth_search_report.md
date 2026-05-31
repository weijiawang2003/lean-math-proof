# TR3 — Retrieval-Aware Depth Search at Scale

**Decision: `FOUND_RC_CANDIDATE_FAMILY`** (3 families meet the ≥2-TRUE_DELTA bar)
**— with `RETRIEVAL_DEPTH_SIGNAL_FOUND` as the broader finding.** Candidate families
are *identified, not promoted*; each still needs the standard separate
literal-RC2⊕candidate validation (off-gate / floors / determinism) before promotion.

---

## 1. Executive summary

- **Case pool:** 150 (SF5 depth-gap A=15, SF5 retrieval/routing B=5, SF4-only
  confirmed failures C=7, SF1 frontier D=24, multi-namespace expansion E=99).
- **Confirmed literal-RC2 failures:** **92 / 150** (44 reused from SF4/TR2 at the
  identical config, 106 run live; well above the ≥50 target). 65 are *new* failures
  from the catalog expansion (Nat 28, Finset 18, List 8, Multiset 4, Set+others).
- **Retrieval index:** 10 790 lemmas, **100 %** statement coverage (SF5 index reused +
  broader Nat/List/Multiset/Order source scan).
- **Depth programs run:** 4 377 gated programs (+4 controls/target) live across 92/92
  theorems; outcomes 3 411 proof_failed / 495 unknown-name / 13 success / 7
  max-recursion / 451 skipped-after-win.
- **TRUE_DELTA over literal RC2: 12** (`every_win_over_literal_rc2 = True`):
  - **9 `TRUE_RETRIEVAL_ONLY_DELTA`** (depth-1 `simp [L]` / definitional unfold)
  - **3 `TRUE_RETRIEVAL_DEPTH_DELTA`** (depth-2 `simp [L] <;> aesop`)
  - **0 `TRUE_DEPTH_ONLY_DELTA`** (no lemma-free program ever beat RC2 — depth without
    retrieval adds nothing here)
- **79 `PROOF_DEPTH_GAP`**, **1 `BASELINE_DUPLICATE`** (`Prop.compl_singleton`, bare
  `aesop` — the SF4-noted routing artifact, correctly denied credit).
- **Best families (≥2 TRUE_DELTA → rc_candidate):** `def_unfold_simp` (5),
  `d1_simp_lemma` (4), `d2_simp_aesop` (3). **Best lemma:** `Set.disjoint_left`
  (3 wins — a genuinely reusable Set-disjointness bridge).
- **Decision:** `FOUND_RC_CANDIDATE_FAMILY` / `RETRIEVAL_DEPTH_SIGNAL_FOUND`.

The headline advance over SF5: where SF5 found 5 single-shot retrieval wins on 20
targets and called the rest a proof-depth gap, TR3 scales to 92 confirmed failures
and converts **12** into wins — including **7 new** beyond SF5 — and shows the win
mechanism is *retrieval* (depth-1 `simp [L]` and depth-2 `simp [L] <;> aesop`), never
lemma-free depth.

---

## 2. Motivation

Three prior results converged: SF4 found 0 cheap deltas over 27 confirmed failures;
TR2 exhausted the *fresh frontier* (47/47 already labelled); SF5 showed the 20
missing-bridge targets are existing Mathlib theorems needing *retrieval-aware
multi-step depth*, not synthesis. TR3 tests that thesis at scale: pair existing-lemma
retrieval with bounded depth-2/3 proof programs and judge every win against literal
RC2 via SX4-style attribution. A second motivation was to test the TR2
frontier-exhaustion claim against a previously-unprobed source (the
`discovered_theorems` catalog) — which turned out **not** exhausted (65 new failures).

---

## 3. Case pool and RC2 confirmation

- Pool deduped by `full_name`; 0 unresolved (all had file paths).
- Confirmation reused SF4/TR2 identical-config results (rc2_release wrapper, ns24
  router, hybrid_evolved, top-k 8, max-steps 8, repaired finished-key) and ran the
  rest live in 9 chunked, checkpointed `eval_rollout_all` workers.
- **92 CONFIRMED_RC2_FAILURE / 58 RC2_SOLVED.** Confirmed failures by namespace:
  Set 31, Nat 28, Finset 18, List 8, Multiset 4, (Eq/Function/Prop 1 each).
- Most expansion cases are "easy" and RC2-solved as expected, but the catalog still
  surfaced 65 fresh confirmed failures — frontier **not** exhausted via this source.

Artifacts: `cases/tr3_case_pool.jsonl`, `out/tr3_case_pool_summary.*`,
`out/tr3_rc2_confirmation.*`.

## 4. Retrieval index and quality

Index = 10 790 declarations (reused SF5 index + broader Nat/List/Multiset/Order
source scan), 100 % with statement text, incl. `def`/`abbrev` (for definitional
unfolds). Retrieval per confirmed failure: top-20 by lexical TF-IDF + namespace/path
proximity + feature overlap + name-pattern similarity, injecting SF5 winning lemmas
and a goal-driven definition channel. Quality is high where it matters — e.g.
`Set.disjoint_left` surfaces for every Set-disjointness target, `Set.subset_pair_iff_eq`
for the pair targets. Limitation: 495 `unknown_name` outcomes — the broad index
includes lemmas not in scope at a target's source position (a scope-aware index is the
clear next improvement).

Artifacts: `out/tr3_retrieval_index*.{jsonl,json,md}`, `out/tr3_retrieval_results.*`.

## 5. Depth program generation

Gated depth-1/2/3 programs, ≤10 lemmas & ≤60 programs/target, deterministic order, no
source-specific scripts. Gates: Set-eq→`ext`/antisymm; Set-iff→`constructor`/`intro`;
Nat/arith→`omega`/`nlinarith`; Multiset.toFinset→toFinset simp. Families span
definitional unfold (`simp [Def,DefOn]`), depth-1 retrieval (`exact/simpa/simp/rw [L]`),
lemma-free depth-only controls, depth-2 retrieval (`simp [L] <;> aesop`,
`rw [L] <;> simp_all`, `ext x <;> simp [L]`, …), and depth-3 conservative composites.
4 377 programs over 92 targets (~48/target).

Artifacts: `out/tr3_depth_program_plan.*`.

## 6. Live search results

92/92 theorems opened live (serialized worker, per-tactic SIGALRM, OS hard timeout,
per-theorem checkpoint/resume, `--stop-after-win`). 12 candidate-win targets, 1
baseline-duplicate, 79 no-win. Program-level: 13 successes, 3 411 proof-failed, 495
unknown-name (out-of-scope lemmas), 7 max-recursion, 0 parse errors, 0 open flakes,
451 skipped-after-win. Win depth distribution: 9 at depth-1, 3 at depth-2.

Artifacts: `out/tr3_depth_program_results.*`, `out/depth_run.log`,
`out/depth_program_checkpoint.json`.

## 7. Attribution

Every credited win is over a CONFIRMED literal-RC2 failure; controls
(`simp/simp_all/aesop/classical <;> aesop`) guard BASELINE_DUPLICATE; non-failures
guard PRODUCTION_SUBSUMED. Single-shot tactics on the initial state cannot be
subsumed by RC2's best-first search (the SX3 over-credit class does not apply).

| class | n |
|---|---|
| `TRUE_RETRIEVAL_ONLY_DELTA` | 9 |
| `TRUE_RETRIEVAL_DEPTH_DELTA` | 3 |
| `TRUE_DEPTH_ONLY_DELTA` | 0 |
| `PROOF_DEPTH_GAP` | 79 |
| `BASELINE_DUPLICATE` | 1 |

**TRUE_DELTA by namespace:** Set 9, List 2, Finset 1, **Nat 0**. The 12 wins
(`every_win_over_literal_rc2 = True`):

| target | class | program | depth |
|---|---|---|---|
| Set.monotoneOn_iff_monotone | retrieval_only | `simp [Monotone, MonotoneOn]` | 1 |
| Set.antitoneOn_iff_antitone | retrieval_only | `simp [Antitone, AntitoneOn]` | 1 |
| Set.strictMonoOn_iff_strictMono | retrieval_only | `simp [StrictMono, StrictMonoOn]` | 1 |
| Set.strictAntiOn_iff_strictAnti | retrieval_only | `simp [StrictAnti, StrictAntiOn]` | 1 |
| Finset.mem_disjUnion | retrieval_only | `simp [Finset.disjUnion]` | 1 |
| List.toFinset.ext_iff | retrieval_only | `simp [Finset.ext_iff]` | 1 |
| List.toFinset_eq | retrieval_only | `simp [Multiset.toFinset_eq]` | 1 |
| Set.compl_union_self | retrieval_only | `simp [Set.union_eq_compl_compl_inter_compl]` | 1 |
| Set.disjoint_singleton_left | retrieval_only | `simp [Set.disjoint_left]` | 1 |
| Set.Nonempty.subset_pair_iff_eq | retrieval_depth | `simp [Set.subset_pair_iff_eq] <;> aesop` | 2 |
| Set.disjoint_iff_forall_ne | retrieval_depth | `simp [Set.disjoint_left] <;> aesop` | 2 |
| Set.disjoint_right | retrieval_depth | `simp [Set.disjoint_left] <;> aesop` | 2 |

5 reproduce SF5 (the 4 definitional unfolds + the pair-routing case); **7 are new**
(Finset/List/Set disjointness + complement). The 79 PROOF_DEPTH_GAP are dominated by
Nat (28) and Set (22) — Nat arithmetic failures get no retrieval traction
(`omega`/`nlinarith` never closed one), confirming a genuine multi-step depth gap
distinct from the retrieval-addressable Set/Finset/List cluster.

Artifacts: `out/tr3_attribution.*`.

## 8. Family and lemma analysis

**Promotion-eligible families (≥2 TRUE_DELTA, advisory — NOT promoted):**

| family | depth | tried | TRUE_DELTA | recommendation |
|---|---|---|---|---|
| `def_unfold_simp` | 1 | 49 | 5 | rc_candidate |
| `d1_simp_lemma` | 1 | 510 | 4 | rc_candidate |
| `d2_simp_aesop` | 2 | 243 | 3 | rc_candidate |

**Useful lemmas (≥1 win):** `Set.disjoint_left` (3 — the standout reusable bridge),
the predicate `def`s `Monotone/MonotoneOn/Antitone/AntitoneOn/StrictMono(On)/StrictAnti(On)`,
`Set.subset_pair_iff_eq`, `Finset.disjUnion`, `Finset.ext_iff`, `Multiset.toFinset_eq`,
`Set.union_eq_compl_compl_inter_compl`. 14 distinct useful lemmas total; the rest of
the retrieved pool is noise for these targets.

Note the `d1_simp_lemma` / `d2_simp_aesop` families fire broadly (510 / 243 programs)
— they are **not** narrowly gated, so an off-gate/floors evaluation is mandatory
before any of them could be promoted (see §10). `def_unfold_simp` is the cleanest
narrow candidate (49 fires, 5 wins, all `<pred>On_iff_<pred>` shaped).

Artifacts: `out/tr3_family_analysis.*`, `out/tr3_lemma_usage_analysis.*`.

## 9. Training export

92 additive examples (12 positive TRUE_DELTA, 80 negative depth-gap/baseline),
TR1-compatible schema, `source_artifact: tr3_depth_search`. Namespaces: Set/Nat/
Finset/List/Multiset. Does **not** overwrite TR1/TR2/SF5. A manifest stitches
TR1+SF5+TR3 (label spaces differ → consume additively / channel-tagged, not merged
blindly). `helps_tr4 = True` (12 verified positives across 3 namespaces give TR4 real
retrieval-depth signal to scale on).

Artifacts: `data/tr3_training_examples.jsonl`,
`data/tr1_tr2_sf5_tr3_dataset_manifest.json`, `data/tr3_training_delta_summary.*`.

### Optional retrain (Part 11, EXPLORATORY)

Small LogisticRegression (TR1 family) over the union label space, leave-one-out
macro-F1: **TR1 0.484 → TR1+SF5 0.463 → TR1+SF5+TR3 0.567**. Adding TR3 raises macro-F1
(+0.08 over TR1, +0.10 over TR1+SF5) — the retrieval-depth labels carry signal. This
is exploratory only (mixed label spaces, tiny grouped folds); not a production router.
Artifacts: `out/tr3_retrained_router_results.*`.

## 10. Decision

**`FOUND_RC_CANDIDATE_FAMILY`** — three families clear the ≥2-TRUE_DELTA-over-literal-RC2
bar (`def_unfold_simp` 5, `d1_simp_lemma` 4, `d2_simp_aesop` 3), all SX4-survived with
`every_win_over_literal_rc2 = True`, and a reusable bridge lemma (`Set.disjoint_left`)
recurs across 3 wins. The broader **`RETRIEVAL_DEPTH_SIGNAL_FOUND`** also holds:
retrieval is the win mechanism (0 lemma-free depth-only wins), at both depth-1 and
depth-2.

What is **not** yet established (and blocks promotion): the remaining promotion gates
— **0 off-gate**, **floors preserved**, and a **determinism re-run** — were *not*
measured here (TR3 ran only on confirmed failures, no floor/regression set). The
broad-firing families (`d1_simp_lemma`, `d2_simp_aesop`) especially require an
off-gate audit. So these are *candidate* families, not RC components.

## 11. Next steps

Because a retrieval-depth signal and candidate families were found:

1. **Validate each candidate family separately** through a literal RC2⊕candidate run
   on the canonical floors (demo_v1 / medium / large) + a negative-control namespace,
   exactly as SET_ITE_SIMP→RC2 was validated — measuring credited delta, 0 off-gate,
   floors preserved, deterministic hash. Start with the narrow `def_unfold_simp`
   (`<pred>On_iff_<pred>` gate) and the `Set.disjoint_left` simp lemma, which are the
   cleanest/narrowest.
2. **Train a retrieval-aware router (TR4)** on TR1+SF5+TR3 (additive; the retrain shows
   positive lift) to *predict when to fire* a retrieval/def-unfold/depth-2 action,
   rather than firing the broad battery unconditionally.
3. **For the 79 PROOF_DEPTH_GAP** (esp. 28 Nat arithmetic) where retrieval gives no
   traction: a bounded search-depth increase or proof-state transition modeling — a
   separate depth experiment, not retrieval.
4. **Scope-aware retrieval index** to cut the 495 unknown-name outcomes (resolve
   imports at each target's source position).

No promotion, no RC4, no production change in TR3.

## 12. Protected-file confirmation

- `project/evolve/experiments/rc1/rc1_production_wrapper.json` — **untouched**.
- `project/evolve/experiments/rc2_release/rc2_production_wrapper.json` — **untouched**
  (read-only as the RC2 strategy-config for confirmation).
- `project/evolve/routing/ns24_router.json` — **untouched** (read-only route-config).
- NS9 genome/checkpoints, REL1/RC1/RC2 reports, TR1/TR2 datasets — **untouched**
  (TR3 wrote a separate additive `tr3_training_examples.jsonl`).
- No production routing changed; no RC4; no README update; no commit.
  All artifacts under `project/evolve/experiments/tr3/` & `project/evolve/reports/tr3/`,
  scripts `scripts/tr3_*.py`. `git diff --stat HEAD` over the three protected wrappers
  is empty.
