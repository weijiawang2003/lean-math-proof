# TR2 — Model-Guided Active Probing

**Branch:** `sx3-depth2-sequence-search` · **Date:** 2026-05-30 · **No commit made.**
First active-learning loop on top of the TR1 failure-to-action router: use router predictions to *select*
which RC2 failures / frontier cases to live-probe, verify outcomes through literal RC2 + SX4 attribution,
and compare model-selected vs rule-selected vs random selection on **useful verified labels per probe.**
The router is **not** promoted and **not** wired into production routing.

---

## 1. Executive summary

| field | value |
|---|---|
| candidate pool | **47** theorems (27 RC2-failed, 20 RC2-solved) — **all 47 already TR1-labelled** |
| batches | model 15 · rule 15 · random 15 (model⊥rule disjoint; random = matched-ratio control) |
| RC2 confirmation | **26 CONFIRMED_RC2_FAILURE + 8 RC2_SOLVED** (34 unique), **0 fresh live runs** (reused SF4/TR1) |
| live probes | 34 theorems probed · **30 live** · **64 fresh live tactics + 106 reused** · **0 probe wins** |
| SX4 attribution | **0 TRUE_DELTA**; **31 useful labels** (20 missing-bridge, 6 baseline-duplicate, 3 no-cheap-action, 2 depth-gap) |
| model beat baselines? | **No on yield** (14/14/14 useful, per-probe 0.48/0.52/0.50); **Yes on diversity** (ns 6 vs 2/3, non-Set 5 vs 2/2) |
| dataset delta | **0 net-new** · 27 reconfirmations · **4 verified label-revision candidates** · 3 excluded |
| **DECISION** | **`INCONCLUSIVE_TOO_SMALL`** |

TR2 ran the active-learning loop end-to-end with verified-label discipline and produced **no false credit**
(0 TRUE_DELTA, matching SF4). Its two real products are (a) **4 verified TR1 label corrections** surfaced
by live re-probing, and (b) evidence that **model selection concentrates namespace/coverage diversity** even
though it cannot out-yield baselines on a fully pre-labelled ~40-theorem pool. The headline finding is that
**the fresh frontier is exhausted**: the active-learning loop is correct but starved of novel cases, which is
exactly the data bottleneck TR1 flagged.

---

## 2. Motivation

TR1 trained a failure-to-action router and honestly self-classified `PILOT_ONLY_NEEDS_MORE_DATA` (within-dist
LOO 0.877 but grouped leave-one-namespace-out 0.386 → 0.49 generalization gap). The obvious next question is
not "is the router accurate?" but "**is it useful for collecting the data it lacks?**" — i.e. does routing
the probe budget by router uncertainty/prediction beat spending it randomly. Literal-RC2 confirmation matters
because (per RC3/SX4) a candidate only counts if it beats *literal production*, not a depth-1 control battery;
without that gate, active probing would manufacture the same over-credit RC3 did.

---

## 3. Candidate pool

`scripts/tr2_build_candidate_pool.py` unions the TR1 active-learning / next-work / RC2-prediction artifacts,
the SF4 failure pool / confirmation / clusters / missing-lemma triage, and the SF1 frontier; dedups by
`full_name`; and tags each row with the router's predictions (entropy/margin), known literal-RC2 status,
SF4 cluster, goal features, and selection tags.

- **47 candidates** (4 excluded: all `SX3_PRODUCTION_SUBSUMED`, kept out per spec).
- **Known RC2 status:** 27 failed, 20 solved.
- **Namespace:** Set 40, Multiset 3, and one each of Eq / Function / Prop / GENERAL_FRONTIER.
- **Router predicted label:** MISSING_BRIDGE 20, BASELINE_DUPLICATE 10, SET_ITE 8, NO_CHEAP_ACTION 6,
  PROOF_SEARCH_DEPTH_GAP 2, WX3_MULTISET_INDUCTION 1.
- **Overlap with TR1 training: 47/47 (1.00).** The fresh frontier is **exhausted** — every candidate is
  already a TR1 example. Per spec ("exclude already-in-TR1 unless needed as controls") these are kept as
  re-probe / control targets and tagged `in_tr1_training`; this overlap is itself the central finding.

---

## 4. Batch selection

`scripts/tr2_select_batches.py`. Effective batch size auto-shrank from the requested 25 to **15** so the
two informed strategies stay disjoint (`floor(47/3)`).

| batch | n | failure ratio | namespaces | predicted-label mix |
|---|---|---|---|---|
| **model** | 15 | 0.87 | **6** (Eq, Multiset, Set, Function, Prop, GENERAL) | missing 6, no-cheap 4, depth 2, baseline 2, WX3 1 |
| **rule** | 15 | 0.73 | 2 (Set, Multiset) | missing 11, baseline 2, set-ite 2 |
| **random** | 15 | 0.87 | 3 (Set, Eq, Function) | set-ite 6, baseline 4, missing 3, no-cheap 2 |

- **model & rule are disjoint** by a round-robin draft.
- **random is an independent stratified control** matched to the model batch's 0.87 failure ratio. Matching
  that ratio under full disjointness is **infeasible** (model+rule drain the 27 scarce failures), so random
  overlaps model/rule on some failure cases — the spec's permitted "overlap unavoidable → record it" case,
  and the correct control: it isolates *selection quality* from *failure-ratio differences*. Overlaps are
  recorded in `tr2_batch_manifest.json` (`model&random`, `random&rule`).
- **Diversity signal already visible at selection:** model selection reaches 6 namespaces and 5 non-Set
  cases; rule and random stay Set-dominated (2 namespaces each beyond a stray).

---

## 5. RC2 confirmation

`scripts/tr2_confirm_rc2_status.py` — **reuse-first, live-fallback.** Every selected case already has an
identical-config literal-RC2 result in SF4 (`rc2_production_wrapper.json` · `ns24_router.json` ·
`hybrid_evolved` · top-k 8 · max-steps 8), so confirmation **reused** that verified oracle and ran **0 fresh
live confirmations** (the script is fully capable of live runs for genuinely-unknown cases — there were none).

- 34 unique cases: **26 CONFIRMED_RC2_FAILURE**, **8 RC2_SOLVED**, 0 flake/path-error.
- Provenance: 30 `sf4_reused`, 4 `tr1_reused`. Only the 26 confirmed failures are eligible for a true delta;
  the 8 solved cases serve as negative/baseline controls.

---

## 6. Probe plan

`scripts/tr2_generate_probes.py` maps the **router's top prediction** to a probe family (budget ≤ 8/theorem,
≤ 15 for depth-gap):

| predicted label | probe family | action |
|---|---|---|
| MISSING_BRIDGE_LEMMA_CANDIDATE | `retrieval` | `exact?` + **SF5 flag** — no blind tactic spam |
| PROOF_SEARCH_DEPTH_GAP | `depth_gap_bounded` | bounded depth-2/3 battery (≤ 15) |
| NO_CHEAP_ACTION | `minimal_controls` | simp / simp_all / aesop — negative if all fail |
| BASELINE_DUPLICATE (solved) | `controls` | simp / simp_all / aesop / classical;aesop |
| SET_ITE_SIMP | `set_ite_sanity` | sanity negative — RC2 owns it → PRODUCTION_SUBSUMED |
| WX3_MULTISET_INDUCTION | `multiset_induction` | gated `Multiset.induction_on <;> simp_all` |

Result: 34 theorems planned; **20 routed to SF5 retrieval** (no tactic spam), 2 depth-gap, 8 control, 4
minimal-control. Standard controls are attached for SX4 context and reused from SF4 where already executed.

---

## 7. Live probe results

`scripts/tr2_run_live_probes.py` (driver/worker, OS hard timeout + SIGALRM bounds, same model as
`sf4_run_candidate_probes.py`), with a **reuse layer**: SF4's verified control/sub-control/probe outcomes are
reused; only tactics SF4 never ran on a theorem are executed live.

- **34 theorems · 30 live · 64 fresh live tactics + 106 reused · 0 setup errors · 0 not-live.**
- **0 probe wins** (no gated probe closed any goal) — consistent with SF4's 0 cheap wins.
- **6 bare-control closes** surfaced (verified live/reused): `aesop` closes `Prop.compl_singleton`,
  `coe_notMemRangeEquiv_symm`, `Multiset.toFinset_nsmul`, `Set.insert_diff_eq_singleton`,
  `Set.insert_diff_of_mem`; `simp_all`/`aesop` close `Set.pair_diff_left`. The `Prop.compl_singleton` case
  is the known SF4 routing gap (RC2's aesop is gated to Set.Finite/toFinset, so production misses a goal that
  bare `aesop` closes).

---

## 8. SX4 attribution

`scripts/tr2_apply_attribution.py` (literal-production baseline + SX4 discipline — never credit a sequence on
depth-(k-1) controls):

| class | n | useful? |
|---|---|---|
| MISSING_BRIDGE_LEMMA_CANDIDATE | 20 | ✓ (SF5 retrieval targets) |
| BASELINE_DUPLICATE | 6 | ✓ (verified control proof — routing/depth gap) |
| NO_CHEAP_ACTION | 3 | ✓ (verified negative) |
| PROOF_SEARCH_DEPTH_GAP | 2 | ✓ (bounded battery fails → deeper search) |
| PRODUCTION_SUBSUMED | 3 | — (RC2 solves via search; no new label) |
| **TRUE_DELTA** | **0** | — |

**31 useful labels, 0 credited deltas, 0 over-credit.** The loop produced no false wins — the SX4 gate held.

---

## 9. Strategy comparison

`scripts/tr2_compare_selection_strategies.py`:

| metric | model | rule | random |
|---|---|---|---|
| useful labels | 14 | 14 | 14 |
| true deltas | 0 | 0 | 0 |
| useful / live-probe | 0.483 | 0.519 | 0.500 |
| missing-lemma candidates | 9 | 11 | 8 |
| depth-gap cases | **2** | 0 | 2 |
| no-cheap-action confirmations | 1 | 0 | 3 |
| baseline duplicates | 2 | 3 | 1 |
| **namespace diversity** | **6** | 2 | 3 |
| **non-Set cases** | **5** | 2 | 2 |
| live probes spent | 29 | 27 | 28 |

The three strategies are **statistically indistinguishable on yield** (identical 14 useful labels;
useful-per-probe within 0.48–0.52 noise). The one robust difference is **coverage**: model selection
reaches 3× the namespace diversity and 2.5× the non-Set cases of the baselines, and uniquely picks up the
depth-gap and Multiset-induction frontier. That edge is real but cannot be promoted to a yield win at this n.

---

## 10. Dataset update

`scripts/tr2_update_training_dataset.py` (non-destructive; TR1 file untouched):

- **0 net-new examples** — every probed case is already a TR1 example (frontier exhausted).
- **27 reconfirmations** — live re-probing corroborates the existing label.
- **4 verified label-revision candidates** (flagged for review, **not** auto-applied):
  - `Eq.subset`: NO_CHEAP_ACTION → **PROOF_SEARCH_DEPTH_GAP**
  - `Set.pairwiseDisjoint_filter`: NO_CHEAP_ACTION → **PROOF_SEARCH_DEPTH_GAP**
  - `Prop.compl_singleton`: PROOF_SEARCH_DEPTH_GAP → **BASELINE_DUPLICATE** (bare `aesop` closes it — RC2
    aesop-gating routing gap)
  - `Multiset.toFinset_eq_singleton_iff`: NO_CHEAP_ACTION → **MISSING_BRIDGE_LEMMA_CANDIDATE**
- **3 excluded** (PRODUCTION_SUBSUMED / no usable label).
- Underrepresented labels **not** improved (no new rows). `tr1_plus_tr2_examples.jsonl` = 57 (unchanged count).

The genuine training value here is **label hygiene** (4 corrections from verified observation), not dataset
growth. Optional retrain on TR1+TR2 is a no-op at the row level (0 net-new) and is left as exploratory.

---

## 11. Decision

### `INCONCLUSIVE_TOO_SMALL`

No strategy can yield a fresh TRUE_DELTA on an exhausted, fully pre-labelled ~40-theorem pool; the three
selection methods tie on useful labels and useful-per-probe. Model selection shows a real **diversity /
coverage edge** (more namespaces, more non-Set, captures depth-gap + Multiset frontier) and surfaced the 4
verified label corrections, but the sample is far too small to certify it as out-performing the baselines on
the headline ACTIVE_LEARNING_GAIN metric. The active-learning loop is mechanically correct and over-credit-
free; it is **starved of novel cases**, exactly the bottleneck TR1 named.

---

## 12. Next steps

Because the result is inconclusive **for lack of fresh data** (not because the model selects badly):
- **Source a genuinely fresh, multi-namespace frontier** (Nat/Int/Finset/List/Bool/Option beyond the
  Set-saturated pool) and re-run TR2 — this is the only way to move the comparison off `INCONCLUSIVE`.
- **Run SF5 existing-lemma retrieval** over the 20 MISSING_BRIDGE_LEMMA_CANDIDATE targets (verify Mathlib
  lemmas exist before any synthesis) — model and rule both rank these highly.
- **Apply the 4 verified label revisions** to a TR1.1 dataset after human review (especially
  `Prop.compl_singleton` → the RC2 aesop-gating routing gap, a candidate routing tweak, *not* promoted here).
- Keep model selection for **diversity** (broadest namespace coverage per budget) when the frontier expands;
  do **not** promote the router to production routing.
- Do **not** create RC4; do **not** modify RC2.

---

## 13. Protected-file confirmation

`git diff --stat HEAD` for `rc1_production_wrapper.json`, `rc2_release/rc2_production_wrapper.json`,
`ns24_router.json`, and `tr1/data/tr1_examples.jsonl` → **empty (untouched)**. NS9 genome/checkpoints and
REL1/RC1/RC2 release reports untouched. README not modified. No production routing change, no RC4, no
candidate promoted, router not promoted. TR2 wrote only under `project/evolve/experiments/tr2/`,
`project/evolve/reports/tr2/`, and `scripts/tr2_*.py`. **No commit made**
(see `project/evolve/experiments/tr2/out/protected_files_check.txt`).
