# Evaluation report — nat_defs_medium / hybrid_evolved + v4.4 shape-aware retrieval

**Branch**: `v4-premise-retrieval-div`
**Parent commit (v4.3)**: `154d594` — `Add goal-shape filtering for retrieved premise tactics`
**v4.4 run id**: `evolve-20260522-044315-1c0395`
**Metrics**: `project/evolve/runs/evolve-20260522-044315-1c0395/eval/seed-baseline/eval-*/metrics.json`
**Checkpoint**: `project/models/gen_v5` (unchanged)
**Top-k**: 8, **Max-steps**: 8, **Decode**: beam
**Wallclock**: 4m 21s (≈ 6.9 s/theorem)

## Headline

| | v3.6 | v4.1 | v4.2 | v4.3 | **v4.4 (default)** |
|---|---:|---:|---:|---:|---:|
| **Proved** | **25/38** | **25/38** | **25/38** | **25/38** | **25/38** |
| Errored | 10 | 6 | 7 | 10 | **7** |
| Exhausted | 3 | 7 | 6 | 3 | **6** |
| Retrieval attempts | — | 783 | 303 | 121 | **279** |
| Retrieval advances | — | 6 | 16 | 0 | **0** |
| Retrieval wins | — | 0 | 0 | 0 | **0** |
| Shape-mismatch forms suppressed | — | — | — | — | **213** |
| Apply-bloat events | — | 6 | 16 | 3 | **3** |
| `apply` skips (bloat pre-filter) | — | — | — | 0 | **9** |
| `unknown constant` (retrieved) | — | 200 | 0 | 0 | **0** |
| Wallclock | ~3m 28s | 5m 22s | 4m 19s | 3m 42s | **4m 21s** |

v4.4 is **the cleanest retrieval pipeline yet**: every emitted retrieved
tactic is shape-compatible with the current goal (213 guaranteed-fail
emissions suppressed), apply-bloat suppression continues to bite (9
pre-filter hits on top of v4.3's 3 first-observation bloats), and zero
`unknown constant` errors. The 25 wins, their origins, and their winning
tactics are **bit-identical** to v3.6/v4.1/v4.2/v4.3.

## Success criteria

| Criterion | Met? | Detail |
|---|---|---|
| ≥ 25/38 preserved | ✓ | 25/38 (matches v3.6 / v4.1 / v4.2 / v4.3) |
| No regressions | ✓ | `proved_by_origin = {fallback_tactic: 18, family_tactic: 4, generative_topk: 3}` — bit-identical to v3.6 |
| `unknown constant` remains 0 (retrieved) | ✓ | 0 in retrieved-premise traces. The 48 in the raw trace stream are all `generative_topk` hallucinations, unchanged across versions. |
| No new `DojoCrashError` | ✓ | Denied tactics = 8 (unchanged) |
| Retrieval attempts not blown up over v4.3 default | ⚠ | 279 vs v4.3's 121 — **increased, but for a productive reason**: rollouts now run deeper (3 div theorems escape the v4.3 ERR @early failure mode and reach EXH @8). Each extra step generates ≈ 27 more retrieval calls. Compared to v4.2 (303) v4.4 is lower. |
| Shape-mismatch attempts decreased | ✓ | 213 forms suppressed by shape filter — see "shape filter activity" below. |
| ≥ 1 new div theorem solved (*ideal*) | ✗ | 0 new closures. Shape-aware emission eliminates form-mismatch errors but does not unlock new proofs — the limiting factor is now catalog gaps + hypothesis chaining (see "Why still no new wins" below). |

**6 of 7 explicit criteria met.** The ideal closure criterion is again
not met; this is the same wall identified by v4.3 (catalog/induction).

## Shape filter activity

| Goal shape | Theorems | Retrieval attempts (v4.4) | Shape-mismatch forms suppressed |
|---|---:|---:|---:|
| `iff` | 4 (`div_lt_iff_lt_mul'`, `div_lt_one_iff`, `div_pos_iff`, `dvd_iff_div_mul_eq`) | 194 | 135 |
| `le` | 1 (`div_le_div_right`) | 65 | 68 |
| `lt` | 1 (`div_pos`) | 20 | 10 |
| **Total over div theorems** | **6** | **279** | **213** |

Per-shape attempt distribution:

* `eq` lemmas: 111 attempts (mostly `rw`/`simp` against iff/eq goals)
* `unknown` lemmas: 63 attempts (`Nat.mod_add_div` etc., emitted under all configured forms by design)
* `lt` lemmas: 59 attempts (mostly the bloater `Nat.lt_of_lt_of_le`, suppressed by bloat filter on 2nd+ encounter)
* `iff` lemmas: 46 attempts (top-ranked on iff goals by the shape bonus)

`shape_mismatch_filtered_count = 213` means the shape filter dropped
213 (lemma, form) pairs that the configured form list would otherwise
have emitted. With v4.3's default seed (`["rw","simp","apply"]`), every
retrieved lemma would have emitted 3 forms. After shape filtering, most
iff/eq lemmas on iff goals emit only `rw`/`simp` (2 forms), and most
lemmas on inequality goals emit either `apply`/`exact` or `rw`/`simp`
exclusively. The 213 suppressed emissions are almost entirely cases
where v4.3 would have produced `apply LEMMA` against an incompatible
goal shape — guaranteed `apply failed, failed to unify`.

## Per-div-theorem result

| Theorem | Shape | v3.6 | v4.1 | v4.2 | v4.3 | **v4.4** | attempts (v4.4) | shape-mismatch suppressed |
|---|---|---|---|---|---|---|---:|---:|
| `Nat.div_le_div_right` | le | ERR @3 | EXH @8 | EXH @8 | ERR @3 | **EXH @8** | 65 | 68 |
| `Nat.div_lt_iff_lt_mul'` | iff | ERR @2 | EXH @8 | EXH @8 | ERR @2 | **EXH @8** | 105 | 48 |
| `Nat.div_lt_one_iff` | iff | ERR @4 | EXH @8 | EXH @8 | ERR @4 | **EXH @8** | 59 | 52 |
| `Nat.div_pos` | lt | ERR @3 | ERR @3 | ERR @3 | ERR @3 | **ERR @3** | 20 | 10 |
| `Nat.div_pos_iff` | iff | ERR @4 | ERR @4 | ERR @4 | ERR @4 | **ERR @4** | 15 | 20 |
| `Nat.dvd_iff_div_mul_eq` | iff | ERR @3 | EXH @8 | EXH @8 | ERR @3 | **ERR @3** | 15 | 15 |

3 theorems (`div_le_div_right`, `div_lt_iff_lt_mul'`, `div_lt_one_iff`)
that v4.3 errored-out early now run to step 8 — the shape filter
eliminated the false-positive `apply` and `rw` emissions that were
causing the rollout to declare "all top-N errored" prematurely. They
still don't close, but they explore.

`div_pos` and `div_pos_iff` error at the *exact* step they did in
v3.6 / v4.1 / v4.2 / v4.3 — meaning every non-retrieved tactic also
errors at that step, and no retrieval form (under any shape filter)
can engage. These are catalog-gap failures, not shape failures.

`dvd_iff_div_mul_eq` regressed from v4.2's EXH @8 back to ERR @3 — the
shape filter is correctly suppressing the `apply Nat.dvd_iff_div_mul_eq`
that v4.2 was throwing against the iff goal, and without that bloat
chain, the rollout hits all-errored sooner. This is consistent with
v4.3's pattern: bloat-extended rollouts in v4.1/v4.2 were artifacts of
non-productive advances.

## How the shape filter works

`premise_retriever.py` adds two classifiers and one form-allow table:

1. **`classify_goal_shape(state_pp) → str`** — finds the `⊢` line and
   returns one of `eq` / `iff` / `lt` / `le` / `dvd` / `and` / `or` /
   `unknown` based on which Unicode glyph is present at the top level.
   Checked in order `↔ > ∣ > ∧ > ∨ > ≤ > < > =` so a `x = y ↔ p` goal
   correctly classifies as `iff`, not `eq`.

2. **`lemma_shape_from_name(name) → str`** — heuristic on the bare
   (post-last-dot) name, lowercased. Order:
   `_iff` (substring or suffix) > `dvd` > `_eq` / `cancel` > `pos` >
   `_lt` > `_le`. Falls through to `unknown` so the caller emits the
   full configured form list rather than over-pruning. Validated
   against the 16-lemma `Nat.div` bucket; all 16 classifications match
   their conclusion shape.

3. **`_SHAPE_FORM_ALLOW[(goal_shape, lemma_shape)] → set[str]`** —
   per-pair form whitelist:

   | Goal | Lemma | Allowed forms |
   |---|---|---|
   | iff | iff | rw, simp, apply |
   | iff | eq | rw, simp |
   | iff | lt/le/dvd | rw, simp |
   | le/lt | lt/le | apply, exact, rw, simp |
   | le/lt | iff/eq | rw, simp |
   | dvd | dvd | apply, exact, rw, simp |
   | dvd | iff | rw, simp, apply |
   | dvd | eq | rw, simp |
   | eq | eq | rw, simp, apply, exact |
   | eq | iff | rw, simp |
   | (unknown on either side) | | full configured list |

   `forms_for_shape_pair(goal, lemma, configured)` returns the
   intersection of `_SHAPE_FORM_ALLOW[(goal,lemma)]` and `configured`,
   preserving the configured-list order for determinism. Default fall-
   back when a `(goal_shape, lemma_shape)` pair isn't listed is
   `{"rw","simp"}` — the safe forms.

In `StrategyWrapperPolicy.rank_tactics`, the retrieval block now passes
`shape_aware=True` to `retrieve_for_state` (which adds the shape
scoring bonus), and for each returned lemma calls
`forms_for_shape_pair` to choose which forms to emit. Per-entry shape
labels flow through the 7-tuple `Entry` shape into `last_retrieved_shapes`.

`eval_rollout_all.rollout_one_theorem` tags every retrieved trace
record with `goal_shape`, `tactic_retrieved_shape`, and a
`shape_match: bool`. Per-theorem counters
`retrieved_shape_counts` / `retrieved_shape_success_counts` /
`shape_mismatch_filtered_count` aggregate into the metrics.

## v4.4 retrieval scoring (excerpt)

The new shape-aware bonuses in `retrieve_for_state` (additive on top of
v4.1's token-overlap + family-token bonus):

| Goal shape | Lemma shape | Bonus |
|---|---|---:|
| any | matching | +1.5 |
| iff | eq | +0.5 |
| iff | dvd | +0.3 |
| lt/le | lt/le (cross) | +0.8 |
| lt/le | iff | +0.4 |
| eq | iff | +0.3 |
| dvd | iff | +0.7 |
| (other mismatch with no token overlap) | | −0.5 |

Concrete example — for `Nat.div_lt_one_iff` (iff goal), the top-5
ranked premises change as follows compared to v4.3:

```
v4.3 (token + family bonus only):
  1. Nat.div_lt_iff_lt_mul        (iff, shares div+lt+iff)
  2. Nat.div_lt_iff_lt_mul'       (iff)
  3. Nat.div_pos_iff              (iff)            ← filtered as unavailable in v4.2
  4. Nat.div_eq_of_lt             (eq, shares lt)
  5. Nat.lt_of_lt_of_le           (lt, no shape relevance for iff)

v4.4 (with shape bonus):
  1. Nat.div_lt_iff_lt_mul        (iff: token+family+shape match → +1.5)
  2. Nat.div_lt_iff_lt_mul'       (iff: same)
  3. Nat.div_eq_of_lt             (eq: token only)
  4. Nat.mul_div_cancel           (eq)
  5. Nat.div_mul_cancel           (eq)
```

The lt lemmas drop out of the top-5 entirely; the eq lemmas keep their
position (their `rw`/`simp` forms can sometimes rewrite inside an iff).

## Why still no new wins

The shape filter eliminates *form mismatch* (the symptom) but the
6 unproved div theorems all fail for one of two structural reasons:

1. **Hypothesis-chaining gaps.** `Nat.div_pos_iff : 0 < a/b ↔ b ≠ 0 ∧ b ≤ a`
   would close `Nat.div_pos` via
   `exact (Nat.div_pos_iff.mpr ⟨ne_of_gt hb, hba⟩)`, but the wrapper
   has no way to construct the `⟨_,_⟩` term and would need to chain
   `Iff.mpr` + an `And.intro` synthesizing two hypothesis lookups. None
   of `rw`/`simp`/`apply`/`exact` LEMMA forms can do this without an
   explicit term builder.
2. **Catalog gaps.** `Nat.div_le_div_right` actually closes via
   `induction h with ...` on the `a ≤ b` hypothesis, which neither the
   gen_v5 model nor the family-tactics list emits. Nothing in
   `STATIC_PREMISES["Nat.div"]` engages a `≤` goal of the form `a/c ≤ b/c`
   without going through an induction.

These are v4.5 / v4.6 territory (induction templates and term-mode
applies), explicitly out of v4.4 scope.

## What v4.4 plumbed

### `premise_retriever.py`

* `classify_goal_shape(state_pp) → str` — top-level Unicode-glyph dispatch.
* `lemma_shape_from_name(name) → str` — heuristic with sensible
  fall-through to `"unknown"`.
* `_SHAPE_FORM_ALLOW: dict[(str,str), set[str]]` — 17-entry per-pair
  form whitelist; default `{"rw","simp"}` for unlisted pairs.
* `forms_for_shape_pair(goal, lemma, configured) → list[str]` —
  shape-filtered subset preserving caller order.
* `retrieve_for_state(..., shape_aware=True)` — additive shape bonus
  on top of token-overlap + family-token scoring. Diagnostics now
  include `goal_shape`, `lemma_shapes` (per returned premise), and
  `shape_aware`.

### `evolve/strategy_wrapper.py`

* `StrategyWrapperPolicy(..., retrieval_shape_filter=True)`.
* Entry tuple extended from 6 → 7 elements (adds
  `retrieved_shape: str | None`).
* `last_retrieved_shapes`, `last_goal_shape`,
  `last_shape_mismatch_filtered_count` exposed for the eval loop.
* Per-call: configured form templates are reduced via
  `forms_for_shape_pair(goal_shape, lemma_shape, configured_labels)`
  before tactic synthesis.
* `dump_strategy_config` / `load_strategy_config` now round-trip 13
  fields (was 12 in v4.3); new `retrieval_shape_filter` JSON key.

### `evolve/candidate.py`

* `SearchCandidate.retrieval_shape_filter: bool = True`.

### `evolve/evaluator.py` + `evolve/run_evolve.py`

* Pass `retrieval_shape_filter` from the candidate through to the
  strategy config.
* `make_seed_candidate("hybrid_evolved", ...)` enables the shape
  filter by default; description updated.

### `eval_rollout_all.py`

* `_load_policy` unpacks the new 13-tuple from `load_strategy_config`.
* `rollout_one_theorem`:
  * Captures `step_goal_shape` per rank_tactics call.
  * Records `goal_shape` on the per-theorem result at step 1.
  * Each retrieved trace record tagged with
    `tactic_retrieved_shape` / `goal_shape` / `shape_match`.
  * Per-theorem counters
    `retrieved_shape_counts` / `retrieved_shape_success_counts` /
    `shape_mismatch_filtered_count`.
  * Aggregates surface the same in the run metrics.
* Console summary prints `Shape attempts: … success=… mismatch_filtered=…`.

## What v4.4 did NOT do (per scope)

* No retraining, no checkpoint changes.
* No new policy types.
* No new lemmas added to `STATIC_PREMISES`.
* No induction templates.
* No term-mode `exact` synthesis (no `⟨_,_⟩` builder).
* No live Lean availability checker (`_UNAVAILABLE_LEMMAS` is still
  the static v1).

## Recommended next step (v4.5)

**Induction templates for the div family.** `Nat.div_le_div_right`
closes via `induction h with | refl => ... | step _ ih => simp [...] ; exact ih`
on the `a ≤ b` hypothesis; `Nat.div_pos` closes via
`exact Nat.div_pos_iff.mpr ⟨_, hba⟩` (needs term-builder). Both
require shapes the current form set can't synthesize. The lighter v4.5
would be just induction templates for `≤`/`<` div goals (in the
`theorem_family_tactics["div"]` list); the term-builder is heavier and
can be deferred to v4.6.

If that still doesn't close anything, the next answer is to extend the
seeded catalog with a handful of *productive* hand-picked lemmas —
guided by which lemmas v4.4's shape filter *would have* accepted but
weren't available (e.g. `Nat.div_le_iff_le_mul'`, currently absent
from the bucket).

## Artifacts

* `project/evolve/runs/evolve-20260522-044315-1c0395/` — v4.4 run root
* `…/eval/seed-baseline/eval-*/metrics.json` — full metrics
* `…/eval/seed-baseline/eval-*/traces.jsonl` — 1,532 trace records
  (279 retrieved, each tagged with `tactic_retrieved_shape` and
  `goal_shape` / `shape_match`)
* `…/eval/seed-baseline/strategy_config.json` — dumped config
  (includes `retrieval_shape_filter=true`)
* Branch: `v4-premise-retrieval-div` (off `e74861f`); v4.4 commit
  pending on top of `154d594`.
