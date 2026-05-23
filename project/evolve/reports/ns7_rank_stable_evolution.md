# NS7 — rank-stable skeleton evolution

NS6 (commit `2b6044b`) reached 20 enabled skeletons preserving 37/38
medium and 49/65 large. The credit-aware safe pruning still produced
6 regressions, all on a single theorem (`Nat.div_lt_iff_lt_mul'`)
caused by a *second-order rank-coupling* effect: disabling an
uncredited skeleton shifts the wrapper's top-K window so that a
correctly-protected retrieval skeleton drops out of the ranked list.

NS7 attacks the rank-coupling class directly:

1. **Stable skeleton IDs** so we can talk about the *same* skeleton
   across mutations even after `from_legacy_strategy_config`
   re-indexes names.
2. A **rank-diff diagnostic tool** that compares the bag's
   deterministic skeleton-emit order between two genomes.
3. A **protected-skeletons set** persisted to disk, listing each
   credit-bearing skeleton's `(theorem, state_hash)` plus the
   observed required-rank-max in trace coordinates.
4. A **pre-flight rank-coupling detector** that runs *before* the
   Lean eval and rejects a candidate if any protected skeleton is
   dropped or pushed back in the bag's skeleton-emit order.
5. A **credit-aware `archive_seed`** that uses the
   `direct_wins*10 + assist*5 + advances*1 - 10*regr` score instead
   of wins-only selection, with unconditional protection for any
   skeleton with non-zero assist credit.

## Hard constraints respected

- No retraining; no checkpoint changes.
- No broad refactor — every change is additive.
- Preserved: `nat_defs_medium 37/38` and `nat_defs_large_v5 49/65`.
- No LLM mutator; no gen_v5+1 training.
- Run artifacts under `project/evolve/ns7_runs/` gitignored.

## Stage 1 — stable skeleton IDs

Added a `stable_id` property to `Skeleton`:

```
canonical = origin | shape | family | specificity | template.strip()
stable_id = sha1(canonical)[:12]
```

The id is invariant across `from_legacy_strategy_config` rebuilds —
two skeletons that share identity fields share an id regardless of
their `name`. Retrieved-premise skeletons append `(premise, form)` to
the canonical string. Verified: the NS6 best (20 enabled) has 25 ↔ 20
distinct stable_ids, and disabling/rebuilding preserves the id of
every still-enabled skeleton.

`EmittedTactic` now carries `skeleton_stable_id` and the wrapper's
parallel `last_skeleton_stable_ids` list feeds it into per-step trace
rows.

## Stage 2 — rank-diff diagnostic

`scripts/ns7_rank_diff.py` reads two genomes and the protected file,
then for every shape with protected entries:

- Computes `_enabled_skeletons_by_shape(bag, shape)` — the
  deterministic order the wrapper would iterate skeletons
  (priority_template by (priority, specificity); then family_tactic,
  term_builder, fallback_tactic, tactic_template all in insertion
  order in the `any` slot).
- Indexes each list by `stable_id`.
- Reports every protected skeleton whose index moved or whose entry
  is missing in the mutated bag.

The report is emitted as markdown for human review.

## Stage 3 — protected skeleton set

`scripts/ns7_protected_set.py` walks per-step traces and classifies
each emission:

- `direct_win`       — skeleton emitted the closing tactic
- `assist_win`       — skeleton advanced, next K accepted steps closed
- `critical_advance` — skeleton advanced, next K had another advance

For the baseline genome (NS6 best, 20 enabled, fresh medium+large
traces with NS7's stable-id surfacing), the protected set is:

- **17 distinct stable_ids protected**, **93 entries**
- by reason: `{direct_win: 79, assist_win: 4, critical_advance: 10}`

Persisted to `project/evolve/ns7_runs/baseline/protected_skeletons.json`
(gitignored — regeneratable from fresh baseline traces).

## Stage 4 — pre-flight rank-coupling detector

`evolve/rank_coupling.py::check_rank_coupling(baseline, mutated,
protected_entries)` returns a list of `RankViolation` for every
protected skeleton whose **bag-side skeleton-emit-index** moves
backward or whose stable_id is absent from the mutated bag.

Important scoping decision: the trace's `required_rank_max` is
recorded in *wrapper-merged-list* coordinates (skeleton emissions
interleaved with model-output candidates). The detector operates in
*bag-only* coordinates because model outputs are deterministic per
state — they cannot be predicted offline without loading the model.
Comparing across scales is unsafe, so the detector uses
`baseline_bag_rank` throughout and treats `required_rank_max` as a
diagnostic artifact only.

### Sanity test

Disabling each skeleton in the baseline genome one at a time
produces (under the medium-only protected set):

| disable | violations | kind |
|---|---:|---|
| `pt_iff_8` (the omega-bash; 17 direct wins) | 17 | dropped |
| `pt_iff_0` … `pt_iff_7` (each direct-win) | 1 | dropped |
| `pt_iff_2` (no credit) | 0 | SAFE |
| `fam_div_14` (no credit) | 0 | SAFE |
| `fb_19`, `fb_18` (mixed) | 0 / 4 | safe / dropped |

The detector identifies first-order violations correctly. The NS6
cycle-4 mutation (`fb_19, pt_iff_2, fam_div_14`) passes pre-flight
because none of those skeletons are directly protected — the
regression there is a **second-order rank-coupling effect** the
detector cannot see without running the model offline.

## Stage 5 — credit-aware `archive_seed_credit`

New operator. Selection rule:

```
score = 10·direct_wins + 5·assist_wins_k3 + 1·advances
        − 10·regressions − dead_attempt_penalty
```

Plus an unconditional keep-list for any skeleton with non-zero
`assist_wins_k3`. The wins-only `archive_seed` operator is preserved
for comparison.

## Stage 6 — bounded sweep (21 cycles, ~50 min)

**Result**: best at **cycle 1** — baseline confirms 37/38 medium and
49/65 large at 20 enabled skeletons. No mutation crossed the 37
threshold; all compaction attempts regressed by exactly one theorem
(`Nat.div_lt_iff_lt_mul'`).

### Pre-flight performance

| metric | count |
|---|---:|
| total cycles | 21 |
| pre-flight rejected (no Lean eval) | **3** |
| Lean rejected (gate caught at eval) | 10 |
| accepted (no promotion) | 8 |
| promoted | 1 (baseline confirms) |

Pre-flight rejected 3 mutations without paying for a Lean roundtrip:

- c5: `archive_seed_credit top_n=12` — 6 violations (3 affected theorems)
- c11: `archive_seed top_n=18` (wins-only) — 14 violations (6 affected theorems)
- c12: `archive_seed top_n=22` (wins-only) — 12 violations (5 affected theorems)

The wins-only `archive_seed` cycles 11/12 dropped multiple
assist-credit skeletons — exactly the failure NS5 documented. The
detector caught them pre-flight at zero Lean cost.

### Cycle-by-cycle

| c | operator | kwargs | med | l | en | preflight | result |
|---|---|---|---|---|---|---|---|
| 1 | baseline | — | 37 | 49 | 20 | OK | promoted |
| 2 | disable_dead_skeleton | {3,2} | 36 | — | 18 | OK | Lean rejected |
| 3 | disable_dead_skeleton | {5,3} | 36 | — | 17 | OK | Lean rejected |
| 4 | disable_dead_skeleton | {8,5} | 36 | — | 18 | OK | Lean rejected |
| 5 | archive_seed_credit | top_n=12 | — | — | 12 | **REJ(6)** | pre-flight |
| 6 | archive_seed_credit | top_n=15 | 36 | — | 15 | OK | Lean rejected |
| 7 | archive_seed_credit | top_n=18 | 36 | — | 15 | OK | Lean rejected |
| 8 | archive_seed_credit | top_n=20 | 36 | — | 15 | OK | Lean rejected |
| 9 | archive_seed_credit | top_n=22 | 36 | — | 16 | OK | Lean rejected |
| 10 | archive_seed_credit | top_n=25 | 36 | — | 16 | OK | Lean rejected |
| 11 | archive_seed (wins-only) | top_n=18 | — | — | 9 | **REJ(14)** | pre-flight |
| 12 | archive_seed (wins-only) | top_n=22 | — | — | 11 | **REJ(12)** | pre-flight |
| 13 | promote_high_win | scope=priority/iff | 37 | — | 20 | OK | accepted |
| 14 | promote_high_win | scope=fallback/any | 37 | — | 20 | OK | accepted |
| 15 | promote_high_win | scope=tactic_template/any | 37 | — | 20 | OK | accepted |
| 16 | promote_high_win | scope=family_tactic/any | 37 | — | 20 | OK | accepted |
| 17 | demote_generic | scope=priority/iff | 37 | — | 20 | OK | accepted |
| 18 | demote_generic | scope=priority/any | 37 | — | 20 | OK | accepted |
| 19 | disable_dead_skeleton | {5,8} | 36 | — | 18 | OK | Lean rejected |
| 20 | disable_dead_skeleton | {10,12} | 36 | — | 18 | OK | Lean rejected |
| 21 | baseline | — | 37 | — | 20 | OK | accepted |

### Compaction floor analysis

Every disable_dead/archive_seed cycle reached 36/38 — exactly one
theorem short. The Lean trace confirms it was `Nat.div_lt_iff_lt_mul'`
in every case. NS5 (165 cycles) and NS6 (20 cycles) hit the same
floor.

The credit-aware `archive_seed_credit` plateaued at **36/38**
(one theorem better than NS5's wins-only 35/38), because it
correctly retains `retrieved:Nat.div_lt_iff_lt_mul:rw` (assist
credit). But it still loses `Nat.div_lt_iff_lt_mul'` to a different
mechanism: disabling *other* (uncredited) skeletons shifts the
wrapper's top-K cut. The "+1 theorem" gain is real but limited.

### What pre-flight does and does not catch

- ✅ **Catches** mutations that drop a credit-bearing skeleton from
  the bag entirely. NS7 cycles 11-12 are exactly this — wins-only
  `archive_seed` removes `pt_any_13` and `retrieved:Nat.div_lt_iff_lt_mul:rw`,
  both must-protects.
- ✅ **Catches** mutations that push a protected skeleton back in
  the bag's deterministic emit order (priority_template re-sorts,
  etc).
- ❌ **Does NOT catch** mutations that disable *uncredited*
  skeletons whose absence shifts the wrapper's top-K window for a
  *credited* skeleton. This is the NS6 cycle-4 / NS7 cycle-2 class.
  The Lean gate is the only safety net here.

## Comparison to NS6

| | NS6 (20 cycles) | NS7 (21 cycles) |
|---|---|---|
| best medium proved   | 37/38 | 37/38 |
| best large proved    | 49 | 49 |
| best enabled skeletons | 20 | 20 |
| compact-genome ceiling (archive_seed) | 35/38 (wins-only) | **36/38 (credit-aware)** |
| pre-flight rejections | n/a | 3 (saved ~7 min Lean) |
| Lean regressions    | 6 | 10 |
| stable id support   | no | **yes** |
| protected set on disk | no | **yes** (17 sids, 93 entries) |

The compaction floor at 20 enabled skeletons is now demonstrated to
be a *rank-coupling-induced* floor, not a credit-gap floor. NS7
shows that even with perfect credit-aware selection (assist
unconditional protection) the floor holds.

## Files added/changed

- `evolve/skeleton_bag.py` — `Skeleton.stable_id` property,
  `EmittedTactic.skeleton_stable_id` field, retrieved-emit stable_id.
- `evolve/strategy_wrapper.py` — `last_skeleton_stable_ids` parallel list.
- `eval_rollout_all.py` — per-step trace carries `skeleton_stable_id`.
- `evolve/skeleton_mutator.py` — `_credit_score`,
  `top_skeletons_by_credit_score`, `archive_seed_credit` operator.
- `evolve/rank_coupling.py` (new) — `check_rank_coupling`,
  `summarize_violations`.
- `evolve/skeleton_evolve_ns7.py` (new) — runner with pre-flight
  detector.
- `scripts/ns7_protected_set.py` (new) — builds protected JSON.
- `scripts/ns7_rank_diff.py` (new) — markdown diff report.
- `project/evolve/reports/ns7_rank_stable_evolution.md` (this file).
- `.gitignore` — add `project/evolve/ns7_runs/`.

## Recommendation for next step (NS8)

The 20-skeleton floor is a true **rank-coupling barrier**. To break
through, one of:

1. **Offline model-output recording for protected states.** Run the
   model once per protected state and store its top-K output ranks
   alongside the protected entries. Then the pre-flight detector can
   simulate the wrapper's merged list (skeletons + cached model
   outputs) and reject mutations that push a protected skeleton out
   of the merged top-K. This eliminates the second-order
   rank-coupling class without retraining.

2. **`max_extra_tactics_per_state` increase as a mutation.** Right
   now this knob is fixed; expanding it gives the protected skeleton
   more room to land in the wrapper's window. Cost: more Lean
   roundtrips per state. Worth trying once.

3. **Stable-id–keyed archive ingestion.** The archive still indexes
   by `name`. Migrating to stable_id makes cross-run credit
   aggregation correct.

4. **`max_steps` increase as a mutation.** `Nat.div_lt_iff_lt_mul'`
   takes 2 steps; if a 3-step path exists that uses currently-pruned
   skeletons, we'd see it.

5. **Rank-coupling-aware `disable_dead_skeleton`.** Augment the
   credit check with: "would removing this skeleton shift any
   *other* protected skeleton's bag-rank backward?" That extension
   to the operator is small and doesn't need the model.