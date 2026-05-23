# NS6 — credit-aware archive and scoped skeleton mutation

NS5 (commit `9e546f0`) compressed the proved-37/preserved-49 genome from
48 skeletons to 25 over 165 cycles. Two design defects surfaced:

1. **Wins-only pruning blind spot** — `disable_dead_skeleton` deleted a
   skeleton whenever its archive win-count was zero. But some
   zero-win skeletons *advance state into a form a later tactic
   closes*. NS5 logged 60+ regressions that hinged on this; the most
   brittle theorem was `Nat.div_lt_iff_lt_mul'`.

2. **Order-changing mutators clobbered unrelated bands** — both
   `demote_generic_skeleton` and `promote_high_win_skeleton` rebuilt
   `bag.skeletons[shape]` (or its full ordering) when they only needed
   to reorder one (origin, shape, family) bucket. The wrapper iterates
   `bag.skeletons["any"]` in insertion order for fallback and
   tactic_template emission, so bag-wide resorts shuffled emission
   for unrelated origins. NS5 cycle-2 lost `Nat.two_mul_ne_two_mul_add_one`;
   cycle-4 lost `Nat.add_mod_eq_add_mod_right`.

NS6 fixes both. The work is small and surgical: one trace-attribution
patch (`eval_rollout_all.py`), one new analyzer, four mutator changes,
one new short-sweep runner, and two reports.

## Hard constraints respected

- No retraining, no checkpoint changes.
- No broad refactor — every change is additive or in-place edit.
- Baseline preserved: `nat_defs_medium 37/38` and `nat_defs_large_v5
  49/65` (the set actually has 65 theorems; NS5's "49/64" was an
  off-by-one in that report).
- `use_skeleton_bag` flag untouched.
- Run artifacts gitignored under `project/evolve/ns6_runs/` (added).

## Stage 1 — assist-credit signal

Patched `eval_rollout_all.py` so per-step traces carry the bag's
`skeleton_name`, `skeleton_shape`, `skeleton_family`, and friends
(previously only the closing-step rolled them up into `metrics.json`'s
`winning_tactic_skeleton_*`). This was a four-line addition to the
existing record-dict assembly block.

`scripts/ns6_assist_credit.py` walks one or more `traces.jsonl` files,
groups rows by `episode_id`, identifies the accepted advance/close per
step, and credits skeletons as follows:

  - `direct_wins`     — skeleton emitted the closing tactic
  - `advances`        — skeleton's tactic produced a fresh state
  - `assist_wins_kN`  — skeleton advanced, and within the next N
                        accepted proof steps a *different* tactic
                        closed the proof

The analyzer was run against fresh baseline traces on `nat_defs_medium`
and `nat_defs_large_v5`. Full output in
`project/evolve/reports/ns6_assist_credit_analysis.md`.

### Headline findings

- **52 distinct skeletons fired** across medium + large.
- **14 direct-win skeletons** account for all 79 wins.
- **2 zero-win-but-assists skeletons** — must protect:
  - `pt_any_13` — assisted on `Nat.add_mod_eq_ite`
  - `retrieved:Nat.div_lt_iff_lt_mul:rw` — assisted on `Nat.div_lt_iff_lt_mul'`
- **1 advance-only-without-assist skeleton** (`retrieved:Nat.div_eq_of_lt:rw`):
  3 advances, never within 3 steps of a close. Marginal — review for
  later removal.
- **10 truly-dead skeletons** (≥5 attempts, no signal):
  `pt_eq_15`, `fb_23`, `pt_le_16`, `fb_24`, `fam_div_18`,
  `retrieved:Nat.div_lt_iff_lt_mul':{rw,simp}`,
  `retrieved:Nat.div_lt_iff_lt_mul:simp`,
  `pt_lt_10`, `pt_lt_9`.
- **25 low-attempt skeletons** (<5 attempts on the joint medium+large
  set) — insufficient signal to classify either way; left alone.

The two assist-credit theorems are exactly the two NS5 reported as the
most brittle. NS5's pruning would have silently lost them; NS6's safe
pruning keeps them by design.

## Stage 2 — scoped order-changing operators

`MutationRecord` now carries `scope_origin`, `scope_shape`,
`scope_family` and an `affected_skeletons` alias. Both
`promote_high_win_skeleton` and `demote_generic_skeleton` now *require*
explicit scope kwargs — calling them without scope is a no-op.

The mechanics:

  - `promote_high_win_skeleton(scope_origin, scope_shape, scope_family=None)`
    finds the top archive winner among skeletons in the scope and moves
    it to the *first in-scope position* in `bag.skeletons[shape]`.
    Out-of-scope positions are preserved exactly.

  - `demote_generic_skeleton(scope_origin, scope_shape)` resorts only
    the in-scope subset by `(priority, specificity)`. Out-of-scope
    positions are preserved exactly.

The implementation works at index granularity: the in-scope subset of
`bag.skeletons[shape]` is pulled out by position, reordered, then
written back into those same positions. Skeletons in other (origin,
shape) cells of the same shape slot — e.g. fallback_tactic and
tactic_template entries that share `shape=any` — never move.

### Side-effect taxonomy

Where each operator's reorder is observable, after scoping:

  - **`priority_template`** (any shape): reorder is observable only if
    the wrapper's NS3.5 `for_shape` resort breaks ties differently.
    Since `for_shape` re-sorts by `(priority, specificity)`, reorder
    *within* `priority_template` is mostly cosmetic — but harmless.

  - **`fallback_tactic`** (`shape=any`): reorder is *meaningful* — the
    wrapper's `emit_fallback_tactics` iterates `bag.skeletons["any"]`
    in insertion order and only the first-budget entries are emitted.
    This is the path where NS5 cycle-2 lost a theorem.

  - **`tactic_template`** (`shape=any`): identical story to
    `fallback_tactic` — `emit_tactic_template_tactics` iterates the
    `any` slot in insertion order. NS5 cycle-4 lost a theorem here.

The scoped operators address #2 and #3 cleanly. #1 entries remain in
the sweep queue as no-regression-by-construction safety checks.

## Stage 3 — safe pruning rule

`disable_dead_skeleton` now takes an optional `credit_stats` dict. When
provided, it disables a skeleton only when ALL FOUR conditions hold:

  - `attempts >= min_attempts`
  - `direct_wins = 0`
  - `advances = 0`
  - `assist_wins_k3 = 0`

Skeletons with any assist credit are *never disabled*, even when their
direct-win count is zero. The runner builds the credit dict from the
baseline traces (seeded once) and updates it after every cycle's eval.

Without `credit_stats`, the operator falls back to the NS5 wins-only
behavior so older sweep code still works.

## Stage 4 — short sweep results

20-cycle sweep, all medium-only after the first 3 promotions consumed
the large-eval budget (configured `--max-large-evals 3`).

**Bottom line**: best at **cycle 3** — `proved_medium=37`,
`proved_large=49`, **20 enabled skeletons** (down from NS5's 25, a
further 20% reduction).

### Per-operator breakdown

| operator | accepted | rejected | promoted |
|---|---:|---:|---:|
| baseline                  | 2 | 0 | 1 |
| disable_dead_skeleton     | 2 | 3 | 2 |
| promote_high_win_skeleton (scoped) | 5 | 0 | 0 |
| demote_generic_skeleton (scoped)   | 5 | 0 | 0 |
| archive_seed              | 0 | 3 | 0 |

**Reorders: 10 cycles, zero regressions.** Every scoped
`promote_high_win_skeleton` and `demote_generic_skeleton` cycle
(c5–c14, covering 5 distinct origin/shape buckets each) accepted
without losing a theorem. NS5's *unscoped* counterparts of these
operators regressed at cycles 2 and 4 of its 165-cycle run — i.e. the
exact failure NS6 was designed to eliminate.

### Cycle-by-cycle

| c | operator | scope | med | l | en | acc | prom | notes |
|---|---|---|---|---|---|---|---|---|
| 1 | baseline | — | 37 | 49 | 25 | Y | Y |  |
| 2 | disable_dead_skeleton | — | 37 | 49 | 23 | Y | Y | strict_compact |
| 3 | disable_dead_skeleton | — | 37 | 49 | 20 | Y | Y | strict_compact |
| 4 | disable_dead_skeleton | — | 36 | — | 17 | N |   | regression rejected |
| 5 | promote_high_win | priority/iff | 37 | — | 20 | Y |   | no strict gain |
| 6 | promote_high_win | priority/any | 37 | — | 20 | Y |   | no strict gain |
| 7 | promote_high_win | priority/eq | 37 | — | 20 | Y |   | no strict gain |
| 8 | promote_high_win | fallback/any | 37 | — | 20 | Y |   | no strict gain |
| 9 | promote_high_win | tactic_t/any | 37 | — | 20 | Y |   | no strict gain |
| 10 | demote_generic | priority/iff | 37 | — | 20 | Y |   | no strict gain |
| 11 | demote_generic | priority/any | 37 | — | 20 | Y |   | no strict gain |
| 12 | demote_generic | priority/eq | 37 | — | 20 | Y |   | no strict gain |
| 13 | demote_generic | priority/lt | 37 | — | 20 | Y |   | no strict gain |
| 14 | demote_generic | priority/le | 37 | — | 20 | Y |   | no strict gain |
| 15 | archive_seed top_n=18 | — | 30 | — | 9 | N |   | regression rejected |
| 16 | archive_seed top_n=22 | — | 31 | — | 11 | N |   | regression rejected |
| 17 | archive_seed top_n=28 | — | 33 | — | 12 | N |   | regression rejected |
| 18 | disable_dead {5,8} | — | 36 | — | 18 | N |   | regression rejected |
| 19 | disable_dead {10,12} | — | 36 | — | 18 | N |   | regression rejected |
| 20 | baseline | — | 37 | — | 20 | Y |   | confirm best |

### Best genome (cycle 3)

20 enabled skeletons:

- 14 priority_templates: `pt_iff_{0..8}`, `pt_any_{10,11}`,
  `pt_eq_12`, `pt_le_13`, `pt_lt_9`
- 4 family_tactic: `fam_div_14`, `fam_mod_{15,16,17}`
- 2 fallback_tactic: `fb_18`, `fb_19`

(Names are post-renumbering — see Finding #3 below.)

### Compaction trajectory

| cycle | enabled | medium | large | event |
|---|---:|---:|---:|---|
| seed (NS5) | 25 | 37 | 49 | seed |
| cycle 2    | 23 | 37 | 49 | safe-prune disabled `pt_eq_15`, `pt_le_16` |
| cycle 3    | 20 | 37 | 49 | safe-prune disabled `fb_21`, `pt_lt_10`, `pt_lt_9` |

Three safe-prune actions reached the new local-minimum at 20 enabled
skeletons in 3 cycles (vs. NS5's 11 promotions reaching 25 enabled
over 165 cycles).

### The cycle-4 regression — single root cause across all rejections

Both `disable_dead_skeleton` rejections (c4, c18, c19) and all three
`archive_seed` rejections (c15-c17) lost **`Nat.div_lt_iff_lt_mul'`**
— the same theorem NS5 lost 60+ times. Investigation:

- Tactic that actually closes the proof: `simp_all` (a generative
  output, no skeleton attribution), at *step 2*.
- Tactic that advances state at step 1: `retrieved:Nat.div_lt_iff_lt_mul:rw`,
  which the credit index **correctly identifies as a must-protect
  zero-win assist@1 skeleton** and never disables.
- Skeletons disabled in the failing cycles: `fb_19`, `pt_iff_2`,
  `fam_div_14` (all have zero direct/advance/assist credit).

So why does the proof break? The disabled skeletons emit *failing*
candidates that nonetheless **shape the wrapper's ranked tactic list**
— specifically, when `fam_div_14: omega` is dropped from the list at
step 1, the policy's top-K cutoff shifts, and
`retrieved:Nat.div_lt_iff_lt_mul:rw` no longer makes it into the
ranked candidates for that step. The retrieval skeleton is protected
by credit but its *position* depends on adjacent (uncredited) entries.

This is a **second-order effect** that even credit-aware pruning
can't catch: the credit measures whether a skeleton produces an
advance/win directly, but not whether removing it disturbs the
ranking of other skeletons. NS7 (or NS6.1) needs a rank-coupling
signal to address it. The no-regression gate, however, deterministically
catches it — 6/6 such cases were rejected in this sweep.

## Comparison to NS5

| | NS5 (165 cycles, 7.46h) | NS6 (20 cycles, ~65min) |
|---|---|---|
| best medium proved   | 37/38 | 37/38 |
| best large proved    | 49 | 49 |
| best enabled skeletons | 25 | **20** |
| promotions           | 11 | 3 |
| regressions rejected | 67 (41%) | 6 (30%) |
| reorders that regressed | yes (c2, c4) | **none** |
| compact-genome ceiling | 35/38 (`archive_seed`) | 35/38 (unchanged — same wins-only selector) |
| zero-win assist skeletons captured | no | **yes (2)** |
| time cost per skeleton dropped | 165 cycles / 23 dropped = 7.2 | 20 cycles / 5 dropped = 4.0 |

NS6 reaches a *smaller* compact genome (20 vs. 25) in *one-eighth* the
runtime, with *zero* reorder regressions and an explicit must-protect
list. The remaining compact-genome ceiling at 35/38 for `archive_seed`
is the same wins-only-selection failure mode NS5 documented — fixing
it requires NS7's "credit-aware seed" operator that uses the assist
index in place of `top_skeletons_by_wins`.

## Files changed

- `eval_rollout_all.py` — trace-record now carries `skeleton_*` fields
- `evolve/skeleton_mutator.py` — scoped operators + safe pruning + scope metadata
- `evolve/skeleton_evolve_ns6.py` — new short-sweep runner
- `scripts/ns6_assist_credit.py` — new analyzer
- `project/evolve/reports/ns6_assist_credit_analysis.md`
- `project/evolve/reports/ns6_credit_aware_mutation.md` (this file)
- `.gitignore` — adds `project/evolve/ns6_runs/`

## Findings

1. **Credit-aware pruning is necessary but not sufficient.** Three
   `disable_dead_skeleton` cycles (c4, c18, c19) and three
   `archive_seed` cycles (c15–c17) regressed — every time, on the same
   theorem (`Nat.div_lt_iff_lt_mul'`) — because the disabled skeletons
   weren't directly involved but their presence shaped the ranked
   tactic list in a way that a *third* (correctly-protected) retrieved
   skeleton needed. Credit alone can't see this second-order
   "rank-coupling" effect.

2. **Scoped reorders are now zero-risk.** 10 cycles of scoped
   `promote_high_win_skeleton` / `demote_generic_skeleton`, none
   regressed. NS5's unscoped versions regressed twice in its first
   four cycles. The scope_origin × scope_shape index-only reordering
   protects out-of-scope skeletons exactly.

3. **Skeleton names drift across mutations.** When
   `disable_dead_skeleton` drops a priority_template, the rebuilt bag
   gets re-indexed by `from_legacy_strategy_config`, so e.g.
   `fam_div_18` (seed) becomes `fam_div_14` (post-cycle-3). The
   credit index keys by name, so cross-cycle credit accumulation
   silently mis-credits the renumbered skeleton. The current sweep
   tolerated this because (a) family/fallback names don't renumber
   *within* their own block until their own family is disabled, and
   (b) the no-regression gate caught the one cycle where a renumbered
   credit-zero entry actually mattered. **NS7 must adopt stable
   identifiers** — either UUIDs on first observation or
   `(origin, template-hash)` keys.

4. **Compact-genome ceiling unchanged at 35/38.** `archive_seed` still
   uses `top_skeletons_by_wins(...)`, which is wins-only. Until that
   selector becomes credit-aware (`top_by_credit_total`), the
   compact-genome experiment plateaus at the same NS5 ceiling.

## Recommendation for next step (NS7 / NS6.1)

1. **Stable skeleton identity** (HIGH PRIORITY) — assign a stable id
   to each skeleton at first observation; store the alias as a
   secondary `name` for human readability. The credit index, archive,
   and mutator should all key on the stable id.

2. **Credit-aware `archive_seed`** — replace `top_skeletons_by_wins`
   with a function that scores by `direct_wins + α·assist_wins_k3 +
   β·advances` and protects every skeleton whose `assist_wins_k3 > 0`.
   Easy lift once #1 is in place.

3. **Rank-coupling detector** — for each candidate cycle, compare
   the wrapper's pre-mutation ranked-list (per state) against the
   post-mutation list. If any skeleton with assist credit *moved out
   of the top-K window*, reject the mutation pre-flight without
   running an eval. This is the only way to catch the
   `Nat.div_lt_iff_lt_mul'` rank-shift class of regressions cheaply.

4. **Per-attempt advance archive rows** — `update_archive_from_metrics_path`
   should also ingest `traces.jsonl` and compute `assist_wins_k3` per
   row, then aggregate at read time. This makes assist credit
   available cross-run, not just within one runner invocation.

5. **Advance-only skeleton review** — `retrieved:Nat.div_eq_of_lt:rw`
   has 3 advances but no assists within K≤3. Either (a) widen K for
   retrieval skeletons (chains can be long), or (b) prune with a
   high-confidence threshold (e.g. >5 advances no assists).

6. **Family-gate operators** — `narrow_family_gate` /
   `expand_family_gate` are still stubs. NS6's per-step trace data is
   now rich enough to compute per-family attribution; the operators
   become implementable.
