# NS5 — Skeleton Evolution Plan

Status: in-progress (autonomous overnight task).
Date: 2026-05-22.
Branch: `ns5-skeleton-evolution`.
Predecessor commits: NS4.2 4a61ea1, NS4.1 28a3b0c, NS4 88a739b.

## 1. What NS4 achieved

NS4 turned the proof-search wrapper's emission paths into a unified
*skeleton-bag* representation. Concretely:

- Every non-generative emit path (`priority_template`, `family_tactic`,
  `fallback_tactic`, `tactic_template`, `term_builder`,
  `retrieved_premise`) now flows through `evolve.skeleton_bag.SkeletonBag`
  when `use_skeleton_bag=True`.
- Each emitted tactic carries skeleton attribution
  (`skeleton_name`, `skeleton_shape`, `skeleton_family`,
  `skeleton_specificity`, `skeleton_priority`) all the way to the
  per-theorem result row and the `metrics.json` aggregates.
- The skeleton-bag path is bit-for-bit identical to the legacy path on
  both `nat_defs_medium` (37/38) and `nat_defs_large_v5` (49/64).
- Only `generative_topk` remains outside the bag — by design.

In other words: **the genome is no longer a bag of flat strings, it is a
bag of named, shape-gated, family-gated, priority-banded skeletons.**

## 2. Why the bag is now ready for mutation

A skeleton has three properties that a raw tactic string does not:

1. **Identity.** Every skeleton has a stable `skeleton_name`
   (`pt_iff_8`, `fam_mod_30`, `fb_36`, `retrieved:Nat.div_lt_iff_lt_mul:rw`,
   …). Wins, advances, attempts, and regressions can therefore be
   attributed *per skeleton*, not per-string. The same template emitted
   in two different shapes is two different skeletons.
2. **Structure.** Each skeleton has a `shape`, a `family`, a
   `priority`, a `specificity`, and an `enabled` flag. These are the
   mutable axes: a single mutation may toggle `enabled` to False,
   change `shape=any` to `shape=iff`, or change the family gate from
   None to `"div"`.
3. **Observability.** `metrics.json` already emits
   `skeleton_proved_counts`, `skeleton_proved_counts_by_family`,
   `skeleton_proved_counts_by_shape`,
   `skeleton_specificity_proved_counts`, `skeleton_wins`,
   `skeleton_attempt_count`, `skeleton_advanced_count`. The signal a
   mutator needs is already in the eval output — NS5 only has to read
   it.

That is enough to do the AlphaEvolve loop: archive → guided mutation →
eval → archive update.

## 3. What counts as skeleton-level mutation

NS5 limits itself to **safe, archive-guided** operators. Each takes the
current bag (or the legacy strategy-config dict it adapts from) and
produces a new bag with the same shape-language but a different
emission profile.

| Operator | Effect | Safety |
|----------|--------|--------|
| `disable_dead_skeleton`     | Set `enabled=False` on skeletons with 0 wins and many attempts. | Never disables a skeleton with ≥1 archived win. |
| `promote_high_win_skeleton` | Move a high-win skeleton earlier within its shape/family band. | No-op when its specificity/priority is already minimal. |
| `demote_generic_skeleton`   | Within a shape band, push generic skeletons after specific ones. | NS1 already enforces this; reapplied for invariant. |
| `clone_skeleton_to_shape`   | Clone a high-win skeleton from one shape to a compatible shape. | Only `iff → any`, `eq → iff`, `lt → le`, `le → lt`. Cloned skeleton starts disabled until enabled by the runner. |
| `narrow_family_gate`        | If a skeleton wins only in one family, set `family=that_family`. | Only when the archive confirms zero wins outside that family. |
| `expand_family_gate`        | If a skeleton wins in multiple families, set `family=None`. | Reversible by `narrow_family_gate`. |
| `budget_trim`               | Reduce `priority_template_budget` / `family_budgets` after dead-attempt streaks. | Lower bound: 1. |
| `archive_seed`              | Build a candidate from top archived skeletons only. | Used for the *compact genome* experiment. |

We deliberately do **not** implement raw random mutation, slot-vocabulary
mutation, or template-text edits. Those belong in a later (NS5.x / NS6)
phase once the archive has enough data to constrain them.

## 4. What should not be mutated yet

- `generative_topk` — it is the base policy's beam-search output. Not a
  skeleton.
- `retrieval_*` knobs (`retrieval_top_k`, `retrieval_tactic_forms`,
  `retrieval_shape_filter`, …) — they are dynamic-skeleton parameters,
  not mutable skeletons. Changing them invalidates the per-call
  diagnostics parity NS4.2 preserved.
- Template text (the `template` string of a `Skeleton`). Changing it
  produces a skeleton with the same name but different behaviour —
  poisons the archive. Slot-vocabulary mutation is NS6.
- Goal-shape classifier or the `_match_families` function.
- JSON schema of `strategy_config.json`.

If a mutation crashes Dojo on a theorem, NS5 falls back to the existing
`theorem_tactic_denylist` mechanism: the offending (theorem, tactic)
pair is added to the deny list for the current candidate only. No
candidate is promoted to default if it regresses.

## 5. How success is measured

A candidate produced in cycle N is **accepted** when:

1. `proved_count(nat_defs_medium) >= 37` (no regression vs. NS4.2 best).
2. AND at least one of:
   a. `proved_count(nat_defs_medium) > 37` — a real improvement.
   b. `proved_count(nat_defs_medium) == 37` AND
      `proved_count(nat_defs_large_v5) > 49` — large-set transfer gain.
   c. `proved_count == 37` AND `len(enabled_skeletons) < baseline_count`
      AND `skeleton_attempt_count` strictly lower (compact genome wins).
   d. `proved_count == 37` AND wall-clock < baseline (faster).

Additional measurements logged for the report:

- Dead-skeleton count (skeletons with `attempts > N_DEAD_MIN` and
  `wins == 0`). Smaller is better.
- Skeleton-winner diversity (`len(skeleton_proved_counts)`). Higher is
  more robust.
- Compact-archive transfer: does the *archive_seed* candidate (smallest
  skeleton set that proves ≥ 37) preserve 37/38?

## 6. Expected ceiling

This is an honest section: NS5 is not retraining the policy and is not
adding new templates. It is reshaping the existing skeleton population.

- `nat_defs_medium` is likely to stay at 37/38. The remaining failure
  (`Nat.AM_GM`) is an environment / proof-content limitation rather
  than a skeleton-ordering problem — multiple prior waves established
  that the base policy alone cannot close it on `gen_v5`. Reaching 38
  would require either a new template or a model retrain, neither of
  which is in scope.
- `nat_defs_large_v5` is more plausible. 49/64 was hit with the same
  skeletons as nat_defs_medium; large-set theorems often want
  different `shape`/`family`/`priority` settings than the medium tuned
  for. Even a small bag-reordering (e.g. cloning `pt_iff_8` into
  another shape or narrowing a family gate) could surface a new
  proof on a large-only theorem.
- A *negative* result (no improvement on either set, but a smaller
  genome that preserves 37/38 and 49/64) is itself a useful publishable
  outcome — it isolates *which* skeletons are doing all the work.

## 7. Run shape

- Min runtime: 6 hours (`--min-hours 6`).
- Max runtime: 8 hours (`--max-hours 8`).
- Primary set: `nat_defs_medium` (38 theorems, ~3 min/eval typical).
- Secondary set: `nat_defs_large_v5` (~6 min/eval typical) — run only
  when medium passes the no-regression check.
- Cycle plan (target counts; real count adapts to time):
  1. 1× baseline reproduction (medium + large) for archive seeding.
  2. 5× archive-guided mutations on medium.
  3. 5× dead-skeleton-prune variants on medium.
  4. 5× ordering / budget variants on medium.
  5. 3× large-set evaluations of the best survivors.
  6. 1× compact-archive seed candidate (medium + large).
  7. Reserve for cleanup, reporting, large transfers.

If any cycle leaves the medium baseline unmet, the candidate is
**not** promoted and the genome reverts to the prior best.

## 8. Non-goals

- No new theorem set, no new templates, no new retrieval lemmas.
- No mutation of the JSON config schema (`use_skeleton_bag` is the
  only new field NS4 added; NS5 does not add more).
- No checkpoint changes, no retraining, no model-weight changes.
- No removal of legacy code paths. The bag is the *new* layer; the
  legacy path remains the default.
- No commits of run artifacts. Reports and code only.
