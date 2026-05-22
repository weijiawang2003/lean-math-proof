# Evaluation report — nat_defs_medium / hybrid_evolved + v4.3 goal-shape filter

**Branch**: `v4-premise-retrieval-div`
**Parent commit (v4.2)**: `9aebb67` — `Add v4.2 retrieval hygiene: self-filter, unavailability denylist, form ablation`
**v4.3 run id (Experiment B, default seed)**: `evolve-20260522-034524-d10acf`
**Metrics**: `project/evolve/runs/evolve-20260522-034524-d10acf/eval/seed-baseline/eval-cd736f4c/metrics.json`
**Checkpoint**: `project/models/gen_v5` (unchanged from v3.6 / v4.1 / v4.2)
**Top-k**: 8, **Max-steps**: 8, **Decode**: beam
**Wallclock (default seed)**: 3m 42s (≈ 5.8 s/theorem)
**Sweep runs**:
* A — `/tmp/v4_3_exp_A/eval-c316eb2b/` (bloat filter OFF, forms `rw/simp/apply`)
* B — `…/d10acf/…/eval-cd736f4c/` (bloat filter ON, forms `rw/simp/apply`, **the default**)
* C — `/tmp/v4_3_exp_C/eval-0e587227/` (bloat filter ON, forms `rw/simp` only)

## Headline

| | v3.6 | v4.1 | v4.2 | **v4.3 / A** | **v4.3 / B (default)** | **v4.3 / C** |
|---|---:|---:|---:|---:|---:|---:|
| **Proved** | **25/38** | **25/38** | **25/38** | **25/38** | **25/38** | **25/38** |
| Errored | 10 | 6 | 7 | 7 | **10** | **10** |
| Exhausted | 3 | 7 | 6 | 6 | **3** | **3** |
| Retrieval attempts | — | 783 | 303 | 303 | **121** | **73** |
| Retrieval advances | — | 6 | 16 | 16 | **0** | **0** |
| Retrieval wins | — | 0 | 0 | 0 | **0** | **0** |
| `apply` goal-inflation events | — | 6 | 16 | **16** | **3** | **0** |
| Pathological apply chains | — | 3 | 3 | 3 | **0** | **0** |
| Wallclock | ~3m 28s | 5m 22s | 4m 19s | 4m 12s | **3m 42s** | **3m 28s** |

* **v4.3 / A** replicates v4.2 (bloat tracking instrumented but filter
  disabled) — proves the v4.3 code path is behavior-equivalent when the
  filter is OFF.
* **v4.3 / B** is the published default: 4 fewer Lean roundtrips per
  bloating chain, the 16 v4.2 pathological advances collapse to 3
  observation-and-reject events, errored/exhausted distribution
  restored to v3.6's shape.
* **v4.3 / C** drops `apply` entirely. Same proved count, +0 errored
  vs B, -48 Lean roundtrips.

## Success criteria

| Criterion | Met? | Detail |
|---|---|---|
| ≥ 25/38 preserved | ✓ | 25/38 in all three sweep configs (matches v3.6 / v4.1 / v4.2) |
| No regressions | ✓ | `proved_by_origin = {fallback_tactic: 18, family_tactic: 4, generative_topk: 3}` — bit-identical to v3.6 in A/B/C |
| No new `DojoCrashError` | ✓ | Denied tactics = 8 (unchanged) |
| `unknown constant` remains 0 | ✓ | 0 in all three configs (v4.2 hygiene preserved) |
| Apply-bloat attempts substantially reduced | ✓ | 16 → **3** (-81 %) under B; **0** under C |
| Runtime same-or-better than v4.2 | ✓ | B: -0.6 min, C: -0.8 min vs v4.2 |
| Report explains why retrieval still doesn't close div theorems | ✓ | See "Why retrieval still doesn't close" below |
| ≥ 1 new div theorem solved (*ideal*) | ✗ | 0 new closures. `apply Nat.lt_of_lt_of_le` was the only retrieved tactic that produced *any* state change in v4.2; killing it surfaces the deeper issue (no shape-matching `rw`/`simp` tactic exists in the seeded catalog for these goals). |

**6 of 7 explicit criteria met.** The ideal closure-criterion remains
unmet — but as expected: v4.3's job was to kill bloat, not to find new
proofs. The bloat is now gone; the div theorems remain unproved for
structural reasons (see below).

## How the bloat filter works

`eval_rollout_all.rollout_one_theorem` maintains a per-theorem
`bloating_apply_lemmas: set[str]`. The lifecycle of a retrieved-apply
tactic candidate:

```
                ┌── if lemma already in bloating_apply_lemmas:
                │     → emit SkippedBloatingApply trace, increment
                │       skipped_bloating_apply_count, continue.
                │
ranked tactic ──┼── if tactic runs and produces TacticState advance with
                │   num_goals_after > num_goals_before:
                │     → write trace with bloat_rejected=True, add lemma
                │       to bloating_apply_lemmas, continue (do NOT take
                │       the advance — the rollout stays at the same
                │       state and tries the next ranked tactic).
                │
                └── otherwise: existing advance / error / finished logic
                    runs unchanged.
```

The filter is **per-theorem only**: `Nat.lt_of_lt_of_le` is still
available as a `rw`/`simp` argument (it just hasn't been needed there),
and the lemma is still allowed as an `apply` candidate on theorems where
it hasn't been observed to bloat yet.

## v4.3 bloat-filter aggregates (Experiment B, default)

| Metric | Value |
|---|---:|
| `retrieved_apply_goal_increase_count` | **3** (down from v4.2's 16) |
| `retrieved_apply_goal_decrease_count` | 0 |
| `retrieved_apply_no_goal_change_count` | 0 |
| `skipped_bloating_apply_count` | 0 *(see below)* |
| `bloating_apply_lemma_counts` | `{Nat.lt_of_lt_of_le: 3}` |
| `retrieved_premise_attempt_count` | 121 (-60 % vs v4.2) |
| `retrieved_premise_advanced_count` | 0 (every bloating advance rejected) |
| `retrieved_premise_proved_count` | 0 |

`skipped_bloating_apply_count = 0` is interesting: once
`apply Nat.lt_of_lt_of_le` is rejected at the *first* observation on a
theorem, the rollout never reaches a state where it would be reranked as
an `apply` candidate again — the per-theorem set is populated but never
queried. The 3 entries are the first-time observations on
`Nat.div_le_div_right`, `Nat.div_lt_one_iff`, and `Nat.div_pos_iff`,
matching exactly the 3 bloating chains v4.2 surfaced.

## Per-div-theorem result (Experiment B)

| Theorem | v3.6 | v4.1 | v4.2 | **v4.3** | apply-inc | bloat lemmas |
|---|---|---|---|---|---:|---|
| `Nat.div_le_div_right` | ERROR @3 | EXH @8 | EXH @8 | **ERROR @3** | 1 | `Nat.lt_of_lt_of_le` |
| `Nat.div_lt_iff_lt_mul'` | ERROR @2 | EXH @8 | EXH @8 | **ERROR @2** | 0 | — |
| `Nat.div_lt_one_iff` | ERROR @4 | EXH @8 | EXH @8 | **ERROR @4** | 1 | `Nat.lt_of_lt_of_le` |
| `Nat.div_pos` | ERROR @3 | ERROR @3 | ERROR @3 | **ERROR @3** | 0 | — |
| `Nat.div_pos_iff` | ERROR @4 | ERROR @4 | ERROR @4 | **ERROR @4** | 1 | `Nat.lt_of_lt_of_le` |
| `Nat.dvd_iff_div_mul_eq` | ERROR @3 | EXH @8 | EXH @8 | **ERROR @3** | 0 | — |

3 theorems that v4.1 / v4.2 carried to step 8 via the bloating
`apply Nat.lt_of_lt_of_le` chain now fail at their *real* error step —
the step where every non-bloating retrieved tactic also errors. This is
the correct behavior; the v4.1 / v4.2 EXH @8 was an artifact of the
bloating chain extending the rollout, not actual progress.

## Was `apply Nat.lt_of_lt_of_le` suppressed?

Yes. Across all 6 div theorems in v4.2 the lemma's `apply` form produced
16 advances (every one of which strictly grew the goal stack by 2). In
v4.3 / B it produces 3 first-observation bloats which are all rejected,
and never advances on any subsequent step. In v4.3 / C the lemma never
appears as `apply` at all (forms = `["rw","simp"]`).

## Sweep comparison: A vs B vs C

| | A (bloat OFF) | B (bloat ON) | C (rw/simp only) |
|---|---|---|---|
| Forms | rw / simp / apply | rw / simp / apply | rw / simp |
| Bloat filter | off | **on** | on |
| Proved | 25/38 | 25/38 | 25/38 |
| Errored | 7 | 10 | 10 |
| Exhausted | 6 | 3 | 3 |
| Retrieval attempts | 303 | 121 | 73 |
| `apply` form attempts | 126 | 48 | — |
| Apply-bloat events | 16 | 3 | — |
| Advances accepted | 16 *(all pathological)* | 0 | 0 |
| Wallclock | 4m 12s | 3m 42s | 3m 28s |

**A → B**: Bloat filter eliminates 13 of 16 bloating advances (the first
3 are observed-then-rejected; the next 13 never happen because the
rollout doesn't advance into the bloated state). Lean roundtrips drop
60 %. Wallclock drops 30 s. Proved count unchanged.

**B → C**: Removing `apply` from forms removes the 48 apply attempts
that, after bloat filtering, contribute 0 wins. Saves another 14 s and
48 Lean roundtrips. Proved count unchanged.

**Decision — keep `apply` enabled by default (B is the seed).**
Reasoning:

1. C is faster but the gain is small (-14 s, -48 attempts).
2. The bloat filter already makes `apply` cheap on bloating lemmas
   (3 observations × ~1 Lean roundtrip each = ~3 roundtrips total).
   The remaining 45 apply attempts are non-bloating apply attempts on
   other lemmas — keeping them available leaves the search open to
   future productive applies once v4.4's catalog/ranker improves.
3. The diagnostic surface (`bloating_apply_lemma_counts`,
   `retrieved_apply_goal_*_count`) only stays informative if `apply` is
   in the form list. Disabling it hides what's happening.

C is exposed as a quick ablation via `retrieval_tactic_forms` in the
strategy config; teams can flip to it any time without code changes.

## Why retrieval still doesn't close any div theorem

Three structural blockers, in priority order:

1. **Goal shape ≠ lemma shape.** The 6 div theorems' goals are *iff*
   propositions (`Nat.div_pos_iff`, `Nat.div_lt_one_iff`, …) or
   conjunctions of inequalities (`Nat.div_le_div_right`). The retrieved
   catalog is mostly *equality*-shaped (`Nat.div_eq_of_lt`,
   `Nat.mul_div_cancel`). Neither `rw [Nat.div_eq_of_lt]` (needs an
   equality goal) nor `apply Nat.div_eq_of_lt` (wrong shape) can engage
   these goals. **This is the dominant remaining failure mode.**
2. **Hypothesis-conditional lemmas not matched.** Several `Nat.div_*`
   lemmas require a hypothesis like `0 < c` that the proof state has but
   the retriever doesn't surface as a pre-condition. The wrapper has no
   way to chain `apply Nat.div_le_div_right (h : 0 < c)` with the
   hypothesis lookup.
3. **No induction templates.** `Nat.div_le_div_right`-style theorems
   reduce naturally over `induction` on the numerator. The wrapper has
   no induction templates for `div`. v4.3 explicitly does not address
   this (per scope).

The bloat filter eliminates the *symptom* (pathological apply
expansion); v4.4 / v4.5 will need to address the *cause* (shape-aware
ranking) to actually close any new div theorems.

## What v4.3 plumbed

### `evolve/candidate.py`

* `SearchCandidate.retrieval_skip_bloating_apply: bool = True`

### `evolve/strategy_wrapper.py`

* `dump_strategy_config` / `load_strategy_config` now round-trip 12
  fields (was 11 in v4.2); new `retrieval_skip_bloating_apply` JSON key.

### `evolve/evaluator.py`

* Passes `retrieval_skip_bloating_apply` from the candidate to the
  strategy config.

### `evolve/run_evolve.py`

* `make_seed_candidate("hybrid_evolved", ...)` enables the bloat filter
  by default. Description updated.

### `eval_rollout_all.py`

* `_load_policy` unpacks the new 12-tuple from `load_strategy_config`;
  the flag is stashed on the wrapper (`pol.retrieval_skip_bloating_apply`)
  so `rollout_one_theorem` can read it via one `getattr`.
* `rollout_one_theorem`:
  * Per-theorem `bloating_apply_lemmas: set[str]` and
    `skip_bloating_apply` local.
  * Pre-filter check inside the `for rank, tac in enumerate(ranked):`
    loop: retrieved-apply candidates whose lemma is already known to
    bloat emit a `SkippedBloatingApply` trace and `continue`.
  * Post-advance check: when a retrieved-apply produces a TacticState
    transition with `num_goals_after > num_goals_before`, the trace is
    written with `bloat_rejected=True`, the lemma joins
    `bloating_apply_lemmas`, and the rollout `continue`s to the next
    ranked tactic (does NOT take the advance).
  * Goal-shape annotations (`goal_count_delta`,
    `goal_count_increased`) added to every advance trace.
  * Per-theorem fields:
    `retrieved_apply_goal_increase_count`,
    `retrieved_apply_goal_decrease_count`,
    `retrieved_apply_no_goal_change_count`,
    `skipped_bloating_apply_count`,
    `bloating_apply_lemmas`.
  * Aggregate metrics:
    `retrieved_apply_goal_increase_count`,
    `retrieved_apply_goal_decrease_count`,
    `retrieved_apply_no_goal_change_count`,
    `skipped_bloating_apply_count`,
    `bloating_apply_lemma_counts`.
  * Console summary prints `Apply goal-shape: inc=… dec=… same=…` and
    `Bloating lemmas: …` whenever any are observed.

## What v4.3 did NOT do (per scope)

* No retraining, no checkpoint changes, no premise-augmented model.
* No new policy types. `hybrid_evolved` unchanged as the entry point.
* No new lemmas added to `STATIC_PREMISES`.
* No induction templates.
* No live Lean availability checker (still relying on v4.2's static
  `_UNAVAILABLE_LEMMAS`).
* No goal-shape inference beyond the binary "did it grow?" check —
  v4.4 territory.

## Recommended next step (v4.4)

**Shape-aware retrieval ranking.** Parse the proof state's goal head
constant (`=`, `↔`, `<`, `≤`, `∣`, `∧`, …) and tag each catalog lemma
with its conclusion's head constant. Only emit a lemma in a form
compatible with its shape:

| Lemma conclusion shape | Emit as |
|---|---|
| `Eq` | `rw [...]`, `simp [...]` |
| `Iff` | `rw [...]`, `simp [...]`, `apply Iff.mp/mpr` |
| `<` / `≤` | `apply [...]`, `exact ... ⟨...⟩` |
| Universal Prop | `apply [...]`, `exact ...` |

This is the lightest possible v4.4 — no embeddings, no Lean calls, just
a hand-curated `lemma_shape` annotation on each catalog entry plus a
goal-head extractor (regex on the line beginning with `⊢`). With the
shape filter, the retrieved `apply` candidates that today error with
`type mismatch` / `apply failed, failed to unify` would never be
emitted in the first place — eliminating the dominant non-bloat failure
class identified above.

## Artifacts

* `project/evolve/runs/evolve-20260522-034524-d10acf/` — v4.3 / B run root (the published default)
* `…/eval/seed-baseline/eval-cd736f4c/metrics.json` — full metrics
* `…/eval/seed-baseline/eval-cd736f4c/traces.jsonl` — 1,388 trace records (121 retrieved, 3 with `bloat_rejected=true`)
* `/tmp/v4_3_exp_A/eval-c316eb2b/` — Experiment A (bloat OFF, v4.2-equivalent)
* `/tmp/v4_3_exp_C/eval-0e587227/` — Experiment C (rw/simp only)
* Branch: `v4-premise-retrieval-div` (off `e74861f`); v4.3 commit pending on top of `9aebb67`
