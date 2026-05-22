# Evaluation report — nat_defs_medium / hybrid_evolved + v4.2 retrieval hygiene

**Branch**: `v4-premise-retrieval-div`
**Parent commit (v4.1)**: `fd6256d` — `Add premise retrieval hooks for div-family tactic search`
**v4.2 run id**: `evolve-20260522-025521-bc3e5a`
**Metrics**: `project/evolve/runs/evolve-20260522-025521-bc3e5a/eval/seed-baseline/eval-e36049a6/metrics.json`
**Checkpoint**: `project/models/gen_v5` (unchanged from v3.6 / v4.1)
**Top-k**: 8, **Max-steps**: 8, **Decode**: beam
**Wallclock**: 4m 19s (≈ 6.8 s/theorem)
**Baseline B (retrieval off) run**: `/tmp/v4_2_baseline_B/eval-6446e8a4`

## Headline

| | gen_v5 | v3.6 | v4.1 | **v4.2** | Baseline B (v4.2 wrapper, retrieval off) |
|---|---:|---:|---:|---:|---:|
| Proved | 3/38 | 25/38 | 25/38 | **25/38** | 25/38 |
| Δ over v3.6 | — | 0 | 0 | **0** | 0 |
| Errored | — | 10 | 6 | **7** | 10 |
| Exhausted | — | 3 | 7 | **6** | 3 |
| Retrieval attempts | — | — | 783 | **303** | 0 |
| `unknown constant` count | — | — | 200 | **0** | 0 |
| Filtered (self) | — | — | — | **32** | — |
| Filtered (unavailable) | — | — | — | **194** | — |
| Wallclock | — | ~3m 28s | 5m 22s | **4m 19s** | ~3m 28s |

v4.2 ships the **hygiene layer** for v4.1's retrieval — three filters and a
form-and-rank cleanup. The proved count is unchanged, all 200 v4.1
`unknown constant` errors are eliminated, and the retrieval wallclock drops
from `v4.1 − v3.6 ≈ +114 s` to `v4.2 − v3.6 ≈ +51 s` (-55 % overhead).

## Success criteria

| Criterion | Met? | Detail |
|---|---|---|
| ≥ 25/38 preserved | ✓ | 25/38 (matches v3.6 / v4.1) |
| No regressions on previously-solved theorems | ✓ | `proved_by_origin = {fallback_tactic: 18, family_tactic: 4, generative_topk: 3}` — bit-identical to v3.6 |
| No new `DojoCrashError` | ✓ | Per-theorem deny-list count = 8 (unchanged) |
| Substantial reduction in unknown-constant retrieved attempts | ✓ | 200 → **0** (`-100 %`) |
| Eliminate target-theorem self-retrieval | ✓ | 32 self-filter hits across 6 div theorems; 0 self-references reach Lean |
| Runtime remains acceptable | ✓ | 4m 19s (-1m 03s vs v4.1; +51 s vs v3.6) |
| ≥ 1 new div-family theorem solved (*ideal*) | ✗ | 0 new div wins. 16 retrieved-tactic state advances all from one pathological pattern (see "Why no new wins"). |

**6 of 7 criteria met.** The ideal criterion is the same one v4.1 missed; v4.2's
job was hygiene, not new closures.

## v4.2 retrieval aggregates

| Metric | v4.1 | v4.2 | Δ |
|---|---:|---:|---:|
| `retrieved_premise_activation_count` | 6 | 6 | 0 (still all 6 div theorems) |
| `retrieved_premise_attempt_count` | 783 | 303 | **−480 (−61 %)** |
| `retrieved_premise_advanced_count` | 6 | 16 | +10 (deeper search reached after filtering errors) |
| `retrieved_premise_proved_count` | 0 | 0 | 0 |
| `retrieved_premise_wins` | `[]` | `[]` | — |
| `retrieved_premise_filtered_self_count` | — | **32** | new |
| `retrieved_premise_filtered_unavailable_count` | — | **194** | new |
| Form attempts `{rw, simp, apply}` | n/a (4 forms) | `{rw: 70, simp: 107, apply: 126}` | dropped `exact` (zero wins in v4.1) |
| Form-level advances | n/a | `{apply: 16}` | all 16 advances from `apply` |

The 16 `apply` advances are all `apply Nat.lt_of_lt_of_le` — see "Why no new wins".

## Failure-mode breakdown (303 v4.2 retrieved-tactic outcomes)

| Outcome class | v4.1 | v4.2 |
|---|---:|---:|
| `unknown constant` | 200 | **0** |
| `simp made no progress` | 207 | 107 |
| `tactic 'rewrite' failed` | 138 | 70 |
| `tactic 'apply' failed, failed to unify` | 108 | 110 |
| `type mismatch` | 124 | 0 (subsumed into apply/rewrite_failed in v4.2 categorization) |
| state-advancing transition | 6 | 16 |

The drop in `simp made no progress` (-100) and `tactic 'rewrite' failed` (-68) is
collateral benefit of the catalog shrinking: with 7 lemmas removed
(self-references + the 2 truly-unavailable lemmas), each retrieval call emits
fewer `simp [...]` / `rw [...]` attempts.

## Per-div-theorem result

| Theorem | v3.6 | v4.1 | **v4.2** | retr attempts (v4.2) | retr advances | filtered self | filtered unavail |
|---|---|---|---|---:|---:|---:|---:|
| `Nat.div_le_div_right` | ERROR @3 | EXH @8 | **EXH @8** | 56 | 6 (`apply Nat.lt_of_lt_of_le`) | 6 | 36 |
| `Nat.div_lt_iff_lt_mul'` | ERROR @2 | EXH @8 | **EXH @8** | 67 | 0 | 4 | 30 |
| `Nat.div_lt_one_iff` | ERROR @4 | EXH @8 | **EXH @8** | 78 | 10 (`apply Nat.lt_of_lt_of_le`) | 8 | 50 |
| `Nat.div_pos` | ERROR @3 | ERROR @3 | **ERROR @3** | 28 | 0 | 3 | 18 |
| `Nat.div_pos_iff` | ERROR @4 | ERROR @4 | **ERROR @4** | 27 | 0 | 4 | 24 |
| `Nat.dvd_iff_div_mul_eq` | ERROR @3 | EXH @8 | **EXH @8** | 47 | 0 | 7 | 36 |

3 theorems escape the early-error trap and run all 8 steps (same as v4.1).
The 2 ERROR @3-4 theorems are deep-state-of-elaboration failures unrelated to
retrieval — the generative top-k and family/fallback tactics all error at
the same step. Filtering caught roughly 6 self-references and 25 unavailable
references per div theorem.

## Why no new wins

All 16 retrieved-tactic state advances are `apply Nat.lt_of_lt_of_le` on
`Nat.div_le_div_right` (6) and `Nat.div_lt_one_iff` (10). Each `apply`
introduces 2 fresh subgoals (`?a < ?b`, `?b ≤ ?c`) that the wrapper then
cannot close — so the goal stack grows 2 → 4 → 6 → 8 → 10 → 12 across steps.
Visible in `metrics.json` for `Nat.div_lt_one_iff`:

```
num_goals_before: 2  → num_goals_after: 4
num_goals_before: 4  → num_goals_after: 6
…
num_goals_before: 10 → num_goals_after: 12
```

This is the same pathological pattern v4.1 exhibited. v4.2 makes it slightly
worse (16 vs 6 advances) because the catalog now has 9 surviving candidates
per state instead of v4.1's 16, so `apply Nat.lt_of_lt_of_le` is reached
faster in the ranking when it survives `rw`/`simp` errors.

**v4.3 will need a goal-shape filter for `apply`** that rejects lemma
applications which strictly increase the goal count more than once in a row.

## What v4.2 plumbed

### `premise_retriever.py`

* `_UNAVAILABLE_LEMMAS: set[str]` — 7-entry static denylist. Two classes:
  * **Genuinely out of import-closure:** `Nat.div_eq_zero_iff`,
    `Nat.div_le_iff_le_mul` (38 + 46 = 84 v4.1 `unknown constant` errors)
  * **Forward-reference traps** (target theorems unknown at proof
    position of other in-file targets): `Nat.div_le_div_right`,
    `Nat.div_lt_one_iff`, `Nat.div_pos`, `Nat.div_pos_iff`,
    `Nat.dvd_iff_div_mul_eq` (116 v4.1 `unknown constant` errors).
* `_name_namespace_variants(theorem_name)` — exact and unqualified-tail
  name variants for the self-filter.
* `retrieve_for_state(...)` extended with `filter_self`, `filter_unavailable`,
  `return_diagnostics` kwargs. When `return_diagnostics=True` returns
  `(premises, {filtered_self: int, filtered_unavailable: int})`.
* Scoring: kept token-overlap; added family-token bonus (`+0.5` when the
  lemma shares the family token with the theorem name).

### `evolve/strategy_wrapper.py`

* `_FORM_FAMILY_TEMPLATES` — short-name aliases `rw → "rw [{p}]"`, etc., so
  candidate configs can ship `["rw","simp","apply"]` and get the full
  templates expanded automatically.
* `_form_family_label(template)` — canonical form-family label
  (`"rw [{p}]" → "rw"`).
* `StrategyWrapperPolicy` gains `retrieval_filter_self`,
  `retrieval_filter_unavailable` kwargs; each ranked entry now carries a
  6th field (`retrieved_form`) alongside the existing 5.
* `last_retrieval_filtered_self_count`,
  `last_retrieval_filtered_unavailable_count` — per-call diagnostic counts
  surfaced for the eval loop.

### `eval_rollout_all.py`

* Per-theorem result dict gains:
  `retrieved_premise_attempt_by_form`,
  `retrieved_premise_advanced_by_form`,
  `retrieved_premise_filtered_self_count`,
  `retrieved_premise_filtered_unavailable_count`,
  `winning_tactic_retrieved_form`.
* Trace records tagged with `tactic_retrieved_form`.
* Aggregate metrics:
  `retrieved_premise_form_counts`,
  `retrieved_premise_form_success_counts`,
  `retrieved_premise_filtered_self_count`,
  `retrieved_premise_filtered_unavailable_count`.
* Console summary prints retrieval filter + form lines.

### `evolve/candidate.py`

* `SearchCandidate.retrieval_filter_self: bool = True`
* `SearchCandidate.retrieval_filter_unavailable: bool = True`

### `evolve/evaluator.py` + `evolve/run_evolve.py`

* `dump_strategy_config` / `load_strategy_config` round-trip the two
  filter flags; the JSON schema now has 11 fields (was 9 in v4.1).
* `make_seed_candidate("hybrid_evolved", ...)` seed defaults updated:
  `retrieval_top_k = 8` (was 10),
  `retrieval_tactic_forms = ["rw","simp","apply"]` (was empty → 4 forms),
  `retrieval_filter_self = True`,
  `retrieval_filter_unavailable = True`.

## What v4.2 did NOT do (per scope)

* No retraining, no checkpoint changes, no premise-augmented model.
* `gen_v5` unmodified; `hybrid_evolved` is still the only entry point.
* v3.6 reports and `nat_defs_medium_v3_6.md` are untouched.
* No new policy types.
* No expansion of `AM_GM` or induction work.
* No live Lean availability checker — `_UNAVAILABLE_LEMMAS` is the static
  v1 of that. v4.3 should implement a real Lean `#check`-style probe so the
  denylist isn't manually curated.
* No induction-template work — saved for v4.3.

## Decided next steps

1. **v4.3 — goal-shape-aware `apply` filter.** Reject `apply LEMMA`
   candidates that would increase the goal count and whose previous
   `apply` on the same lemma already did so. The current 16-advance
   pathological pattern is the dominant remaining waste.
2. **v4.3 — live Lean availability probe.** Replace the static
   `_UNAVAILABLE_LEMMAS` with a one-time-per-run availability cache:
   for each candidate lemma, attempt `#check LEMMA` against a stub
   theorem in the eval file's import context. Cache the boolean.
   This unblocks safely extending the catalog beyond hand-verified lemmas.
3. **v4.3 — induction templates for div recursion.** `Nat.div_le_div_right`
   needs `induction n with | zero => simp | succ n ih => simp [Nat.div_succ, ih]`-shaped
   tactics that the current wrapper has no template for.

## Comparison details

| | v3.6 | v4.1 | **v4.2** |
|---|---:|---:|---:|
| Proved | 25 | 25 | **25** |
| Errored | 10 | 6 | **7** |
| Exhausted | 3 | 7 | **6** |
| Avg steps (proved) | 1.1 | 1.1 | **1.1** |
| Fallbacks used | 24 | 24 | **24** |
| Family activations | `{div: 6, mod: 5, AM_GM: 1}` | `{div: 6, mod: 5, AM_GM: 1}` | `{div: 6, mod: 5, AM_GM: 1}` |
| Family proofs | `{mod: 4}` | `{mod: 4}` | `{mod: 4}` |
| Denied tactics | 8 | 8 | **8** |
| Retrieved-premise attempts | — | 783 | **303** |
| Retrieved-premise advances | — | 6 | **16** |
| Retrieved-premise proofs | — | 0 | **0** |
| `unknown constant` errors | — | 200 | **0** |
| Retrieval activations | — | 6 | **6** |
| Wallclock | ~3m 28s | 5m 22s | **4m 19s** |

Across all three runs the 25 proved theorems and their winning tactics are
bit-identical. v4.2's contribution is the hygiene layer between v4.1's
plumbing and the v4.3 inference work.

## Artifacts

* `project/evolve/runs/evolve-20260522-025521-bc3e5a/` — v4.2 run root
* `…/eval/seed-baseline/eval-e36049a6/metrics.json` — full metrics
* `…/eval/seed-baseline/eval-e36049a6/traces.jsonl` — 1,604 trace records (303 tagged `retrieved_premise`)
* `…/eval/seed-baseline/strategy_config.json` — dumped config (includes
  `retrieval_filter_self=true`, `retrieval_filter_unavailable=true`,
  `retrieval_tactic_forms=["rw","simp","apply"]`, `retrieval_top_k=8`)
* `/tmp/v4_2_baseline_B/` — Baseline B (retrieval off) artifacts;
  confirms 25/38 with same `proved_by_origin` regardless of retrieval state
* Branch: `v4-premise-retrieval-div` (off `e74861f`)
