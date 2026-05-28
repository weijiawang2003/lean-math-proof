# NS4.1 — Skeleton-Bag Unification Report

Status: complete.
Date: 2026-05-22.
Branch: `ns4-skeleton-bag-prototype`.
Predecessor commits: NS4 prototype 88a739b, NS3.5 d65ac63.
Reference: `ns4_skeleton_bag_design_note.md`, `ns4_skeleton_bag_repro.md`,
`v5_alphaevolve_architecture.md`.

## 1. Which paths now route through SkeletonBag

When `use_skeleton_bag=True`, these origins emit via the bag (NS4.1 additions in **bold**):

| Origin                | Routed through bag in NS4 | Routed through bag in NS4.1 |
|-----------------------|---------------------------|-----------------------------|
| `priority_template`   | yes                       | yes                         |
| **`family_tactic`**   | no                        | **yes** (NS4.1 Stage 1)     |
| **`fallback_tactic`** | no                        | **yes** (NS4.1 Stage 2)     |
| **`tactic_template`** | no                        | **yes** (NS4.1 Stage 2)     |
| **`term_builder`**    | no                        | **yes** (NS4.1 Stage 3)     |
| `generative_topk`     | no — base policy          | no — base policy            |
| `retrieved_premise`   | no — needs live retriever | no — needs live retriever   |

The default code path (`use_skeleton_bag=False`) is unchanged. Default behavior is verified by re-running the NS4 parity test:

| Path                                      | proved | wall  | family_proved_counts             |
|-------------------------------------------|--------|-------|----------------------------------|
| Legacy (`use_skeleton_bag=False`)         | 37/38  | 166 s | identical                        |
| Skeleton-bag (`use_skeleton_bag=True`)    | 37/38  | 163 s | identical                        |

`scripts/ns4_compare_metrics.py` → **PARITY: OK**. Zero diffs across proved counts, origins, winning tactics, family_source attributions.

## 2. Skeleton attribution

New trace fields (populated only when `use_skeleton_bag=True`):

  - `winning_tactic_skeleton_name`        — e.g. `pt_iff_8`, `fam_mod_30`, `fb_36`
  - `winning_tactic_skeleton_shape`       — `iff`, `eq`, `lt`, `le`, `any`, …
  - `winning_tactic_skeleton_family`      — family substring or null
  - `winning_tactic_skeleton_specificity` — `0` (specific) or `1` (generic)
  - `winning_tactic_skeleton_priority`    — emit-band integer

New per-theorem result fields:

  - `skeleton_attempt_count`   — skeleton-sourced candidates run via Lean on this theorem
  - `skeleton_advanced_count` — those that produced a non-error transition (close OR step)
  - `skeletons_seen`          — ordered list of skeleton names attempted on this theorem

New `metrics.json` aggregates:

  - `skeleton_attempt_count`              (total across all theorems)
  - `skeleton_advanced_count`             (total)
  - `skeleton_proved_count`               (proofs whose winner was skeleton-sourced)
  - `skeleton_proved_counts`              — `{skeleton_name: proof_count}`
  - `skeleton_proved_counts_by_family`    — `{family: proof_count}`
  - `skeleton_proved_counts_by_shape`     — `{shape: proof_count}`
  - `skeleton_specificity_proved_counts`  — `{"0": specific_count, "1": generic_count}`
  - `skeleton_wins`                       — per-proof rows for reporting

## 3. Attribution table (nat_defs_medium, ns3-combined genome, bag path)

Aggregate counts on the 37 proved theorems:

| Bucket                          | Value                                                                 |
|---------------------------------|-----------------------------------------------------------------------|
| `skeleton_attempt_count`        | 183 — skeleton-sourced tactics actually run via Lean                  |
| `skeleton_advanced_count`       | 35 — produced a non-error transition (advance or close)              |
| `skeleton_proved_count`         | 34 — winning tactic was skeleton-sourced                              |
| (non-skeleton wins)             | 3 — all `generative_topk` (the model's own top-k beat the skeletons) |

Per-shape proof attribution:

| Shape   | Proofs |
|---------|--------|
| `iff`   | 24     |
| `any`   | 7      |
| `le`    | 1      |
| `eq`    | 1      |
| `lt`    | 1      |

Per-specificity:

| Specificity | Proofs |
|-------------|--------|
| 1 (generic) | 22     |
| 0 (specific)| 12     |

## 4. Top winning skeletons

| Skeleton    | Shape | Origin            | Family | Spec     | Wins | Template (truncated)                                                    |
|-------------|-------|-------------------|--------|----------|------|-------------------------------------------------------------------------|
| `pt_iff_8`  | iff   | priority_template | —      | generic  | 17   | `exact ⟨fun h => by omega, fun h => by omega⟩`                          |
| `fb_36`     | any   | fallback_tactic   | —      | generic  | 4    | `omega`                                                                 |
| `fam_mod_30`| any   | family_tactic     | mod    | specific | 2    | `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]`                              |
| `pt_le_20`  | le    | priority_template | —      | specific | 1    | `by_cases hc : c = 0 <;> [simp [hc]; exact (Nat.le_div_iff_mul_le' …)…]` |
| `pt_any_13` | any   | priority_template | —      | generic  | 1    | `split_ifs <;> omega`                                                   |
| `pt_iff_{1..6}` | iff | priority_template | —      | specific | 1 ea | various Nat-lemma-specific tactics (1 win each = 7 specific wins)       |
| `pt_eq_17`  | eq    | priority_template | —      | specific | 1    | `exact Nat.eq_one_of_mul_eq_one_right (by rwa [Nat.mul_comm])`          |
| `pt_lt_12`  | lt    | priority_template | —      | specific | 1    | `rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]`                     |

The `pt_iff_8` finding ("omega-omega exact pair carries 17 of 17 iff:generic wins") corroborates the NS4 mini-mutation experiment: dropping `constructor <;> intro h_split <;> simp_all` from the iff slot did not regress, because `pt_iff_8` was already the real winner.

## 5. Family/fallback routing — observed behaviour

Before NS4.1, families and fallbacks were emitted by inline blocks in
`StrategyWrapperPolicy.rank_tactics`. NS4.1 delegates them to
`SkeletonBag.emit_family_tactics` and `SkeletonBag.emit_fallback_tactics`
respectively. Two notable invariants preserved:

  - **Family declaration order is preserved.** `_match_families` is the
    only place family substring matching happens; the bag re-uses it.
  - **Specificity sort within a family is now applied.** This is a NEW
    invariant on the family path; previously only priority_templates had
    it (NS1). On the ns3-combined genome it is a no-op because each
    family's declared order already coincides with specificity-sorted
    order. If a future genome declares a generic family-tactic before
    a specific one, the bag path will reorder (specific first); the
    legacy path will not. This is an intended, harmless drift that the
    parity test confirmed doesn't fire on the current genome.

Fallback ordering: insertion order preserved verbatim. No rendering on
literal fallback strings (legacy behaviour).

Tactic-template ordering: insertion order preserved; `_render_template`
applied per Nat-var (legacy behaviour).

term_builder semantics: empty in ns3-combined so not exercised
end-to-end. The bag preserves the legacy **shape-XOR-any** semantics
(NOT NS3.5 shape-then-any), documented in
`SkeletonBag.emit_term_builder_tactics`. This is a deliberate
divergence from the priority path because changing it would alter
behaviour on existing term_builder-using genomes.

## 6. Scoring hook

`EvalMetrics` (`evolve/scoring.py`) gained three optional fields:

  - `skeleton_attempt_count`
  - `skeleton_advanced_count`
  - `skeleton_proved_count`

`score_metrics` is unchanged — the default scalar fitness does not
read these yet. They are surfaced so a future archive / scoring tweak
can rank candidates by skeleton diversity or per-skeleton win-rate
without re-parsing `metrics.json`.

`_parse_eval_metrics` in `evolve/evaluator.py` reads the three fields
from `metrics.json` and forwards them to the `EvalMetrics` instance.

## 7. Files changed in NS4.1

```
evolve/skeleton_bag.py        +4 emit methods (family/fallback/tactic_template/term_builder)
                              + family index (dict[family_name, list[Skeleton]])
evolve/strategy_wrapper.py    family / term_builder / generic blocks branch on use_skeleton_bag
                              parallel skeleton_* lists built per-call
                              priority block bag-init now passes all legacy fields
evolve/scoring.py             EvalMetrics: 3 new optional skeleton_* fields
evolve/evaluator.py           _parse_eval_metrics forwards skeleton_* fields
eval_rollout_all.py           _agg_by_key helper, per-step skeleton attempt/advanced counters,
                              per-theorem winning_tactic_skeleton_* fields,
                              metrics aggregates (skeleton_proved_counts_*, skeleton_wins)
```

No JSON schema changes. No genome rewrites. No retraining.

## 8. NS4.2 recommendation

The remaining slice for unification is the **retrieval** origin:

  - `emit_retrieved_tactics` would need the live retriever, the shape
    classifier output, and the per-call filter counts (filtered_self,
    filtered_unavailable, shape_mismatch_filtered).
  - Skeletons would be **synthesized per-state** rather than pre-built
    by `from_legacy_strategy_config`. This breaks the current model
    (skeletons are fixed at genome load time). NS4.2 should add a
    "dynamic skeleton" subclass — `RetrievedSkeleton` carrying the
    lemma name, the form template, and the lemma shape — and decide
    whether dynamic skeletons participate in the global bag or are
    a parallel stream.

After retrieval, NS4.3 should:

  1. Plumb skeleton attribution to `mutator.py` so it can emit
     skeleton-shaped edits (toggle enabled / change shape / move
     between bands) rather than string-list edits.
  2. Wire `skeleton_proved_counts` into `scoring.py` as a tie-breaker
     or diversity bonus.
  3. Start tagging skeletons with their origin author / cycle so the
     archive can carry attribution across generations.

Do not yet:
  - Remove legacy fields.
  - Reroute `generative_topk` — the base policy needs to stay live.
  - Mutate the JSON schema beyond `use_skeleton_bag`.
  - Retrain.

## 9. Recommendation for now

**Land the NS4.1 commit.** Default behavior is unchanged. Bag path
gains family / fallback / tactic_template / term_builder routing plus
full attribution and is fully introspectable from `metrics.json`. The
attribution table makes the "dead weight" pattern from NS4
(`disable_one` variant) visible at metric-level rather than requiring
a separate sweep: `skeleton_proved_counts` directly shows which
skeletons have zero wins and are candidates for removal.
