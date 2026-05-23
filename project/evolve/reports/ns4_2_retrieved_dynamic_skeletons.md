# NS4.2 — Retrieved Premises as Dynamic Skeletons

Status: complete.
Date: 2026-05-22.
Branch: `ns4-skeleton-bag-prototype`.
Predecessor commits: NS4.1 28a3b0c, NS4 88a739b.
Reference: `ns4_1_skeleton_unification.md`, `v5_alphaevolve_architecture.md`.

## 1. What changed

The retrieved-premise emission path is now routed through
`evolve.skeleton_bag.SkeletonBag.emit_retrieved_tactics` when
`use_skeleton_bag=True`. Unlike the other origins, retrieved entries
are NOT pre-registered as static Skeletons by the adapter — they are
synthesized per-state as `EmittedTactic` instances with dynamic
metadata, because each retrieved tactic depends on:

  - the current `state_pp` (goal shape classification)
  - the theorem `full_name` (self-filter)
  - the active family (catalog bucket)
  - the retrieved lemma set (per-call)
  - the configured form templates

`EmittedTactic` gained three optional fields to carry retrieved
metadata without adding a class hierarchy:

```python
@dataclass
class EmittedTactic:
    ...                           # existing skeleton fields
    retrieved_premise: str | None = None
    retrieved_form: str | None = None
    retrieved_shape: str | None = None
```

Each retrieved tactic emits with:

  - `skeleton_name = f"retrieved:{lemma}:{form_label}"` (e.g. `retrieved:Nat.div_lt_iff_lt_mul:rw`)
  - `shape = goal_shape` (e.g. `iff`)
  - `family = activated_family` (e.g. `div`)
  - `specificity = 0` (specific — retrieved lemmas are by construction targeted)
  - `priority = 12` (`PRIORITY_RETRIEVED`, between family=10 and term_builder=15)
  - `template_source = form_template` (e.g. `"rw [{p}]"`)
  - `family_source = activated_family` (matches legacy attribution)

The legacy `last_retrieved_*` attribute surface on the wrapper is
preserved unchanged: `last_retrieval_activation`,
`last_retrieved_lemma_set`, `last_retrieval_filtered_self_count`,
`last_retrieval_filtered_unavailable_count`,
`last_shape_mismatch_filtered_count`. The diagnostics dict returned
by the bag method carries these values directly back to the wrapper.

## 2. Parity results

### nat_defs_medium (38 theorems)

| Path                                | proved | wall  | errored | family_proved_counts |
|-------------------------------------|--------|-------|---------|----------------------|
| Legacy (`use_skeleton_bag=False`)   | 37/38  | 165 s | 1       | identical            |
| Skeleton-bag (`use_skeleton_bag=True`) | 37/38  | 163 s | 1       | identical            |

`scripts/ns4_compare_metrics.py` → **PARITY: OK** (zero diffs on proved counts, origins, winning tactics, family_source).

### nat_defs_large_v5 (64 available, 65 in set)

| Path                                | proved | wall  | errored | exhausted |
|-------------------------------------|--------|-------|---------|-----------|
| Legacy                              | 49/64  | 335 s | 14      | 1         |
| Skeleton-bag                        | 49/64  | 336 s | 14      | 1         |

`scripts/ns4_compare_metrics.py` → **PARITY: OK**. Origin counts identical: `{tactic_template: 33, family_tactic: 2, generative_topk: 4, fallback_tactic: 10}`.

## 3. Retrieval diagnostic parity

Every per-call diagnostic counter matches bit-for-bit between paths.

**nat_defs_medium:**

| Diagnostic                                | Legacy | Bag |
|-------------------------------------------|--------|-----|
| retrieved_premise_activation_count        | 6      | 6   |
| retrieved_premise_attempt_count           | 1      | 1   |
| retrieved_premise_advanced_count          | 1      | 1   |
| retrieved_premise_proved_count            | 0      | 0   |
| retrieved_premise_filtered_self_count     | 7      | 7   |
| retrieved_premise_filtered_unavailable_count | 44   | 44  |
| shape_mismatch_filtered_count             | 40     | 40  |
| retrieved_premise_form_counts             | `{rw:1}` | `{rw:1}` |
| retrieved_shape_counts                    | `{iff:1}` | `{iff:1}` |

**nat_defs_large_v5:**

| Diagnostic                                | Legacy | Bag |
|-------------------------------------------|--------|-----|
| retrieved_premise_activation_count        | 10     | 10  |
| retrieved_premise_attempt_count           | 90     | 90  |
| retrieved_premise_advanced_count          | 3      | 3   |
| retrieved_premise_proved_count            | 0      | 0   |
| retrieved_premise_filtered_self_count     | 7      | 7   |
| retrieved_premise_filtered_unavailable_count | 107  | 107 |
| shape_mismatch_filtered_count             | 76     | 76  |

## 4. Skeleton metrics with retrieval included

| Set                | Path | skeleton_attempt | skeleton_advanced | skeleton_proved |
|--------------------|------|------------------|-------------------|-----------------|
| nat_defs_medium    | legacy | 0              | 0                 | 0               |
| nat_defs_medium    | bag    | **184** (=183 NS4.1 + 1 retrieved) | **36** (=35 + 1) | 34 |
| nat_defs_large_v5  | legacy | 0              | 0                 | 0               |
| nat_defs_large_v5  | bag    | **792**        | **64**            | 45              |

On nat_defs_medium the retrieval contribution to skeleton counters
is exactly `+1` attempt and `+1` advance (matching the legacy
`retrieved_premise_attempt_count=1`, `retrieved_premise_advanced_count=1`).
The single attempted retrieved tactic was:

```
theorem Nat.div_lt_iff_lt_mul'
  skeletons_seen: [..., "retrieved:Nat.div_lt_iff_lt_mul:rw"]
  retr_attempt=1  retr_advance=1
```

`skeleton_proved_count` is 0 for retrieved entries because retrieval
doesn't close any theorem on this genome — but the `skeletons_seen`
field now contains entries like `retrieved:Nat.div_lt_iff_lt_mul:rw`
so the mutator / archive (NS4.3+) can attribute skeleton-level cost
to retrieval entries even when they don't win.

On nat_defs_large_v5 retrieval had 90 attempts and 3 advances —
all reflected in `skeleton_attempt_count` (90 of 792) and
`skeleton_advanced_count` (3 of 64). Still 0 retrieved wins, so
`skeleton_proved_counts` carries no `retrieved:*` entries.

## 5. Trace attribution

Per-theorem result row gains nothing new (the `winning_tactic_skeleton_*`
fields from NS4.1 already cover retrieved wins automatically because
the parallel `last_skeleton_names` list is built via the unified
`emitted_lookup`). Specifically: a retrieved tactic that wins a future
theorem will populate:

  - `winning_tactic_origin = "retrieved_premise"`
  - `winning_tactic_retrieved_premise = "<lemma>"` (legacy field)
  - `winning_tactic_retrieved_form = "<form_label>"` (legacy field)
  - `winning_tactic_skeleton_name = "retrieved:<lemma>:<form_label>"` (NEW)
  - `winning_tactic_skeleton_shape = "<goal_shape>"` (NEW)
  - `winning_tactic_skeleton_family = "<activated_family>"` (NEW)
  - `winning_tactic_skeleton_specificity = 0` (NEW)
  - `winning_tactic_skeleton_priority = 12` (NEW)

## 6. What remains outside the bag

After NS4.2, only one emit path is still legacy: **`generative_topk`**
— the base policy's beam-search output. This stays legacy by design:

  - The base policy is a live PyTorch model. It does not produce
    pre-buildable skeletons; each call invokes the network.
  - Trying to wrap each generative output as a Skeleton would
    obscure the "the model said this" attribution that the rest of
    the eval pipeline relies on for the proved_by_origin breakdown.

Everything else (`priority_template`, `family_tactic`,
`fallback_tactic`, `tactic_template`, `term_builder`,
`retrieved_premise`) now flows through `SkeletonBag` when
`use_skeleton_bag=True`.

## 7. Files changed in NS4.2

```
evolve/skeleton_bag.py       +emit_retrieved_tactics method
                             +PRIORITY_RETRIEVED constant
                             EmittedTactic: retrieved_premise/form/shape fields
evolve/strategy_wrapper.py   retrieval block branches on use_skeleton_bag
                             last_retrieved_emitted exposed
                             emitted_lookup includes retrieved entries
project/evolve/reports/ns4_2_retrieved_dynamic_skeletons.md  (new)
```

No changes to:
  - `eval_rollout_all.py` (existing `skel_name`-based counters absorb
    retrieved attempts/advances automatically)
  - `evolve/scoring.py` / `evolve/evaluator.py` (no new aggregates)
  - `evolve/candidate.py` (no new genome fields)
  - JSON schema / genome format

## 8. NS4.3 / NS5 recommendation

The skeleton-bag is now the unified representation for every
non-generative emit path. The next slice should focus on **mutation
and archive**, not on routing more origins:

  1. **Plumb skeleton attribution into `mutator.py`.** Today the
     mutator can only edit string lists. With NS4.2 in place, every
     emit is a Skeleton — the mutator should produce skeleton-shaped
     edits: `toggle enabled`, `change shape`, `change priority band`,
     `move between bands`, `change family gate`. Dynamic (retrieved)
     skeletons can be edited by mutating the form list or retrieval
     parameters rather than the bag directly.
  2. **Add a skeleton archive.** Store `(skeleton_name, win_count,
     advance_count, theorems_won)` tuples across generations. Lets
     the loop find skeletons that are *unique* winners (won theorems
     no other genome closed) versus *common* winners.
  3. **Two-tier mutation.** Once skeletons are first-class, implement
     the outer-tier (structural skeleton edits) and inner-tier (slot
     vocabulary, eventually) split from
     `v5_alphaevolve_architecture.md`.
  4. **Wire `skeleton_proved_counts` into `scoring.py`** as a
     diversity bonus or tie-breaker, only after the mutator can act
     on it.

Do not yet:
  - Remove legacy fields.
  - Reroute `generative_topk` — base policy stays live.
  - Mutate the JSON schema beyond what NS4 already added.
  - Retrain.
