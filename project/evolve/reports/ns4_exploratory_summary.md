# NS4 — Exploratory Summary (4-hour Prototype)

Status: complete.
Date: 2026-05-22.
Branch: `ns4-skeleton-bag-prototype`.
Time budget: 4 hours.
Inputs: NS3 (lemma audit, 100c327), NS3.5 (any-fallback semantics, d65ac63).
Reference design: `ns4_skeleton_bag_design_note.md`.
Repro detail: `ns4_skeleton_bag_repro.md`.

## 1. What was implemented

  - `evolve/skeleton_bag.py` — new module with `Skeleton`, `SkeletonBag`,
    `EmittedTactic` dataclasses. `SkeletonBag.from_legacy_strategy_config()`
    converts the existing JSON config into Skeletons.
    `SkeletonBag.emit_priority_tactics()` renders the priority-template
    slot through the new path, preserving NS3.5 ordering exactly.
  - `evolve/strategy_wrapper.py` — adds `use_skeleton_bag` flag to
    `StrategyWrapperPolicy`, load/dump strategy config. When `True`,
    the priority-template emission block delegates to the bag; otherwise
    the legacy inline block runs verbatim.
  - `evolve/candidate.py` — adds `use_skeleton_bag` field (default
    `False`) to `SearchCandidate`.
  - `evolve/autonomous_research_loop.py` — adds the flag to
    `baseline_genome()` and forwards it through `write_strategy_config`.
  - `evolve/evaluator.py` — forwards `use_skeleton_bag` from
    `SearchCandidate` to `dump_strategy_config`.
  - `eval_rollout_all.py` — unpacks the new tuple element and passes
    it to `StrategyWrapperPolicy`.
  - `scripts/ns4_compare_metrics.py` — parity comparator.
  - `scripts/ns4_mini_mutator.py` — proof-of-concept skeleton edits.
  - Two new reports under `project/evolve/reports/`.

## 2. Did skeleton-bag reproduce 37/38?

**Yes.** Full bit-for-bit parity on `nat_defs_medium`:

| Path                                | proved | errored | runtime |
|-------------------------------------|--------|---------|---------|
| Legacy (`use_skeleton_bag=False`)   | 37/38  | 1       | 165 s   |
| Skeleton-bag (`use_skeleton_bag=True`) | 37/38  | 1       | 164 s   |

`scripts/ns4_compare_metrics.py` reports zero differences across:
proved counts, origins, winning tactics, family_source attributions.
Per-origin counts are identical. The single failure (`Nat.AM_GM`) is
the same environment-limited theorem flagged in
`ns3_lemma_audit_results.md`.

## 3. What part of the legacy wrapper was converted

Routed through `SkeletonBag.emit_priority_tactics`:

  - `priority_templates[shape]` and `priority_templates["any"]`,
    including NS1 specificity sort and the NS3.5 "shape first, then any
    as true fallback" semantics.

`SkeletonBag.from_legacy_strategy_config` also ingests the other
emit-path fields (`theorem_family_tactics`, `term_builder_templates`,
`fallback_tactics`, `tactic_templates`) into Skeleton instances for
introspection, but those origins are **not** emitted through the new
path in this prototype — they still flow through the legacy inline
blocks.

For the `ns3-combined` genome the adapter builds 48 skeletons:

```
priority_template: 21
family_tactic:     12
fallback_tactic:   12
tactic_template:   3
shapes seen:       any, eq, iff, le, lt
```

## 4. What remains on the old path

Origins still emitted by the legacy inline code (not by the bag):

  - `generative_topk`        — model rank_tactics output.
  - `family_tactic`          — `theorem_family_tactics`.
  - `retrieved_premise`      — premise retrieval.
  - `term_builder`           — `term_builder_templates`.
  - `fallback_tactic`        — `fallback_tactics`.
  - `tactic_template`        — `tactic_templates` (generic).
  - Per-theorem deny-list filter, per-state extras cap, retrieval
    bloat filter — all unchanged.

## 5. Regressions

**None on the default code path.** `use_skeleton_bag=False` is the
default; legacy callers are unaffected (verified by running the
legacy genome through the new code: 37/38 in 165 s with identical
proved_by_origin distribution).

## 6. Trace attribution

The bag-path preserves the legacy attribution surface verbatim:

  - `winning_tactic_origin` ∈ `priority_template`, …
  - `winning_tactic_family_source` carries `priority:<shape>:<specificity>`
    for priority entries (identical to legacy).
  - `winning_tactic_template_source` carries the raw template string.

New (not yet plumbed to the trace schema):

  - `Skeleton.name`           — stable handle (e.g. `pt_iff_3`) for
                                mutator references.
  - `wrapper.last_priority_emitted` — `list[EmittedTactic]` for the
                                most recent rank_tactics call (matches
                                the priority entries 1:1).

`skeleton_name` is **not** yet recorded on the per-theorem result row.
Doing so cleanly requires threading it through `rollout_one_theorem` —
straightforward but invasive enough to defer past this 4-hour budget.

## 7. Mini skeleton-mutation experiment

Ran three skeleton-level edits via `scripts/ns4_mini_mutator.py`,
all with `use_skeleton_bag=True`:

| Variant            | Edit                                                    | proved | Δ   | Note |
|--------------------|---------------------------------------------------------|--------|-----|------|
| `disable_one`      | Drop iff-slot generic `constructor <;> ... simp_all`    | 37/38  | 0   | The other iff-generic `exact ⟨fun h => by omega, ...⟩` was the real winner; the dropped one was dead weight on this set. |
| `duplicate_to_lt`  | Copy an iff-slot specific (Nat.*-mentioning) to lt slot | 37/38  | 0   | Cross-shape transfer is safe; no extra wins on this set (already saturated). |
| `reorder_iff`      | Remove all iff-slot specifics                            | 30/38  | −7  | Confirms the iff-specific block earns 7 proofs: `Nat.div_lt_one_iff`, `Nat.div_pos_iff`, `Nat.dvd_iff_div_mul_eq`, `Nat.mul_eq_left`, `Nat.mul_eq_right`, `Nat.pow_lt_pow_iff_left`, `Nat.sqrt_lt`. |

Findings:
  1. Skeleton-level edits are measurable end-to-end.
  2. `disable_one` is an actionable lead: a future NS4 sweep should
     try dropping more of the iff-generics one at a time to compress
     the slot without losing proofs.
  3. The cost of `reorder_iff` (−7 proofs) re-validates NS1's
     specificity-first invariant from a different angle.

## 8. Recommendation

**Proceed to incremental NS4 expansion, not a full refactor.** The
prototype demonstrates the architecture works end-to-end without
regression. The cheapest next slice is:

  1. **Route `theorem_family_tactics` through the bag.** It already
     has shape-like gating (family substring) and would benefit from
     NS1 specificity ordering, which is currently absent from the
     family code path. Expected effort: ~1 hr.
  2. **Plumb `skeleton_name` to the per-theorem result row.** This
     unlocks per-skeleton attribution in `scoring.py` and the
     followups generator. Expected effort: ~30 min.
  3. **Run a small skeleton-level mutation sweep** (5-10 edits) on
     `nat_defs_medium`. Use the `disable_one` lead as a starting
     point — there are likely more dead-weight skeletons.
  4. **Defer slot-vocabulary mutation** until the structural
     mutations have been validated end-to-end.

Do not yet:
  - Remove legacy fields.
  - Change the JSON schema.
  - Touch `mutator.py` or `scoring.py` until skeleton attribution is
    plumbed to results.
  - Retrain the model. The NS4 work is wrapper-only and does not
    require new traces.

## 9. Time spent (rough)

  - Stage 0–1 (read code, write design note):    ~25 min
  - Stage 2–3 (skeleton_bag.py + adapter):       ~25 min
  - Stage 4 (wire flag through wrapper / config): ~20 min
  - Stage 5 (parity eval, both paths):            ~30 min (incl. wait)
  - Stage 6 (repro report):                       ~10 min
  - Stage 7 (mini mutator + 3 variants):          ~25 min (incl. wait)
  - Stage 8 (this summary):                       ~10 min

Total: ~2:25, comfortably under the 4-hour budget.
