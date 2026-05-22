# NS4 — Skeleton-Bag Reproduction Report

Status: 4-hour exploratory prototype.
Date: 2026-05-22.
Branch: `ns4-skeleton-bag-prototype`.
Predecessors: NS3 (lemma audit, 100c327), NS3.5 (any-fallback semantics, d65ac63).
Reference design: `ns4_skeleton_bag_design_note.md`.

## Summary

The NS4 prototype routes `priority_templates` emission through a new
`evolve.skeleton_bag.SkeletonBag` object **with bit-for-bit parity**
against the legacy NS3.5 wrapper code path on `nat_defs_medium`.

| Metric                      | Legacy (`use_skeleton_bag=False`) | Skeleton-bag (`use_skeleton_bag=True`) |
|-----------------------------|-----------------------------------|----------------------------------------|
| proved                      | 37 / 38                           | 37 / 38                                |
| errored                     | 1 (`Nat.AM_GM`)                   | 1 (`Nat.AM_GM`)                        |
| wall clock                  | 165 s                             | 164 s                                  |
| proved_by_origin            | `{tactic_template: 28, family_tactic: 2, generative_topk: 3, fallback_tactic: 4}` | identical |
| family_proved_counts        | `{priority:iff:generic: 17, mod: 2, priority:le:specific: 1, priority:iff:specific: 7, priority:any:generic: 1, priority:eq:specific: 1, priority:lt:specific: 1}` | identical |
| theorems differing          | 0                                 | —                                      |
| winning_tactic differences  | 0                                 | —                                      |
| family_source differences   | 0                                 | —                                      |

Cross-checked theorem-by-theorem via `scripts/ns4_compare_metrics.py`:

```
PARITY: OK
```

The remaining failure (`Nat.AM_GM`) is the same environment-limited theorem
already flagged in `ns3_lemma_audit_results.md`. Not template-fixable.

## What is converted

`SkeletonBag.from_legacy_strategy_config(cfg)` ingests the strategy
config and produces one `Skeleton` per entry in:

| Legacy field                  | Skeleton.origin     | shape         | priority band | family            |
|-------------------------------|---------------------|---------------|---------------|-------------------|
| `priority_templates[shape]`   | `priority_template` | `shape`       | 0             | None              |
| `theorem_family_tactics[fam]` | `family_tactic`     | `any`         | 10            | `fam` (substring) |
| `term_builder_templates[s]`   | `term_builder`      | `s`           | 15            | None              |
| `fallback_tactics`            | `fallback_tactic`   | `any`         | 20            | None              |
| `tactic_templates`            | `tactic_template`   | `any`         | 25            | None              |

Specificity is computed by re-using `strategy_wrapper.classify_template_specificity`.

For the `ns3-combined` genome on `nat_defs_medium` the adapter produces:

```
total skeletons: 48
by origin: {priority_template: 21, family_tactic: 12, fallback_tactic: 12, tactic_template: 3}
shapes seen: ['any', 'eq', 'iff', 'le', 'lt']
```

## What is routed through the new path

**Only `priority_template` skeletons are emitted via `SkeletonBag.emit_priority_tactics`.**
The output of that method is appended to `priority_entries` with the same
tuple shape (`(tactic, origin, template_source, family_source, ...)`) the
legacy block produced, so downstream eval code (origin/family_source-based
counters, trace fields) is unaffected.

The remaining origins (`family_tactic`, `term_builder`, `fallback_tactic`,
`tactic_template`, `retrieved_premise`) still run through the legacy
inline code path in `evolve/strategy_wrapper.py`. They are present in the
bag for introspection / future migration, not for emission.

## What is preserved

NS3.5 ordering semantics are preserved exactly:

  1. shape-specific priority templates (NS1-sorted)
  2. shape-generic priority templates
  3. `any`-specific priority templates (NS1-sorted)
  4. `any`-generic priority templates
  5. generative top-k (model)
  6. family / retrieval / term_builder / fallback / tactic_template (legacy path)

In-process verification (no Lean roundtrip) confirms identical tactic
sequences on four representative states (iff with hyp_pos, eq, ite/any-fallback,
lt with hyp_pos). The end-to-end Lean eval confirms identical winning tactics
and identical family_source attributions on every shared proof.

## Trace attribution

`winning_tactic_family_source` already carries the shape + specificity for
priority-template entries (e.g. `priority:iff:specific`). Identical on
legacy and bag paths.

A new `skeleton_name` handle (`pt_<shape>_<idx>`) is computed by the bag,
which gives a stable index for mutator references. This was **not**
plumbed through to `result["winning_tactic_skeleton_name"]` in this
prototype to avoid touching the trace schema — that is the natural next
step when NS4 expands beyond the prototype.

Per-call diagnostics are exposed on the wrapper:

  - `wrapper._skeleton_bag` — the bag (lazily built on first rank call)
  - `wrapper.last_priority_emitted` — `list[EmittedTactic]` for the most
    recent call (parallel to the priority entries in `last_ranked_tactics`)
  - `wrapper.last_priority_template_attempt_count` — unchanged from NS3.5

## Known gaps / non-goals

  - Slot-vocabulary mutation is not implemented (out of scope for the
    4-hour prototype; see `v5_alphaevolve_architecture.md` Direction F).
  - Family / retrieval / term_builder / generic fallback emission still
    lives in the legacy inline block.
  - `skeleton_name` is not yet plumbed to the per-theorem result row.
  - `scoring.py` does not yet read skeleton attribution.
  - `mutator.py` does not yet produce skeleton-shaped edits.

## Files changed

```
evolve/skeleton_bag.py                       (NEW)
evolve/strategy_wrapper.py                   priority block + use_skeleton_bag plumbing
evolve/candidate.py                          use_skeleton_bag field
evolve/autonomous_research_loop.py           baseline_genome + write_strategy_config forward flag
eval_rollout_all.py                          unpack new tuple element + forward
scripts/ns4_compare_metrics.py               (NEW) parity comparator
project/evolve/reports/ns4_skeleton_bag_design_note.md  (NEW)
project/evolve/reports/ns4_skeleton_bag_repro.md        (NEW, this file)
```

## Recommendation

Parity is solid enough to land the prototype behind the
`use_skeleton_bag` flag (default `False`). The natural next slice is
routing `theorem_family_tactics` through the same bag, because it has
the most overlap with priority_templates (both shape-able, both want
NS1 specificity ordering). After that, term_builder is the next
candidate.
