# NS4 — Skeleton-Bag Prototype (Design Note)

Status: 4-hour exploratory prototype.
Predecessors: NS3 (lemma audit, 100c327), NS3.5 (any-fallback semantics, d65ac63).
Baseline: ns3-combined clean genome proves 37/38 on `nat_defs_medium`. The
remaining failure is `Nat.AM_GM` (environment/import limited, not template-fixable).
Long-arc reference: `v5_alphaevolve_architecture.md` ("Direction F").

## 1. Why flat fields are now the bottleneck

The genome carries five overlapping tactic-emit fields, each with its own
shape-gate, ordering rule and slot-substitution path:

  - `priority_templates` — shape-keyed (or `any`), emitted before the model.
    NS1 stable-sorts by specificity. NS3.5 emits shape-specifics, then
    shape-generics, then any-specifics, then any-generics.
  - `tactic_templates` — flat list, no shape gate, emitted after the
    generative top-k as part of generic fallbacks.
  - `theorem_family_tactics` — substring-matched against the theorem name,
    emitted between base and generic.
  - `term_builder_templates` — shape-keyed (or `any`), emitted after
    retrieval, before generic fallbacks.
  - `fallback_tactics` — flat list, emitted last.

Plus the retrieval block (`retrieval_*`) which synthesizes per-lemma
tactics from a separate code path.

Problems that already cost us time in v5 / NS1–NS3.5:

  1. **Ordering rules drift between blocks.** NS1 added specificity
     sorting only to `priority_templates`. The same idea would help
     `theorem_family_tactics` but lives in a different code path.
  2. **`any` semantics differ block-to-block.** NS3.5 fixed
     priority-templates to emit shape THEN `any` as a true fallback;
     `term_builder` still uses the older "exact shape OR any" rule.
  3. **Slot/placeholder rendering is shared but origin tagging is not.**
     `_render_template` is reused; the wrapper still has five copies
     of "for-shape, for-template, dedup-against-seen, append-with-origin".
  4. **Mutation has nowhere to attach.** Today the mutator can swap a
     string inside a list. It cannot move a "good template" from
     `theorem_family_tactics` into `priority_templates` because they
     are not the same object. The genome shape forbids the structural
     edit AlphaEvolve-style evolution requires.

## 2. What skeleton-bag should represent

One first-class object per emission unit, with five orthogonal fields:

  - `shape` — `"iff"|"eq"|"lt"|"le"|"dvd"|"and"|"or"|"unknown"|"any"`.
    The current goal-shape gate, lifted out of per-block dicts.
  - `family` — optional substring (matched against theorem full_name).
    Subsumes `theorem_family_tactics`. `None` means "no family gate".
  - `priority` — integer. Lower = earlier. Within a shape slot, ties
    break by `specificity` (specific < generic), matching NS3.5.
  - `template` — the tactic string with `{var}`/`{hyp_*}` placeholders.
    Renders via the existing `_render_template`.
  - `origin` — the legacy origin tag ("priority_template",
    "fallback_tactic", "family_tactic", "term_builder",
    "tactic_template") so traces keep their attribution surface.

Plus a `name` (for trace attribution + mutator handles) and an
`enabled` flag (for cheap toggle mutations).

A `SkeletonBag` is `dict[shape_or_"any", list[Skeleton]]` plus an
`emit_tactics()` method that reproduces today's wrapper ordering:

```
for skel in bag.for_state(goal_shape, active_families):
    for rendered in render(skel.template, nat_vars, hypotheses):
        yield EmittedTactic(...)
```

Slot-vocabulary mutation (the Direction-F inner tier) is **out of
scope** for this 4-hour prototype.

## 3. Backward-compatibility plan

  - No old fields are removed.
  - No JSON schema changes that break older population files.
  - A new `use_skeleton_bag: bool = False` flag selects the new path.
    Default `False` → the old wrapper code runs verbatim.
  - The skeleton-bag path **only re-routes `priority_templates`** in
    this prototype. Everything else (family, retrieval, term_builder,
    generic fallback) keeps using the old code paths. This is the
    cheapest slice that demonstrates the architecture and is the
    block where NS1–NS3.5 has the most ordering machinery to validate.
  - The adapter `SkeletonBag.from_legacy_candidate(...)` converts the
    flat fields into Skeletons, so the genome JSON is unchanged.
  - The trace fields stay identical (`origin`, `template_source`,
    `family_source` are all derived from the skeleton). New
    `skeleton_name` / `skeleton_priority` / `skeleton_specificity`
    fields are added only when the bag is in use.

## 4. What this prototype implements

Stage 2 — `evolve/skeleton_bag.py`:
  - `Skeleton`, `SkeletonBag`, `EmittedTactic` dataclasses.
  - `SkeletonBag.add()`, `for_state()`, `emit_tactics()`.
  - `classify_template_specificity` re-used from `strategy_wrapper`.

Stage 3 — `SkeletonBag.from_legacy_strategy_config(cfg_dict)`:
  - `priority_templates[shape][i]` → Skeleton(origin="priority_template",
    shape=shape, priority=0).
  - `fallback_tactics[i]` → Skeleton(origin="fallback_tactic",
    shape="any", priority=20).
  - `theorem_family_tactics[fam][i]` → Skeleton(origin="family_tactic",
    family=fam, shape="any", priority=10).
  - `term_builder_templates[shape][i]` → Skeleton(origin="term_builder",
    shape=shape, priority=15).
  - `tactic_templates[i]` → Skeleton(origin="tactic_template",
    shape="any", priority=25).

For this prototype, only the priority_template-origin Skeletons are
actually emitted through the new path. The others are present in the
bag for introspection but the wrapper still emits them via the legacy
path. This avoids drift while we verify parity on the cheap slice.

Stage 4 — config plumbing:
  - `use_skeleton_bag` added to `SearchCandidate` and JSON load/dump.
  - `StrategyWrapperPolicy.__init__` accepts `use_skeleton_bag`.
  - When `True`, the priority-template emit block delegates to
    `SkeletonBag.emit_priority_tactics(...)`. The block's ordering
    semantics must match NS3.5: shape-specific → shape-generic →
    any-specific → any-generic.

Stage 5 — parity test:
  - Old path: `ns3-combined` genome → 37/38.
  - New path: same genome with `use_skeleton_bag=True` → expect 37/38.

## 5. Explicit non-goals (out of scope for 4 hours)

  - Slot-vocabulary mutation.
  - Skeleton-attribution bonus in `scoring.py`.
  - Replacing the family / retrieval / term_builder / fallback emit
    paths.
  - Removing legacy fields.
  - Changes to `mutator.py`.
  - Changes to `eval_rollout_all.py` trace schema beyond optional
    `skeleton_name` field.
  - Training data refresh / model retraining.

## 6. Rollback plan

If the new path fails parity:
  - `use_skeleton_bag` defaults to `False`; nothing else changes.
  - Old wrapper code is untouched.
  - Branch `ns4-skeleton-bag-prototype` is left in place for the next
    iteration; mainline behavior is unchanged.
