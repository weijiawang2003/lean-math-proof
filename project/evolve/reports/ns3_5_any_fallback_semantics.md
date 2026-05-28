# NS3.5 — `any` priority slot becomes a true fallback

Implements step NS3.5 surfaced during the NS3 audit: the wrapper's
priority-template shape gate used to be exclusive — once a genome
configured a slot for shape S, goals of shape S could never reach the
`any` slot, even when every shape-specific template failed. That forced
authors to manually mirror `any` templates into every configured shape
slot (see `_ns3_combined_mirrored` in `evolve/autonomous_research_ns3.py`
for the pre-NS3.5 workaround). NS3.5 makes `any` a true fallback,
emitted *after* the shape-specific slot's templates.

## Before / after emission semantics

### Before (pre-NS3.5)

```
shape_key = pt_goal_shape if pt_goal_shape in priority_templates else "any"
emit shape_key's templates (NS1-sorted by specificity)
# nothing else from priority_templates
```

For `Nat.add_mod_eq_ite` (shape = `le`) with a genome configuring `le`
templates, the `any` slot was unreachable even though it held the
multi-step template that closes the theorem.

### After (NS3.5)

```
slots_to_emit = []
if pt_goal_shape != "any" and pt_goal_shape in priority_templates:
    slots_to_emit.append((pt_goal_shape, ...))
if "any" in priority_templates:
    slots_to_emit.append(("any", ...))
for slot in slots_to_emit:
    emit slot's templates, NS1-sorted by specificity
```

Per-goal emission order is now:

1. shape-slot specifics (NS1-sorted)
2. shape-slot generics
3. `any`-slot specifics (NS1-sorted)
4. `any`-slot generics
5. generative_topk
6. family / retrieval / fallback / tactic_templates layers

Templates within each slot are deduplicated against the global `seen`
set, so a template that appears in both shape and `any` slots only
counts once (the shape-slot copy fires first, the `any`-slot copy is a
no-op).

`priority_template_budget` is shared across the slots — when set, it
caps the total emitted count after both slots have been consulted.

## Trace tag

`tactic_family_source` for priority entries already encoded the slot
label as `priority:{slot}:{spec}`. With NS3.5 it now naturally
distinguishes `priority:iff:specific` from `priority:any:specific`,
making "did the win come from the shape slot or the any slot?" a
direct readout from `family_proved_counts`.

## Genome cleanup

`_ns3_combined` lost ~8 lines of manual `eq`/`le` slot mirroring:

| slot     | pre-NS3.5 entries | post-NS3.5 entries |
|----------|--------------------|---------------------|
| iff      | 10                 | 10                  |
| lt       | 3                  | 3                   |
| any      | 4                  | 4                   |
| eq       | 6 (2 specific + 4 mirrored from any) | **2** |
| le       | 6 (2 specific + 4 mirrored from any) | **2** |
| budget   | 24                 | **18**              |

The leaner genome is exactly what the audit document suggested:
one patch per target failure, plus the v5-27 base. The historic
`_ns3_combined_mirrored` form is retained as a variant
(`ns3-combined-mirrored`) so the ablation can compare both forms
under the new wrapper.

## Eval on nat_defs_medium

| variant | proved | runtime | regressions | crashes | unknown const |
|---|---|---|---|---|---|
| **v5-27-w4-master (sanity)**          | **31 / 38** | 220 s | 0 | 0 | 0 |
| **`ns3-combined` (clean, no mirror)** | **37 / 38** | 157 s | 0 | 0 | 0 |
| **`ns3-combined-mirrored`**           | **37 / 38** | 156 s | 0 | 0 | 0 |

All three preserve their pre-NS3.5 scores. v5-27 needed no `any`
fallback (its iff/lt slots never relied on it for an unconfigured
shape), so its behaviour is byte-identical. Both forms of
`ns3-combined` hit 37/38 with the same six new wins
(`Nat.dvd_iff_div_mul_eq`, `Nat.eq_one_of_mul_eq_one_left`,
`Nat.add_mod_eq_ite`, `Nat.div_le_div_right`, `Nat.sqrt_lt`,
`Nat.pow_lt_pow_iff_left`).

### Trace breakdown confirms the fallback semantics

`family_proved_counts` for the clean ns3-combined:

```
priority:iff:specific:    7
priority:iff:generic:    17
priority:lt:specific:     1
priority:le:specific:     1
priority:eq:specific:     1
priority:any:generic:     1    ← the new fallback win
mod:                      2
```

The `priority:any:generic: 1` entry is `Nat.add_mod_eq_ite` closing
via `split_ifs <;> omega`. The goal is shape `le` (the if-then-else
contains a `≤`); the `le` slot has only specific templates (`gcongr`
and the `by_cases hc : c = 0 ...` form), neither of which closes
this goal. Under the old semantics the wrapper would stop there and
move on to the family layer. Under NS3.5 it continues into the `any`
slot, where the multi-step `cases k <;> [skip; rw [Nat.add_mod]; ...]`
advances the state, then `split_ifs <;> omega` closes — fired from
the `any` slot as a generic fallback.

The mirrored variant trace shows `priority:le:generic: 1` instead of
`priority:any:generic: 1` (the same theorem closes via the same
tactic, but is attributed to the `le` slot because the manual mirror
put a copy there). Same headline number, slightly less informative
trace — confirming the manual mirroring was a workaround, not a
diagnostic feature.

## Final genome path

`project/evolve/autonomous_runs/v5-ns3-20260522-222000-9beeab/eval/ns3-combined/genome.json`

(Replaces the pre-NS3.5 reference at
`v5-ns3-20260522-200519-d374c5/eval/ns3-combined/genome.json` — same
behaviour, leaner genome.)

## Recommendation for NS4

NS3.5 is the last cheap wrapper-only fix on the v5 architecture. The
remaining concerns (asymmetric branches, true skeleton-bag mutation,
LLM-driven candidate proposal) all need the `evolve/skeleton_bag.py`
refactor outlined in `v5_alphaevolve_architecture.md`. NS4 is the
correct next investment.
