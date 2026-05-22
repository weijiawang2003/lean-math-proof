# NS1 — per-slot specificity ordering for `priority_templates`

Implements step NS1 from `v5_next_steps.md`. The wrapper now stable-sorts
each shape slot of `priority_templates` so that **specific** templates
(those naming a dotted Mathlib lemma or using a typed hypothesis
placeholder) emit before any **generic** template (`omega`, `simp_all`,
`exact ⟨fun h => by omega, fun h => by omega⟩`, `split_ifs <;> omega`,
…). The change closes the wave-5 v5-31 regression, where putting an
omega-omega template at the top of the iff slot shadowed every specific
template downstream because of the wrapper's "first-non-erroring-wins"
semantics within a slot.

## What changed

### New helper — `classify_template_specificity`

In `evolve/strategy_wrapper.py`:

```python
def classify_template_specificity(template: str) -> tuple[int, str]:
    """Returns (rank, label). rank=0 means specific, rank=1 means generic."""
    if _PRIORITY_HYP_PLACEHOLDER_RE.search(template):   # {hyp_pos}, {hyp_ne_zero}, …
        return (0, "specific")
    if _PRIORITY_MATHLIB_LEMMA_RE.search(template):     # Nat.foo, Nat.foo_bar, Mathlib.X.y
        return (0, "specific")
    return (1, "generic")
```

The two regexes:

- `_PRIORITY_HYP_PLACEHOLDER_RE = re.compile(r"\{hyp_[a-zA-Z_][A-Za-z0-9_]*\}")`
- `_PRIORITY_MATHLIB_LEMMA_RE = re.compile(r"[A-Z][A-Za-z0-9_']*(?:\.[a-zA-Z_][A-Za-z0-9_']*)+")`

The lemma-name regex requires the leading identifier to start with a capital
(matches `Nat`, `Mathlib`, `List`, etc.) and to contain at least one `.`-
separated lowercase tail. This catches every Mathlib-style namespace path
without false-positiving on tactic names like `omega`, `simp_all`,
`split_ifs`, `constructor`, etc.

### Sort step in `rank_tactics`

Before iterating `self.priority_templates[pt_shape_key]`, the wrapper now
stable-sorts the slot's template list by the classifier's `rank` field.
Templates within each class keep their declared order — so authors still
control specific-vs-specific ordering, and the wrapper only fixes the
specific-vs-generic axis.

### Trace tag

`family_source` for priority templates changed from `"priority:{shape}"`
to `"priority:{shape}:{spec_label}"` (e.g. `priority:iff:specific`,
`priority:any:generic`). All downstream consumers treat `family_source`
opaquely (it's used as a dict key for `family_proved_counts`), so the
finer-grained labels are pure diagnostic improvement — no parsing breakage.

## Classifier sanity check

Running the classifier over the v5-27 iff slot (9 entries) keeps the
existing declared order unchanged: the seven specific templates rank=0,
the two generics (`exact ⟨fun h => by omega, fun h => by omega⟩` and
`constructor <;> intro h_split <;> simp_all`) rank=1. Stable sort is a
no-op there — the genome already has correct specificity ordering.

Running the classifier over the v5-31 iff slot reorders the 7 entries
from "generic, generic, specific, specific, specific, specific, specific"
to "specific, specific, specific, specific, specific, generic, generic",
which is exactly what v5-27's iff slot looks like.

## Eval results

| variant                                | proved | rate | runtime | new wins (vs v4.7 26/38) |
|---|---|---|---|---|
| v5-27-w4-master (pre-NS1, scoreboard) | 31/38  | 82% | 214 s   | div_lt_one_iff, div_pos, div_pos_iff, mul_eq_left, mul_eq_right |
| **v5-27-w4-master + NS1**             | **31/38** | **82%** | **217 s** | **all 5 (unchanged)** |
| v5-31-w5-iff-reorder (pre-NS1)        | 27/38  | 71% | 240 s   | div_pos only (regressed by 4) |
| **v5-31-w5-iff-reorder + NS1**        | **31/38** | **82%** | **211 s** | **all 5 (recovered)** |

Both post-NS1 runs prove the exact same 31-theorem set with the same
`proved_by_origin` breakdown (`tactic_template: 22, family_tactic: 2,
generative_topk: 3, fallback_tactic: 4`). The auto-sort rescues v5-31
to v5-27's score, confirming that what was previously interpreted as a
4-theorem genome regression was in fact a wrapper ordering bug. The
trace tag breaks out as:

```
priority:iff:specific  → 4 wins (div_lt_one_iff, mul_eq_left, mul_eq_right + 1 of div_pos/div_pos_iff)
priority:iff:generic   → 17 wins (omega-omega closes most pre-existing iff baseline wins)
priority:lt:specific   → 1 win
```

Genome paths:
  - `project/evolve/autonomous_runs/ns1_v5_27_repro/eval-*/eval-*/metrics.json`
  - `project/evolve/autonomous_runs/ns1_v5_31_iff_reorder_fixed/eval-*/eval-*/metrics.json`

`nat_defs_large_v5` was not re-evaluated tonight — v5-27's iff slot is
already correctly ordered, so the sort is a no-op there and the prior
43/64 result remains the authoritative number (re-run cheaply with
`python -m evolve.run_large_v5 --best-genome <path> --theorem-set nat_defs_large_v5 --out-dir <out>`
when needed).

## Why it works

The wrapper's `rank_tactics` returns a single ranked tactic list per
state, and the eval loop tries them in order until one is accepted by
Lean. Once any tactic *advances* the state (without producing an error),
later tactics in the same list are never tried for that step — they
would all start from the new state, not the original one. The wrapper
calls this "first-non-erroring-wins" semantics.

The same semantics holds within a slot: when the iff slot is `[generic1,
generic2, specific1, …]`, `generic1` may itself "succeed" (no Lean
error) without closing the goal — it just rewrites the state into a
shape from which `specific1` no longer unifies. The four wins lost in
wave 5 followed this exact pattern.

Stable-sorting the slot by specificity guarantees that whenever a
specific template would otherwise have unified, it gets a chance to run
*first*. Generics still run if every specific fails, so we don't lose
the omega/simp_all safety net.

## Risk

Low. The change is wrapper-only; no genome schema changed; the v5-27
genome's behaviour is preserved (its iff slot is already correctly
ordered). The classifier is conservative — anything we're unsure about
falls into "generic" and runs in its declared position relative to other
generics. The trace-label change is additive (existing aggregations
either work unchanged or get a finer-grained breakdown for free).

## Files touched

  - `evolve/strategy_wrapper.py` — `+39` LoC for the classifier and
    `+5 / -2` LoC for the sort step + tag change in `rank_tactics`.

## Next

Whether to commit depends on the eval outcomes. If v5-27 is preserved
and v5-31 recovers, this becomes the new wrapper baseline and any
future hand-curated priority_templates can rely on the auto-sort to
guard against ordering regressions. The next step on the v5-next-steps
list is **NS3** — lemma-name retrieval pass for unsolved theorems
(targeting `Nat.div_le_div_right`, `Nat.dvd_iff_div_mul_eq`,
`Nat.pow_lt_pow_iff_left`).
