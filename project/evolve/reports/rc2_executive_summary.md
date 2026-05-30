# RC2 Executive Summary

**RC2 = RC1 ⊕ SET_ITE_SIMP** (`simp [Set.ite]`), a single narrowly gated Set.ite
simplification action added non-destructively on top of the RC1 production wrapper.

## RC1 baseline (this benchmark, 8 surfaces / 176 theorems)
- Solved: **113 / 176** (authoritative `finished` key).
- Canonical floors: demo_v1 11/15, nat_defs_medium 37/38, nat_defs_large_v5 49/65.
- 0 regressions (baseline reference).

## RC2 result
- **Official credited improvement: +5** clean single-shot `SET_ITE_SIMP` wins over
  literal RC1: `Set.ite_empty_right`, `Set.ite_right`, `Set.ite_empty`,
  `Set.ite_empty_left`, `Set.ite_left`. Minimal-sufficient relabel: 5/5 TRUE,
  0 baseline-duplicate.
- RC2 benchmark solved: 131 / 176 (raw, includes the deferred depth-2 wins — see caveat).

## Safety
- **0 regressions** (RC2 is byte-identical to RC1 on every non-`Set.ite` theorem,
  by construction — the gate filters only wrapper-added entries; base output ungated).
- **0 off-gate emissions** (gate name-prefixed to `Set.ite`).
- Canonical floors preserved: demo_v1 11/15, nat_defs_medium 37/38, nat_defs_large_v5 49/65.
- Deterministic: hash-stable across 3 independent RC2 runs.

## Caveat (deferred to SX3, excluded from official delta)
The raw full-wrapper benchmark observed additional deterministic wins on 4 depth-2
theorems (`Set.ite_inter`, `Set.ite_inter_self`, `Set.ite_compl`,
`Set.ite_inter_compl_self`). Live forensics prove these close via the **sequence**
`simp [Set.ite] <;> aesop` (bare `aesop`/`simp_all` and single-shot `simp [Set.ite]`
all fail) — genuine **SX3 depth-2 sequence candidates**, NOT RC2 single-shot wins.
They are excluded from the official credited delta and deferred to SX3. The headline
RC2 improvement is **+5**, not the raw +18 surface-summed figure.

## Recommended production config
```
project/evolve/experiments/rc2_release/rc2_production_wrapper.json
```
Router unchanged: `project/evolve/routing/ns24_router.json`. RC1 remains preserved as
the previous baseline.
