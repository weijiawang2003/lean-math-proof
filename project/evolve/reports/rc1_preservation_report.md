# RC1 preservation report

The RC1 production wrapper (`project/evolve/experiments/rc1/rc1_production_wrapper.json`)
composes three proven, deterministic components and **changes nothing about the
production baseline outside its two namespace gates**.

## Configuration integrity

- **NS9 best genome unchanged.** RC1 is a *separate composed config*; it
  deep-copies the NS9 genome and never writes `project/evolve/best/ns9_best_genome.json`
  (verified: the on-disk genome has no `theorem_name_tactic_gates` / symbolic
  blocks).
- **NS24 router unchanged.** RC1 runs against the existing
  `project/evolve/routing/ns24_router.json`; no route or checkpoint is touched.
- **AX4 learned predictor: disabled** (`symbolic_predictor.enabled = false`).
- **SX1 sequence search: disabled** (`symbolic_sequence_search.enabled = false`).
- **Broad Set aesop and MX1 Set/Finset ext/cases actions: excluded** entirely.

## Namespace gates (static check)

`project/data/rc1_preservation_meta.json` (gate-logic over the standard
theorem-set names — computed with the real `gate_allows` / name-gate functions,
not assumed):

| surface | n | Multiset-symbolic admissible | aesop admissible | off-gate |
|---|---|---|---|---|
| demo_v1 | 15 | 0 | 0 | 0 |
| nat_defs_medium | 38 | 0 | 0 | 0 |
| nat_defs_large_v5 | 65 | 0 | 0 | 0 |
| ns14_set_finset_extra | 20 | 0 | 0 | 0 |
| ns17_set_extra | 30 | 0 | 0 | 0 |
| ns17_finset_extra | 30 | 0 | 0 | 0 |
| wx2_list_cases_easy | 40 | 0 | 0 | 0 |
| wx2_list_cases_medium | 35 | 0 | 0 | 0 |
| cx2_int_iff_omega_easy | 12 | 0 | 0 | 0 |
| ax4_multiset_induction_heldout | 45 | **45** | 0 | 0 |
| mx2_set_finite_frontier | 10 | 0 | **10** | 0 |

- The **Multiset induction action** is admissible **only** on `Multiset.*`
  theorems (45/45 on the Multiset set, 0 on every other surface).
- An **`aesop`** tactic is admissible **only** on `Set.Finite.`/`Set.toFinset`
  theorems (10/10 on the Set.Finite frontier, 0 elsewhere — including
  `ns17_set_extra`, whose names are not in the gated family).
- **Total off-gate emissions = 0** for both components, across Nat, Int, demo,
  Set (non-Finite), Finset, List, Option.

## Regressions = 0

All RC1 additions are **additive to the NS9 ranked list** (appended, never
reordering or replacing base/generative entries) and **namespace-gated**, so no
production win can be lost. Confirmed empirically by the RC1 benchmark
(`project/data/rc1_full_benchmark_meta.json`): on every floor/control surface
RC1 equals NS9 with **0 regressions** —

| floor / control | NS9 | RC1 | Δ | regr |
|---|---|---|---|---|
| demo_v1 | 11 | 11 | 0 | 0 |
| nat_defs_medium | 37 | 37 | 0 | 0 |
| ns17_set_extra | 18 | 18 | 0 | 0 |
| ns17_finset_extra | 15 | 15 | 0 | 0 |

— and by the WX3 (Multiset) and MX2 (Set) live evals, which each showed 0
regressions on their own surfaces. Canonical NS9 floors are preserved: medium
37/38, large 49/65, demo 11/15.

## End-to-end confirmation

A live RC1 run on `mx2_set_aesop_known` (the combined config, not a composition)
closed both `Set.Finite.toFinset_insert` and `Set.Finite.toFinset_offDiag` via
`aesop`, confirming the Multiset block does not interfere with the Set aesop
fallback and that RC1 behaves exactly as the composition predicts.
