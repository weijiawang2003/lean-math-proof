# Post-NS24 status — one page

State of the AlphaEvolve-style Lean proof-search project after NS24 and
the CX3 mining arc.

## The two tracks

- **NS9 wrapper** — a 17-skeleton strategy genome (search-time tactic
  ordering + retrieval). Strong on Nat; on Int it adds ~0 over the
  routed model; on Bool/Option it is a **no-op** (see CX3).
- **Learn track** — routed fine-tunes (CodeT5-small, 60M). A regex
  namespace router sends `^Nat\.`/`^Int\.`/`^Finset\.`/`^Set\.` to
  specialist checkpoints, everything else to `gen_v5_ns12_balanced`.

## Milestones

| arc | what | result |
|---|---|---|
| **NS15** | Nat distillation | broad Nat transfer; routed Nat specialist |
| **CX1** | catalog 527 → 1,817 | reopened mining (NS20 was *catalog* exhaustion, not framework) |
| **NS21** | Finset/aesop imitation | local Finset gains, no held-out transfer (mostly memorization) |
| **CX2** | Int iff/omega mining | 20/78 wrapper-only; iff_omega + fallback_omega gates met |
| **NS22** | Int omega distillation | **+22 raw Int wins (35 → 57)**; short `omega` absorbed, long iff-pair did not |
| **NS23** | minimal-tactic relabel | 9/10 Int iff_omega were `omega`-minimal; label = shortest sufficient tactic, not wrapper template |
| **NS24** | Int minimal-omega aggregate | **near-null +1 (57 → 58)**; Int omega surface **saturated** |
| **CX3** | Bool/Option mining | **negative**: wrapper-only = 0; only headroom is a structured `cases_simp` pool (NS22 non-memorizable class) |

## Current best (router = `ns24_router`)

- Nat → `gen_v5_ns15_nat_oversample` · Int →
  `gen_v5_ns24_int_minimal_omega_10x` · Finset →
  `gen_v5_ns21_finset_aesop_20x` · Set/default → `gen_v5_ns12_balanced`.
- Preservation: routed demo 10/15, nat_medium 23/38; wrapper demo
  11/15, nat_medium 37/38.

## What CX3 established

On a genuinely fresh namespace (Bool/Option, no base prior), the routed
default model already solves everything the wrapper does (43/83
identical), and the only count-meeting headroom is a compound
state-dependent `cases <;> simp` tactic — exactly the structured class
NS22 showed won't memorize at 60M. The mandatory minimal-tactic relabel
(NS23 discipline) is what caught this and prevented a wasteful NS25.

## Next target

The "fresh short-token family analogous to Int/omega" thesis did not
pan out for Bool/Option. Highest-yield options, in order:
1. **List/Multiset short-tactic surface** — large, unprobed by the
   wrapper-only-vs-routed lens; most likely to contain a clean
   `simp`/`aesop` short-token gate.
2. **State-conditioned `cases_simp` NS25 probe** (research bet) — test
   whether a variable-fill compound tactic memorizes at 60M, distinct
   from NS22's fixed template.
3. **Genuinely-unseen held-out Int** (~50 sub-bitwise/dvd candidates the
   CX2 audit left unprobed) — measures transfer vs the saturated pool.

See `project/evolve/reports/cx3_bool_option_decide_mining_report.md`,
`project/evolve/reports/ns24_int_minimal_omega_training_report.md`.
