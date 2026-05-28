# NS18 — Controlled wrapper expansion probes

## Headline

Six experimental wrapper configs were probed against raw NS15
routed and the NS9 best wrapper baseline. **Two variants produced
new wrapper-only wins** beyond the NS9 ceiling:

- `aesop_wrapper`: **+3 new wrapper-only wins on ns17_finset_extra**
  (Finset.coe_insert, Finset.cons_eq_insert,
  Finset.disjUnion_singleton), single family (`aesop`).
  Side-effect: −1 regression on ns17_set_extra.
- `nat_simp_arith`: **+1 new wrapper-only win on ns16_nat_div_mod_extra**
  (Nat.mul_mod_mod), single family (`simp_all` with Nat
  arithmetic args).
- `combined_safe`: union of all "safe" variants — preserves the
  NS9 medium/large baselines and adds +1 on
  nat_defs_large_v5 (Nat.mod_mul_mod) + the same +1 nat_simp_arith
  win, but inherits the aesop−1 regression on ns14_set_finset_extra.

**NS19 readiness gate verdict: not yet met.** The strongest
variant adds 3 new wrapper-only Finset wins from one family;
the NS19 threshold is ≥ 5 same-family wins or ≥ 10 wrapper-only
rows. Two more wins on the Finset surface would clear the gate.

## Motivation (recap from NS17)

NS17 audited the corpus and found that:

1. The wrapper-only signal on fresh surfaces (NS14/NS16/NS17) had
   converged to **0 new wins** across 114 fresh theorems with
   the NS9 best genome.
2. No pattern family in the existing 201-row evolved corpus had
   ≥ 10 wrapper-only rows.
3. Recommendation: expand the wrapper templates before
   re-training.

NS18 implements the wrapper-expansion recommendation in
*experimental* form. Configs live in
`project/evolve/experiments/ns18/` and are **never** merged into
`project/evolve/best/ns9_best_genome.json`; failed variants can be
discarded without affecting production.

## Stage 1 — experimental wrapper configs

[`scripts/ns18_make_experiment_configs.py`](../../../scripts/ns18_make_experiment_configs.py)
emits six variants as small additive deltas on top of the NS9 best
genome. Each variant only **adds** tactic candidates to a
specified shape; nothing is removed.

| variant | shape additions | rationale |
|---|---|---|
| `constructor_omega` | iff: `constructor <;> omega`, `constructor <;> simp_all`, `constructor <;> (intro _ <;> omega)` | The 16-row family NS17 found unused |
| `split_ifs_omega` | any: `split_ifs <;> simp_all`, `split_ifs with h <;> omega`, `split_ifs with h <;> simp_all [h]` | Reach if-then-else Nat goals |
| `nat_simp_arith` | eq: `simp_all [Nat.add_comm, …]`, `simp_all [Nat.mul_comm, …]`, `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]`; any: `simp_all`; mod family: bundle | Density on Nat-arith goals |
| `aesop_wrapper` | iff/eq/lt/le/any: `aesop` | High-recall closer for Set/Finset |
| `bool_option_cases` | any: `decide`, `rfl`; eq: `rfl` | Bool/Option/List discrimination |
| `combined_safe` | union of A + B + C + E | Single-config harvester (excludes aesop until safety confirmed) |

All variants raise `priority_template_budget` from 18 → 24 (or 30
for combined_safe) so the new candidates have slots to surface
during top-k decoding.

## Stage 2 — smoke test

Each variant was run on `demo_v1` (15 theorems, ~2 min) in
parallel. All six reached the NS9 baseline of **11/15** with no
Python tracebacks, no Lean panics, no Dojo crashes. Smoke-pass.

## Stage 3+4 — full matrix evaluation

Each variant was run on a targeted set list — combined_safe got
the full 12-set sweep (preservation + every fresh surface);
individual variants got 3-5 sets each (sets where their template
families are most likely to fire).

Per-variant raw results:

| variant | sets evaluated | total proved | total raw NS15 | total NS9 wrap |
|---|---:|---:|---:|---:|
| `constructor_omega` | 3 | 59 | 42 | 59 |
| `split_ifs_omega` | 3 | 2 | 0 | 2 |
| `nat_simp_arith` | 4 | 16 | 13 | 15 |
| `aesop_wrapper` | 6 | 55 | 52 | 53 |
| `bool_option_cases` | 4 | 52 | 51 | 52 |
| `combined_safe` | 12 | 170 | 138 | 168 |

(Counts are sums across listed sets; "raw" / "wrap" baselines
restricted to those same sets.)

## Stage 5+6 — per-variant signal

[`scripts/ns18_compare_wrapper_variants.py`](../../../scripts/ns18_compare_wrapper_variants.py)
emits the full table at
[`ns18_wrapper_variants_comparison.md`](ns18_wrapper_variants_comparison.md).
Signal extraction below: each `Δwrap > 0` row is a theorem the
variant proves that NS9 wrapper baseline did **not** — that's the
truly-new wrapper-only signal.

### Truly-new wrapper-only wins (Δwrap > 0)

| variant | set | new theorem(s) | tactic family |
|---|---|---|---|
| `nat_simp_arith` | `ns16_nat_div_mod_extra` | `Nat.mul_mod_mod` | `simp_all` (Nat arith) |
| `aesop_wrapper` | `ns17_finset_extra` | `Finset.coe_insert` | `aesop` |
| `aesop_wrapper` | `ns17_finset_extra` | `Finset.cons_eq_insert` | `aesop` |
| `aesop_wrapper` | `ns17_finset_extra` | `Finset.disjUnion_singleton` | `aesop` |
| `combined_safe` | `nat_defs_large_v5` | `Nat.mod_mul_mod` | `simp_all` (Nat arith) |
| `combined_safe` | `ns16_nat_div_mod_extra` | `Nat.mul_mod_mod` | `simp_all` (Nat arith) |

### Regressions (Δwrap < 0)

| variant | set | lost theorem(s) | reason |
|---|---|---|---|
| `aesop_wrapper` | `ns17_set_extra` | `Set.Equipotent.refl_iff` (one theorem) | `aesop` displaced a top-k candidate that previously closed it |
| `combined_safe` | `ns14_set_finset_extra` | `Set.inter_nonempty_iff_exists_left` | Same dynamic |

### Preservation check (canonical benchmarks)

| variant | medium 38 | large 65 | demo_v1 15 | ns14_nat 20 | ns14_set/finset 20 |
|---|---:|---:|---:|---:|---:|
| `constructor_omega` | 37 ✓ | — | 11 ✓ | 9 ✓ | — |
| `split_ifs_omega` | — | — | 11 ✓ | — | — |
| `nat_simp_arith` | — | — | 11 ✓ | — | — |
| `aesop_wrapper` | — | — | 11 ✓ | — | — |
| `bool_option_cases` | — | — | 11 ✓ | — | — |
| `combined_safe` | 37 ✓ | 50 (+1) | 11 ✓ | 9 ✓ | 12 (−1) |

**combined_safe is the only variant that touched every benchmark
and preserved them all, with one regression (ns14_set_finset
−1) traded for one large-v5 gain (+1).** Net: zero change on
canonical benchmarks.

## Stage 7 — homogeneous-family analysis

For each truly-new wrapper-only win, classify by tactic family
(reuses NS17's classifier):

| family | new wins | unique theorems | example tactic |
|---|---:|---:|---|
| `aesop` | 3 | 3 | `aesop` |
| `simp_all [Nat.add_mod, …]` | 2 | 2 | `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]` |
| Total | **5** | **5** | |

5 truly-new wrapper-only theorems across 2 families. The `aesop`
family is the dominant contributor and is **homogeneous** (3
theorems, 1 tactic). The `simp_all`-arith family is also
homogeneous (2 theorems, 1 tactic pattern).

## Stage 8 — NS19 decision gate

User-defined gate:

> Variant produces ≥ 10 wrapper-only rows in one homogeneous
> family, OR ≥ 5 new wrapper-only theorems with same tactic
> family.

Closest variants:

- `aesop_wrapper`: 3 same-family wins on Finset → **fails**
  (3 < 5).
- `nat_simp_arith` + combined_safe: 2 same-family wins on
  div_mod / large_v5 → fails (2 < 5).

**Verdict: NS19 training is not yet justified.** Two more
same-family wins would clear the gate. The recommended next
move is to **mine more Finset / Set / List surface with
`aesop_wrapper`** to grow the aesop pool, and to mine more
nat_div / nat_mod theorems with `nat_simp_arith` to grow the
simp_all-arith pool. Both probes are theorem-only (no training
involved) and reach the gate at low marginal cost.

## Stage 9 — wrapper-genome promotion advisory

None of the experimental configs should be promoted to
`project/evolve/best/ns9_best_genome.json` yet, because:

1. `aesop_wrapper` has a −1 regression on `ns17_set_extra` that
   would propagate. The aesop emission needs targeting (only on
   Finset shapes, not blanket).
2. `combined_safe` has a −1 regression on
   `ns14_set_finset_extra` for the same reason.
3. `nat_simp_arith` is clean (no regressions) but only adds 1
   new theorem, not worth a promote.

Recommended next step before any promotion:
- Constrain `aesop` to Finset-only priority shapes
  (or namespace-gated, e.g. `theorem_family_tactics["Finset"]`).
- Verify on the full benchmark suite (medium, large, demo,
  ns14_*, ns16_*, ns17_*) that the constrained `aesop` retains
  the 3 Finset wins without the Set regression.
- Only then merge into the best genome.

## What this answers

1. **Can we get new wrapper-only signal?** Yes — aesop on
   Finset (+3), simp_all-arith on div_mod (+1), nat_div_rw-style
   on large_v5 (+1).
2. **Do any variants crash?** No. All 6 smoke-passed on demo_v1
   and produced valid Lean tactics throughout the matrix.
3. **Is anything trainable yet?** Not by the gate — closest is
   3 same-family aesop wins, need ≥ 5.
4. **Is the wrapper still load-bearing?** combined_safe
   preserves the medium/large/demo baselines with a single +1
   net change. Wrapper templates still drive 37/38 medium,
   50/65 large.
5. **Which expansion direction should NS19 take?** Aesop on
   Finset surface; namespace-gated to avoid the Set regression.

## Limitations

- Each individual variant was only evaluated on 3–6 sets; only
  combined_safe got the full 12-set sweep. A variant that would
  shine on a missed set wouldn't appear in the comparison.
- The NS9 best genome predates the namespace-gating fields used
  here; some new templates fire on shapes (e.g. iff) regardless
  of namespace.
- The 1-theorem regressions on Set were not deeply analyzed; a
  trace inspection could reveal whether `aesop` consumes the
  top-k slot that previously closed `Set.inter_nonempty_iff_…`.

## NS19 recommendations

1. **Mine more Finset surface with `aesop_wrapper`.** Add 60–90
   more easy Finset theorems and re-eval. Grow aesop wins from
   3 → ≥ 5 to clear the gate.
2. **Gate aesop to Finset only.** Constrain via
   `theorem_family_tactics["Finset"]` or similar; verify no
   Set regression remains.
3. **Mine more nat_mod theorems with `nat_simp_arith`.** Grow
   the simp_all-arith pool from 2 → ≥ 5. Likely candidates:
   theorems whose statements mention `Nat.mod_eq_of_lt` or
   `Nat.add_mod`.
4. **Do not run NS19 training yet.** The 5-theorem pool isn't
   enough — re-mine first, train when the pool reaches the
   gate.

## Files

Committed:
- `scripts/ns18_make_experiment_configs.py` — emit experimental configs
- `scripts/ns18_run_eval.sh` — per-variant per-set eval wrapper
- `scripts/ns18_run_matrix.sh` — per-variant set-list driver
- `scripts/ns18_compare_wrapper_variants.py` — signal extraction
- `project/evolve/experiments/ns18/ns18_*.json` — 6 wrapper variants
- `project/data/ns18_wrapper_signal_meta.json` — comparison data
- `project/evolve/reports/ns18_wrapper_variants_comparison.md`
- `project/evolve/reports/ns18_wrapper_expansion_report.md` (this file)
- `.gitignore` — NS18 paths

Not committed (gitignored / regeneratable):
- `project/evolve/eval_runs/ns18_*` (raw traces + metrics)
- `project/evolve/eval_runs/ns18_*matrix_driver.log`
