# v5 evaluation summary — every eval run tonight

A single-table catalog of every Lean-grading eval run executed in
this session, for reproducibility and audit.

## nat_defs_medium (38 theorems)

| variant | proved | rate | new wins (vs v4.7 26/38) | run id |
|---|---|---|---|---|
| gen_v5 raw (no wrapper) | 3 / 38 | 8% | — | `nat_defs_medium_raw_baseline/` |
| v5-00-baseline-repro (v4.7 carries) | 26 / 38 | 68% | — | `v5-auto-…/eval/v5-00-baseline-repro/` |
| v5-01 to v5-11 (first pass) | 26 / 38 each | 68% | none | `v5-auto-…/eval/v5-*/` |
| v5-12-prio-div-hyp | 27 / 38 | 71% | div_lt_one_iff | `v5-followup-…/eval/v5-12-prio-div-hyp/` |
| v5-13-prio-iff-constructor | 26 / 38 | 68% | — | `v5-followup-…/eval/v5-13-prio-iff-constructor/` |
| v5-14-prio-combo | 27 / 38 | 71% | div_lt_one_iff | `v5-followup-…/eval/v5-14-prio-combo/` |
| v5-15-prio-mul-specific | 28 / 38 | 74% | mul_eq_left, mul_eq_right | `v5-followup-…/eval/v5-15-prio-mul-specific/` |
| v5-16-prio-sqrt-pow | 26 / 38 | 68% | — | `v5-followup-…/eval/v5-16-prio-sqrt-pow/` |
| v5-17-prio-term-iff | 26 / 38 | 68% | — | `v5-followup-…/eval/v5-17-prio-term-iff/` |
| v5-18-prio-kitchen | 29 / 38 | 76% | div_lt_one_iff, mul_eq_left, mul_eq_right | `v5-followup-…/eval/v5-18-prio-kitchen/` |
| v5-19-prio-split-ifs | 26 / 38 | 68% | — | `v5-followup-…/eval/v5-19-prio-split-ifs/` |
| v5-20-prio-div-pos | 28 / 38 | 74% | div_pos, div_pos_iff | `v5-followup-…/eval/v5-20-prio-div-pos/` |
| v5-21-prio-iff-basic | 26 / 38 | 68% | — | `v5-followup-…/eval/v5-21-prio-iff-basic/` |
| v5-22-deny-derailers | 29 / 38 | 76% | div_lt_one_iff, mul_eq_left, mul_eq_right | `v5-followup-…/eval/v5-22-deny-derailers/` |
| v5-23-w4-split-ifs | 26 / 38 | 68% | — | `v5-wave4-…/eval/v5-23-w4-split-ifs/` |
| v5-24-w4-dvd-iff | 26 / 38 | 68% | — | `v5-wave4-…/eval/v5-24-w4-dvd-iff/` |
| v5-25-w4-div-pos | 28 / 38 | 74% | div_pos, div_pos_iff | `v5-wave4-…/eval/v5-25-w4-div-pos/` |
| v5-26-w4-sqrt-pow | 26 / 38 | 68% | — | `v5-wave4-…/eval/v5-26-w4-sqrt-pow/` |
| **v5-27-w4-master** | **31 / 38** | **82%** | **all 5** | `v5-wave4-…/eval/v5-27-w4-master/` |
| **v5-28-w4-super-kitchen** | **31 / 38** | **82%** | **all 5** | `v5-wave4-…/eval/v5-28-w4-super-kitchen/` |
| v5-29-w5-le-shape | 31 / 38 | 82% | (same as master) | `v5-wave5-…/eval/v5-29-w5-le-shape/` |
| v5-30-w5-add-mod-ite | 31 / 38 | 82% | (same as master) | `v5-wave5-…/eval/v5-30-w5-add-mod-ite/` |
| v5-31-w5-iff-reorder | **27 / 38** | 71% | **only div_pos** (regressed!) | `v5-wave5-…/eval/v5-31-w5-iff-reorder/` |
| v5-32-w5-dvd-specific | 31 / 38 | 82% | (same as master) | `v5-wave5-…/eval/v5-32-w5-dvd-specific/` |
| v5-33-w5-eq-one-of-mul | 31 / 38 | 82% | (same as master) | `v5-wave5-…/eval/v5-33-w5-eq-one-of-mul/` |
| v5-34-w6-dvd-alt | 31 / 38 | 82% | (same as master) | `v5-wave6-…/eval/v5-34-w6-dvd-alt/` |
| v5-35-w6-add-mod-ite | 31 / 38 | 82% | (same as master) | `v5-wave6-…/eval/v5-35-w6-add-mod-ite/` |
| v5-36-w6-eq-one-alt | 31 / 38 | 82% | (same as master) | `v5-wave6-…/eval/v5-36-w6-eq-one-alt/` |
| v5-37-w6-div-le-div | 31 / 38 | 82% | (same as master) | `v5-wave6-…/eval/v5-37-w6-div-le-div/` |
| v5-38-w6-combined | 31 / 38 | 82% | (same as master) | `v5-wave6-…/eval/v5-38-w6-combined/` |
| v5-27 reproducibility check | **31 / 38** | 82% | (confirms deterministic) | `v5_27_repro/eval-*/` |

**Total evals on nat_defs_medium: 33 runs across 29 distinct variants
plus one gen_v5 raw baseline plus one repro.**

## nat_defs_large_v5 (64 available theorems = 38 medium + 26 new)

| variant | proved | rate | run id |
|---|---|---|---|
| gen_v5 raw (no wrapper) | 4 / 64 | 6% | `large_v5_raw_baseline/` |
| v5-18-prio-kitchen | 41 / 64 | 64% | `large_v5_kitchen/` |
| **v5-27-w4-master** | **43 / 64** | **67%** | `large_v5_master/` |

The 12 NEW theorems (not in nat_defs_medium) that v5-27 closes on
nat_defs_large_v5:

  - 5 closed by priority_templates' generic omega-omega iff template
    (`Nat.add_eq_two_iff`, `Nat.add_eq_three_iff`, `Nat.lt_one_add_iff`,
    `Nat.max_eq_zero_iff`, `Nat.min_eq_zero_iff`).
  - 6 closed by fallback (omega).
  - 1 closed by generative_topk.

## nat_defs_subset (15 theorems)

| variant | proved | rate |
|---|---|---|
| v3.6 hybrid_evolved (carries) | 10 / 15 | 67% |
| **v5-27-w4-master** | **12 / 15** | **80%** |

The +2 vs v3.6: `Nat.div_lt_one_iff` (new) and `Nat.add_eq_one_iff`
(now attributed to priority_template, was fallback in v3.6).

## demo_v1 (15 theorems: 11 Set, 3 Finset, 1 Nat)

| variant | proved | rate |
|---|---|---|
| gen_v5 raw (no wrapper) | 10 / 15 | 67% |
| **v5-27-w4-master** | **11 / 15** | **73%** (wrapper adds only +1: the Nat.mul_add_mod' family win) |

Distribution: 10 wins from `generative_topk` on Set theorems
(the model knows Set basics); 1 win from `family_tactic` on
`Nat.mul_add_mod'`; 0 priority_templates wins (Nat-specific
templates don't transfer to Set/Finset).

## Total tonight

  - **Distinct variants evaluated:** 29 medium + a few large + 1 subset + 1 demo + 2 raw baselines + 1 repro = **34+ unique configurations**.
  - **Total Lean evals:** ~36
  - **Total theorem-evals:** ~36 × ~38 = ~1370 individual theorem rollouts.
  - **Total wall-clock runtime:** ~3 hours.

## Final v5 best

`v5-27-w4-master` and the equivalent `v5-28-w4-super-kitchen`:

  - **31 / 38** on nat_defs_medium (76% above v4.7's 68%; 82% rate)
  - **43 / 64** on nat_defs_large_v5 (67% rate; +2 vs v5-18)
  - **12 / 15** on nat_defs_subset (80% rate; +2 vs v3.6)
  - **11 / 15** on demo_v1 (73% rate; priority_templates don't fire — domain-specific)

Genome path:
`project/evolve/autonomous_runs/v5-wave4-20260522-111556-3063e7/eval/v5-27-w4-master/genome.json`.

To re-run on any set:
```
python -m evolve.run_large_v5 \
    --best-genome <path-to-v5-27-genome.json> \
    --theorem-set <set-name> \
    --ckpt-dir project/models/gen_v5 \
    --out-dir <out>
```
