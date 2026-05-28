# v5 — reading guide

Quick orientation to the v5 autonomous-research-loop work shipped on
branch `v5-autonomous-proof-program-evolution`.

## Headline

| metric | value |
|---|---|
| v4.7 baseline (nat_defs_medium) | 26 / 38 |
| **v5 best (v5-27-w4-master)** | **31 / 38 (+5)** |
| v5-27 on nat_defs_large_v5 (64 thm) | 43 / 64 (67%) |
| gen_v5 raw on nat_defs_medium | 3 / 38 (8%) |
| gen_v5 raw on nat_defs_large_v5 | 4 / 64 (6%) |

Newly closed by v5 (vs v4.7):
  - `Nat.div_lt_one_iff`
  - `Nat.div_pos`
  - `Nat.div_pos_iff`
  - `Nat.mul_eq_left`
  - `Nat.mul_eq_right`

All five close via the new `priority_templates` slot — templates
that emit BEFORE the generative model's output. The fix to a
structural ordering bug surfaced by the autonomous loop.

## Reading order (from quickest to deepest)

  1. **`v5_central_findings.md`** — 7 distilled claims with evidence.
     Start here if you want the short answer.
  2. **`v5_priority_templates_insight.md`** — the structural finding
     (why ordering matters; how priority_templates fixes it).
  3. **`v5_autonomous_exploration.md`** — full scoreboards and
     per-variant analysis.
  4. **`nat_defs_medium_failure_classification_v5.md`** — what was
     unsolved going in.
  5. **`v5_research_plan.md`** — the steering plan written before
     any coding.
  6. **`v5_alphaevolve_architecture.md`** — v6 design proposal.
  7. **`v5_trace_to_training_plan.md`** — Direction E (no training
     tonight; pipeline shipped).
  8. **`nat_defs_medium_summary.md`** — running history; v3.6 → v5.

## Commits on this branch

  - `4e5c2e1` — main v5 code + reports
  - `79f6ae7` — wave 5 robustness probes + cross-domain check
  - `00cccc3` — central findings
  - `d64fc6e` — wave 6 targeted variants for remaining 7 failures
  - `ddfe44c` — wave 6 results + confirmed gen_v5 raw baselines

## Run artifacts (not committed; for reference paths)

  - `project/evolve/autonomous_runs/v5-auto-20260522-095802-1fcaa0/` — first pass (12 variants)
  - `project/evolve/autonomous_runs/v5-followup-20260522-103058-537f36/` — followup (11 variants); best v5-18 at 29/38
  - `project/evolve/autonomous_runs/v5-wave4-20260522-111556-3063e7/` — wave 4 (6 variants); best v5-27/v5-28 at 31/38
  - `project/evolve/autonomous_runs/v5-wave5-…/` — wave 5 (5 variants); confirms 31/38 ceiling
  - `project/evolve/autonomous_runs/v5-wave6-…/` — wave 6 (5 variants); no new closures
  - `project/evolve/autonomous_runs/large_v5_kitchen/` — v5-18 on nat_defs_large_v5
  - `project/evolve/autonomous_runs/large_v5_master/` — v5-27 on nat_defs_large_v5 (the published 43/64)
  - `project/evolve/autonomous_runs/nat_defs_medium_raw_baseline/` — gen_v5 raw on medium
  - `project/evolve/autonomous_runs/large_v5_raw_baseline/` — gen_v5 raw on large
  - `project/evolve/autonomous_runs/demo_v1_master/` — cross-domain check

## Code shipped

  - `evolve/strategy_wrapper.py` — `ORIGIN_TERM_BUILDER` and
    `priority_templates` slots; the new wrapper logic.
  - `evolve/autonomous_research_loop.py` — first-pass driver.
  - `evolve/autonomous_research_followup.py` — second-pass driver.
  - `evolve/autonomous_research_wave3.py` — adaptive seeding (not run tonight).
  - `evolve/autonomous_research_wave4.py` — targeted variants combining wins.
  - `evolve/autonomous_research_wave5.py` — robustness probes.
  - `evolve/autonomous_research_wave6.py` — final targeted attempts.
  - `evolve/run_large_v5.py` — Direction D eval.
  - `evolve/analyze_v5_runs.py` — cross-run scoreboard analyzer.
  - `evolve/v5_followup_variants.py` — followup-variant helper.
  - `scripts/build_v5_training_data.py` — Direction E pipeline (157 pairs).
  - `scripts/v5_followup_tldr.sh`, `scripts/launch_wave3.sh`.

## Best v5 genome

```
project/evolve/autonomous_runs/v5-wave4-20260522-111556-3063e7/
  eval/v5-27-w4-master/genome.json
```

Drop into any future eval via
`python -m evolve.run_large_v5 --best-genome <path> --theorem-set <set>`.

## What didn't close (the 7 remaining)

  - `Nat.AM_GM` — `nlinarith` unavailable in env.
  - `Nat.add_mod_eq_ite` — `split_ifs` advances but no closer.
  - `Nat.eq_one_of_mul_eq_one_left` — needs case analysis.
  - `Nat.div_le_div_right` — no working Mathlib lemma found.
  - `Nat.sqrt_lt` — `Nat.sqrt_lt'` doesn't exist in env.
  - `Nat.pow_lt_pow_iff_left` — self-reference; no alt form.
  - `Nat.dvd_iff_div_mul_eq` — dvd skeletons don't unify.
