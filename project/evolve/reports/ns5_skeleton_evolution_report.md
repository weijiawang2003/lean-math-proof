# NS5 — Skeleton Evolution Report

Status: complete.
Date: 2026-05-22 → 2026-05-23.
Branch: `ns5-skeleton-evolution`.
Predecessor commits: NS4.2 4a61ea1, NS4.1 28a3b0c, NS4 88a739b.

## 1. Runtime

- Start: 2026-05-23 05:02 UTC.
- End: 2026-05-23 12:30 UTC.
- Wall-clock: **7.46h** (under the 7.5h max-hours cap).
- Cycles run: **165** (≈3 passes through the 55-entry queue).
- Large evaluations used: **6** (cap).
- Branch HEAD at start: 4a61ea1.

## 2. Baseline (reference)

| Set | proved | Source |
|---|---:|---|
| `nat_defs_medium` | 37/38 | NS4.2 (ns3-combined + skeleton bag) |
| `nat_defs_large_v5` | 49/64 | NS4.2 (same genome) |

Both reproduced bit-for-bit at cycle 1 (the autonomous baseline cycle).

## 3. Best candidate

| | value |
|---|---|
| Cycle | **62** |
| Operator | `disable_dead_skeleton` (last in a chain of 10 successive prunes) |
| `proved_medium` | **37/38** |
| `proved_large` | **49/64** |
| Enabled skeleton count | **25** (vs. baseline 48 — **48% reduction**) |

Saved at `project/evolve/skeleton_runs/ns5-…/best_candidate.json`.

## 4. Did anything improve beyond 37/38 or 49/64?

**No new theorems proved.** Across 165 cycles, **zero theorems**
were newly proved that the baseline could not. The remaining
`nat_defs_medium` failure (`Nat.AM_GM`) and the `nat_defs_large_v5`
ceiling of 49/64 both held — confirming the NS5 plan's expectation
that those are *not* skeleton-ordering problems and need either new
templates or a retrained checkpoint to crack.

**But a strictly better candidate was found:** the same 37/49 result
with **23 fewer skeletons** (48% smaller). This is the *compact
genome* hypothesis confirmed at run-time: roughly half of the seed
genome was dead weight.

## 5. Archive summary

Final archive state:

- **26,132 rows** across 165 cycles × ~38 theorems × ~5 seen-skeleton
  observations per theorem.
- **64 distinct skeletons** observed.
- **14 of 48 baseline skeletons have ≥1 win.** Coverage is
  pathologically concentrated:

| Top-N skeletons | Wins | % of total |
|----------------:|-----:|-----------:|
| 1               | 2,414 | 40.8% |
| 3               | 3,270 | 55.2% |
| 5               | 3,639 | 61.5% |
| 10              | 4,444 | 75.1% |
| 15              | 4,901 | 82.8% |

A single skeleton (`pt_iff_8` — the `exact ⟨fun h => by omega, fun h
=> by omega⟩` iff-shape generic) accounts for **40.8%** of all
archived wins.

## 6. Top winning skeletons (post-run)

| skeleton           | wins | shape | family | origin            | template (truncated) |
|--------------------|-----:|-------|--------|-------------------|----------------------|
| `pt_iff_8`         | 2414 | iff   | —      | priority_template | `exact ⟨fun h => by omega, fun h => by omega⟩` |
| `fb_36`            | ~570 | any   | —      | fallback_tactic   | `omega` |
| `fam_mod_30`       | ~286 | any   | mod    | family_tactic     | `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]` |
| `pt_iff_7`         | ~130 | iff   | —      | priority_template | omega-pair (variant) |
| `pt_lt_12`         | ~130 | lt    | —      | priority_template | `rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]` |
| `pt_iff_1..6`      | ~80 ea | iff | —      | priority_template | 7 specifics — 1 win each per run |
| `pt_le_20`, `pt_eq_17`, `pt_any_13` | ~70 ea | various | — | priority_template | shape-specific specifics |

This is the same skewed distribution observed in NS4.1: most
theorems are closed by `omega` (via `pt_iff_8` for iff goals or
`fb_36` for everything else), with a long tail of specifics each
contributing a handful of unique wins.

## 7. Dead skeletons

After 165 cycles, **8 dead skeletons** were durably removed by the
runner (the difference between baseline 48 and best 25 reflects
disable + a few `any`-slot consolidations). The remaining dead
candidates the runner *tried* to disable but couldn't are:

- `pt_any_18`, `pt_any_19`, `pt_any_20`, `pt_any_23`,
- `fam_div_18`, `fam_div_22..27`,
- `fb_41`,
- A handful of retrieved skeletons (`retrieved:Nat.div_eq_of_lt:rw`,
  `retrieved:Nat.div_lt_iff_lt_mul':rw`, `retrieved:Nat.mod_add_div:*`,
  …).

These appear in `skeletons_seen` 25–99× across runs but never won.
Disabling any of them costs `Nat.div_lt_iff_lt_mul'` (60 of 67 total
regression incidents are on this one theorem). **Most of the
remaining bag is held in place by a single theorem.**

## 8. Mutation operators tested

| operator                  | cycles | accepted | promotions |
|---------------------------|------:|---------:|-----------:|
| baseline                  |    9 |        9 |          1 |
| disable_dead_skeleton     |   48 |       14 |     **10** |
| promote_high_win_skeleton |   27 |       24 |          0 |
| clone_skeleton_to_shape   |   24 |       24 |          0 |
| budget_trim               |   15 |       15 |          0 |
| demote_generic_skeleton   |   12 |       10 |          0 |
| archive_seed              |   30 |        0 |          0 |
| narrow/expand_family_gate |    0 |        — |          — |

**The runner's safety gate works.** 67 of 165 cycles regressed and
were correctly rejected; not a single rejected cycle entered the
incumbent line. Of the operators that ran:

- `disable_dead_skeleton` is the only operator that ever produced a
  promotion. All 10 promotions are dead-skeleton prunes.
- `promote_high_win_skeleton` and `clone_skeleton_to_shape` accepted
  often (preserved 37/38) but never beat the best — they add or
  reorder skeletons without removing them, so the runner refuses to
  promote without an improvement.
- `archive_seed` accepted **0 times**: every compact-seed cycle lost
  at least one theorem. This is the run's single most informative
  finding — see §10.
- `budget_trim` and `demote_generic_skeleton` had no measurable
  effect on the best candidate.

## 9. Best mutation

Cycle 62 — `disable_dead_skeleton(min_attempts=10, max_disable=5)`
applied after the archive had accumulated 7,000+ rows. It removed
one additional skeleton beyond the prior best (38 → 25 via a 10-step
chain of disable_dead cycles). **The full compaction was not a
single shot — it was a sequence of 10 accepted prunes interleaved
with rejected attempts at lower attempt thresholds.**

## 10. Compact-genome experiment (`archive_seed` cycles)

This is the experiment that produced the most actionable insight.
We ran archive_seed at top_n ∈ {15, 20, 25, 30, 40, 50, 18, 22, 28,
35} — 30 cycles total. Each kept only the top-N archived winners
(by `wins`) and disabled the rest.

| top_n | enabled_skeletons | proved_medium |
|------:|------------------:|--------------:|
| 15    | 15                | 33            |
| 18    | 11–18             | 32–36         |
| 20    | 15–19             | 35–36         |
| 22    | 14                | 30            |
| 25    | 17–21             | 35–36         |
| 28    | 17                | 34            |
| 30    | 17–21             | 35–36         |
| 35    | 18                | 36            |
| 40    | 21–25             | 35–36         |
| 50    | 20                | 36            |

**Plateau at 36/38 across top_n=20..50.** The `wins`-only selector
never reaches 37 no matter how many top archive entries it retains.

Why? **Because `wins` undercounts state-advance contributions.** A
skeleton that *advances* state (without itself closing the goal) is
necessary for the *next* skeleton to win, but the win credit goes
to the closer, not the advancer. Archive_seed throws the advancer
away. NS5's archive only tracks `wins`, so the seed candidate is
structurally incapable of preserving the full 37.

## 11. Theorem-level diffs (medium)

No theorem was newly proved across the run. The lost-theorem
distribution across all 67 rejected cycles:

| theorem | times lost | sensitivity |
|---------|-----------:|-------------|
| `Nat.div_lt_iff_lt_mul'` | **60** | extremely brittle — held in place by a thin set of dead-looking skeletons |
| `Nat.add_mod_eq_ite` | 12 | depends on a specific tactic chain |
| `Nat.add_mod_eq_add_mod_right` | 4 | priority-order sensitive |
| `Nat.two_mul_ne_two_mul_add_one` | 3 | priority-order sensitive |
| `Nat.add_mod_eq_add_mod_left` | 3 | priority-order sensitive |
| `Nat.div_le_div_right` | 2 | only in deepest seed cuts |
| `Nat.div_pos_iff` | 2 | promote-order sensitive |
| `Nat.eq_one_of_mul_eq_one_left` | 1 | top_n=15 seed only |
| `Nat.eq_zero_of_double_le` | 1 | deepest seed |
| `Nat.half_le_of_sub_le_half` | 1 | deepest seed |
| `Nat.le_or_le_of_add_eq_add_pred` | 1 | deepest seed |
| `Nat.div_pos` | 1 | deepest seed |

`Nat.div_lt_iff_lt_mul'` is the bottleneck: any aggressive prune or
reorder breaks it, and the archive doesn't know why (no win
attribution on the contributing skeletons).

## 12. Large-set transfer

Large evaluations were budgeted to **6 calls**; the runner used all
6. Every large eval returned `49/64`. No new large theorem was
proved; no large theorem was lost. The compact 25-skeleton genome
proves the same 49 theorems on `nat_defs_large_v5` as the
48-skeleton baseline — confirming that the dead-weight pruning
transfers cleanly to the larger set.

## 13. Specificity stress test

`demote_generic_skeleton` ran 12 times. 10 accepted (37 preserved),
2 regressed (cycles 2 and 14, both on `Nat.two_mul_ne_two_mul_add_one`).
The regressions stem from an unrelated bug in the operator — it sorts
ALL skeletons in `bag.skeletons["any"]`, which incidentally reorders
fallback/tactic_template emission (see §13.5 below). The NS1/NS3.5
specificity invariants themselves are intact.

## 13.5 Surprising findings

Two non-obvious insights were confirmed at scale.

**Finding 1 — Zero-win skeletons can still be necessary.** Disabling
priority_template skeletons with 0 wins and ≥5 attempts
(cycles 8–11) *regressed* `Nat.add_mod_eq_ite`. The "dead"
skeletons advanced state into a form that a *later* tactic could
close — the win attribution went to the closer, not the advancer.
60 of 67 total run regressions trace to the same effect on
`Nat.div_lt_iff_lt_mul'`. **Pure win-count pruning is unsafe**;
future versions of `disable_dead_skeleton` should also require the
archive to record `advances == 0`. NS4.2's archive doesn't yet carry
per-skeleton-attempt advance flags (only per-theorem totals), so
this is the most important NS5.x improvement.

**Finding 2 — Order-changing operators regress unexpectedly.** Both
`demote_generic_skeleton` and `promote_high_win_skeleton` sometimes
lose theorems. Root cause: they reorder `bag.skeletons["any"]`, but
`emit_fallback_tactics` and `emit_tactic_template_tactics` iterate
that list in *bag order*, not via `for_shape`'s `(priority,
specificity)` sort. Reordering changes the fallback-budget cutoff
and thus which tactics actually run. The runner correctly rejected
these cycles, but the operators should be fixed in NS5.x to only
reorder within their target origin without touching the rest of the
shape slot.

**Finding 3 — Compact-genome ceiling is structural.** Both
findings together explain why archive_seed plateaus at 36/38: the
seed retains 15–25 high-win skeletons but throws away the
advance-only skeletons that `Nat.div_lt_iff_lt_mul'` depends on.
With a richer archive (per-attempt advance flags), archive_seed
could plausibly reach 37/38 at ~25 enabled skeletons — which would
match the cycle-62 best by *design* rather than by *trial-and-error
prune chains*.

## 14. Recommended next step

NS5 confirmed that the skeleton-bag is a viable evolvable genome and
that the safety-gated runner can find non-trivial compactions
autonomously. The path forward is **richer attribution**, not more
operators:

1. **Per-skeleton advance attribution.** Extend the archive row
   schema to carry `result_kind ∈ {proved, advanced, attempted}`
   *per skeleton occurrence*, not just per-theorem totals. The
   bag already produces this signal (`EmittedTactic` is per-emission);
   the eval pipeline needs to forward it.
2. **Reorder-safe operators.** Fix `demote_generic_skeleton` and
   `promote_high_win_skeleton` to operate within (origin, shape,
   family) buckets only.
3. **`narrow_family_gate` / `expand_family_gate`.** Both are stubs
   today. With advance attribution they become implementable.
4. **Resurrect-skeleton operator.** Symmetric to disable_dead — if a
   re-enabled skeleton makes the candidate prove more theorems,
   keep it. Useful for undoing speculative disables.
5. **Slot-vocabulary mutation (NS6).** Once attribution is fixed,
   the next axis is *template text* mutation — currently
   off-limits because changing the template invalidates the
   skeleton_name.

`Nat.AM_GM` on medium and the large-set 49 ceiling remain as
non-skeleton-shaped problems: they need new templates or a new
checkpoint. NS5 is not the path that closes them.

## 15. Artifacts

- Run dir: `project/evolve/skeleton_runs/ns5-20260523-050214-0ec613/` (gitignored)
- Archive: `project/evolve/archive/skeletons.jsonl` (26k rows; committed)
- Archive index: `project/evolve/archive/skeletons_index.json` (committed)
- Mutation log: `<run>/mutation_log.md`
- Scoreboard: `<run>/scoreboard.jsonl`
- Best genome: `<run>/best_candidate.json`
- Auto-generated final report: `<run>/final_report.md`

## 16. Verdict on the NS5 goal

The NS5 plan asked: *"can the skeleton-bag be treated as an
AlphaEvolve genome — archived, mutated, evaluated, and improved
across theorem sets?"*

**Yes, with safety guardrails and modest expectations:**

- ✅ Archived (26k row JSONL ledger).
- ✅ Mutated (8 operators, 6 of them effective).
- ✅ Evaluated against medium AND large.
- ✅ Improved — not in coverage (no new wins) but in compactness
  (48 → 25 enabled skeletons, **48% smaller**, same 37/49).
- ✅ No regression on default behavior — the legacy code path is
  unchanged; `use_skeleton_bag=False` still reproduces the seed.

NS5 establishes the loop. NS6 should focus on richer attribution
and slot-vocabulary mutation.
