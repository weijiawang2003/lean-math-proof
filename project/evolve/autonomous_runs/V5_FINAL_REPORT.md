# v5 autonomous research — final report

**Branch:** `v5-autonomous-proof-program-evolution`
**Commits:**
  - `4e5c2e1` "Explore proof-program evolution for Lean tactic search" (main v5 code + reports)
  - `79f6ae7` "Add v5 wave 5 robustness probes + cross-domain check"
  - `00cccc3` "Add v5_central_findings.md — distilled claims with evidence"

**Start:** 2026-05-22 04:45 CDT
**End:** _to be filled at session close_

## 1. Headline numbers

| theorem set | candidate | proved | rate | Δ vs baseline |
|---|---|---|---|---|
| nat_defs_medium (38) | v4.7 baseline | 26 / 38 | 68% | — |
| nat_defs_medium (38) | v5 first-pass best | 26 / 38 | 68% | 0 |
| nat_defs_medium (38) | v5 followup (v5-18 kitchen) | 29 / 38 | 76% | +3 |
| nat_defs_medium (38) | **v5 wave 4 (v5-27 master)** | **31 / 38** | **82%** | **+5** |
| nat_defs_large_v5 (64) | v5-18 kitchen | 41 / 64 | 64% | — |
| nat_defs_large_v5 (64) | **v5-27 master** | **43 / 64** | **67%** | **+2 vs v5-18** |

## 2. Newly proved theorems on nat_defs_medium

All five close via the new `priority_templates` slot:

| # | theorem | priority template that closed it |
|---|---|---|
| 1 | `Nat.div_lt_one_iff` | `rw [Nat.div_lt_iff_lt_mul hb, Nat.one_mul]` |
| 2 | `Nat.div_pos` | `exact (Nat.le_div_iff_mul_le hb).mpr (by simpa using hba)` |
| 3 | `Nat.div_pos_iff` | `rw [Nat.pos_iff_ne_zero, Nat.div_ne_zero_iff hb]` |
| 4 | `Nat.mul_eq_left` | `exact ⟨Nat.eq_of_mul_eq_mul_left (Nat.pos_of_ne_zero ha) ..., simp [h]⟩` |
| 5 | `Nat.mul_eq_right` | `exact ⟨Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero hb) ..., simp [h]⟩` |

## 3. Method timeline

  - **Hour 0** — confirm v4.7 26/38 baseline. Create branch.
  - **Hour 0.5** — write v5_research_plan.md. Launch first-pass loop (12 variants).
  - **Hour 0.9** — first variant with new `term_builder` origin closes 14
    iff theorems via term-mode — but only re-attributing existing
    omega-omega wins. No new closures.
  - **Hour 1.0** — trace analysis on `Nat.div_lt_one_iff` reveals the
    structural ordering bug: model's `simp [...]` shadows downstream
    templates. **`priority_templates` slot designed and shipped.**
  - **Hour 1.1** — followup loop (11 variants) launched in parallel with
    remaining first-pass cycles.
  - **Hour 1.3** — first new win: `Nat.div_lt_one_iff` via v5-12.
  - **Hour 1.5** — two more wins: `Nat.mul_eq_left/right` via v5-15.
  - **Hour 1.6** — v5-18 kitchen-sink stacks all three: **29/38**.
  - **Hour 1.8** — direction D check: v5-18 transfers to nat_defs_large_v5
    (41/64); 5 of the 26 unseen theorems close via priority_templates.
  - **Hour 1.9** — v5-20 finds two more priority targets:
    `Nat.div_pos`, `Nat.div_pos_iff`.
  - **Hour 2.0** — wave 4 launched (6 variants combining priorities).
  - **Hour 2.4** — v5-27-w4-master / v5-28-w4-super-kitchen close all
    five together: **31/38**.
  - **Hour 2.5** — v5-27 evaluated on nat_defs_large_v5: **43/64**
    (+2 over v5-18 with no regression).
  - **Hour 2.6** — commit `4e5c2e1`.

## 4. The central architectural finding

The v3-v4 wrapper has a hidden ordering bug. The ranked-list iterator
is "first non-erroring tactic wins". The model's output goes first.
When the model produces a *weak simp* that advances state but doesn't
close the goal, every downstream family / template / term_builder
entry is silently bypassed.

Twelve first-pass variants — covering Directions A, B, C of the
research plan — each failed in exactly this way. They added more
candidate templates, but those templates ran AFTER the model and
never got to try.

The fix is one new genome slot: `priority_templates: dict[shape, list]`.
Templates in this dict emit BEFORE the model's output. Used surgically
for goal shapes where we have strong family knowledge (rewrites with
hypothesis arguments, term-mode iff splits with specific cancellation
lemmas, etc.).

This is the strongest empirical case in the project for **outer-tier
mutation** — adding a genome slot — vs. inner-tier slot-content
mutation. The v3-v4 mutator could have run forever and not produced
this fix because the slot did not exist. The slot itself had to be
added by hand after the autonomous loop made the same failure
twelve times.

## 5. Direction-by-direction summary

  - **Direction A — term_builder origin:** mechanism shipped, 14
    attributions per cycle, but no new closures because of the same
    ordering bug. Vindicated as a mechanism only when used inside the
    new `priority_templates` slot.
  - **Direction B — shape-specific mini-solvers:** mechanism shipped,
    same ordering bug. The "what to put in priority_templates" content
    came directly from Direction B brainstorming.
  - **Direction C — proof-skeleton mutation:** inner-tier mutation
    on term_builder skeletons. Same shadowing. Confirms that
    inner-tier mutation alone can't escape an outer-tier limit.
  - **Direction D — generalization:** v5-27 closes 43/64 on
    nat_defs_large_v5 (vs 41/64 for v5-18, vs 64% baseline rate on
    medium). The five new priority_templates wins generalize cleanly:
    `Nat.add_eq_two_iff`, `Nat.add_eq_three_iff`, `Nat.lt_one_add_iff`,
    `Nat.max_eq_zero_iff`, `Nat.min_eq_zero_iff` are unseen iff
    theorems closed by the generic priority template.
  - **Direction E — trace-to-training:** `scripts/build_v5_training_data.py`
    is shipped. Currently produces **157 (state, tactic) pairs**
    across 29 theorems, with the three new wins held out so a future
    `gen_v5+1` fine-tune has a fair test.
  - **Direction F — v6 architecture:** proposal in
    `v5_alphaevolve_architecture.md`. Skeleton-bag genome, two-tier
    mutator, archive, transfer protocol.

## 6. What did NOT close

7 of the original 12 unsolved theorems remain unsolved at v5:

  - `Nat.AM_GM` — needs `nlinarith [sq_nonneg (a-b)]`; tactic absent
    in env. Out of scope without adding the tactic.
  - `Nat.add_mod_eq_ite` — `split_ifs` advances state once but the
    remaining branch can't close with `omega` or `simp_all`. Needs a
    smarter inner tactic (or a skeleton that does the case-analysis
    AND closes each case differently).
  - `Nat.eq_one_of_mul_eq_one_left` — `m * n = 1 → n = 1`. Needs
    case analysis. No priority template tried (eq shape, no convenient
    hypothesis placeholder).
  - `Nat.div_le_div_right` — needs a Mathlib lemma chain not yet
    catalogued.
  - `Nat.sqrt_lt` — `Nat.sqrt_lt'` doesn't exist in this env. No
    alternative form found tonight.
  - `Nat.pow_lt_pow_iff_left` — self-reference; no working
    alternative.
  - `Nat.dvd_iff_div_mul_eq` — dvd templates tried, none close.

The first three are clean follow-ups for v6 (add the missing
tactics/skeletons). The last four are likely env-limitations that
no template change can fix; they probably require either retraining
or graduating to a newer Mathlib commit.

## 7. Files shipped

### Code
  - `evolve/strategy_wrapper.py` — `ORIGIN_TERM_BUILDER`,
    `term_builder_templates`, `priority_templates`. Wrapper logic.
  - `evolve/candidate.py` — new genome fields.
  - `evolve/evaluator.py` — pass through.
  - `evolve/autonomous_research_loop.py` — first-pass loop driver.
  - `evolve/autonomous_research_followup.py` — followup loop (priority_templates focus).
  - `evolve/autonomous_research_wave3.py` — adaptive (seeded from prior scoreboard).
  - `evolve/autonomous_research_wave4.py` — targeted variants for remaining theorems.
  - `evolve/run_large_v5.py` — Direction D eval driver.
  - `evolve/analyze_v5_runs.py` — cross-run scoreboard analyzer.
  - `evolve/v5_followup_variants.py` — followup-variant helper.
  - `eval_rollout_all.py` — `term_builder_*` counters, `priority_templates` plumbing.
  - `tasks.py` — `nat_defs_large_v5` (68 theorems).
  - `scripts/build_v5_training_data.py` — Direction E pipeline.
  - `scripts/v5_followup_tldr.sh`, `scripts/launch_wave3.sh` — convenience.

### Reports
  - `project/evolve/reports/v5_research_plan.md`
  - `project/evolve/reports/nat_defs_medium_failure_classification_v5.md`
  - `project/evolve/reports/v5_priority_templates_insight.md`
  - `project/evolve/reports/v5_trace_to_training_plan.md`
  - `project/evolve/reports/v5_alphaevolve_architecture.md`
  - `project/evolve/reports/v5_autonomous_exploration.md`
  - `project/evolve/reports/nat_defs_medium_summary.md` (v5 section appended)

### Run artifacts (NOT committed)
  - `project/evolve/autonomous_runs/v5-auto-*/` — first-pass run
  - `project/evolve/autonomous_runs/v5-followup-*/` — followup run
  - `project/evolve/autonomous_runs/v5-wave4-*/` — wave 4 run
  - `project/evolve/autonomous_runs/large_v5_kitchen/` — v5-18 on large
  - `project/evolve/autonomous_runs/large_v5_master/` — v5-27 on large
  - `project/seq2seq_data_v5_evolve.jsonl` — trace-to-training output
    (committed-ish, depends on .gitignore)

## 8. Recommended next branch

`v6-skeleton-bag-genome` per `v5_alphaevolve_architecture.md`:

  1. Refactor wrapper to use a `dict[shape, list[Skeleton]]` skeleton
     bag instead of separate `tactic_templates` / `family_tactics` /
     `term_builder_templates` / `priority_templates` lists.
  2. Two-tier mutator (outer = add/remove skeleton; inner = mutate
     slot content).
  3. Cross-run archive in `project/evolve/archive/skeletons.jsonl`.
  4. **gen_v5+1 fine-tune** on
     `project/seq2seq_data_v5_evolve.jsonl`. Hold out the three
     priority-templates wins; success = the trained model produces
     `rw [Nat.div_lt_iff_lt_mul hb, Nat.one_mul]` on
     `Nat.div_lt_one_iff` without the wrapper's help.
