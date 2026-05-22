# nat_defs_medium — experiment progression

A concise audit of the AlphaEvolve-style outer loop on Mathlib `Nat.Defs`,
spanning iterations **v3 → v3.6**. All numbers below are real Lean
evaluations through `eval_rollout_all.py`; no retraining; same `gen_v5`
checkpoint at every stage; no premise retrieval; no external LLM API.

## Headline

| stage | theorem set | size | proved | rate |
|---|---|---|---|---|
| gen_v5 baseline (no wrapper) | `nat_defs_subset` | 15 | **0** | 0% |
| v3 — omega in fallback | `nat_defs_subset` | 15 | **8** | 53% |
| v3.1 — `Nat.add_mod` fallback | `nat_defs_subset` | 15 | **9** | 60% |
| v3.2 — fallback re-ordering | `nat_defs_subset` | 15 | **10** | 67% |
| v3.3 — anti-loop diagnostic (off by default) | `nat_defs_subset` | 15 | **10** | 67% |
| v3.4 — theorem-family tactics | `nat_defs_subset` | 15 | **10** | 67% |
| v3.5 — cleanup + scale-out | `nat_defs_medium` | 38 | **25** | 66% |
| v3.6 — per-theorem deny-list (no crashes) | `nat_defs_medium` | 38 | **25** | 66% |
| gen_v5 baseline (no wrapper) | `nat_defs_medium` | 38 | **3** | 8% |
| **hybrid_evolved (v3.6) — Δ over baseline** | `nat_defs_medium` | 38 | **+22** | **+58 pp** |

Same model. Same Lean. The 22-theorem gap is closed entirely by a
deterministic strategy wrapper that selects ranked fallback tactics,
theorem-family-specific tactics, and a small per-theorem tactic
deny-list — all evolved (or hand-edited where the evolved ordering is
locked in as the new seed) inside an AlphaEvolve outer loop.

## What the wrapper actually does

The wrapper is `evolve.strategy_wrapper.StrategyWrapperPolicy`:

1. **Base policy**: `GenerativePolicy(gen_v5)` returns its beam-search
   top-k tactics. Unchanged. Same checkpoint as the baseline.
2. **Theorem-family tactics**: if the theorem name contains a configured
   substring (`div`, `mod`, `AM_GM`), prepend that family's tactic list
   (deduped against base, capped by per-family budget). Adds *targeted*
   knowledge per theorem family.
3. **Generic fallback tactics**: append a deterministic, evolved list of
   generic closers (`omega`, `simp_all`, `simp [Nat.add_mod, …]`, etc.).
4. **Tactic templates**: append per-Nat-variable rendered templates
   (`induction {var}`, `cases {var}`, etc.). Per-state variable
   extraction via regex.
5. **Per-theorem deny-list (v3.6)**: filter out a small set of
   `(theorem_name, tactic_substring)` pairs known to crash the Dojo
   REPL on that specific goal. Targeted, not global.

The final list is capped at a per-state budget and tried in order
(top-k fallback semantics): try tactic 1, if it errors try tactic 2,
etc. Either a tactic finishes the proof, a tactic advances state and we
recurse to the next step, or all of them error and the theorem fails.

## Win attribution on `nat_defs_medium` (38 theorems)

`proved_by_origin = {fallback_tactic: 18, family_tactic: 4, generative_topk: 3}`

| origin | wins | examples |
|---|---|---|
| `fallback_tactic` (`omega`, `simp_all`, …) | 18 | every `Nat.add_*`, `Nat.lt_*`, `Nat.le_*`, `Nat.eq_*`, `Nat.sub_*`, `Nat.two_mul_*` win |
| `family_tactic` (mod family) | 4 | `Nat.add_mod_eq_add_mod_left/right` via `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]`; `Nat.mod_two_ne_one/zero` via `omega` |
| `generative_topk` (gen_v5 model) | 3 | `Nat.lt_iff_add_one_le` (`simp_arith`), `Nat.succ_succ_ne_one` (`simp [Nat.mul_zero]`), `Nat.pred_eq_of_eq_succ` (`simp_all`) |

The model's own 3 wins are also baseline wins (no regression). The other
22 are pure wrapper contribution.

## Generalization evidence

`nat_defs_medium` extends `nat_defs_subset` with 23 new theorems drawn
from `Mathlib/Data/Nat/Defs.lean` across the `add`, `mul`, `lt`, `le`,
`eq`, `sub`, `mod`, `div`, `one`, `succ`, `pred`, `sqrt`, `pow`, `dvd`,
`two` name prefixes. None of these 23 were targeted during v3 → v3.4
evolution — the seed was tuned against the 15-theorem subset.

| segment | size | proved | rate |
|---|---|---|---|
| inherited subset (15) | 15 | 10 | **66.7%** |
| **new in medium (23)** | **23** | **15** | **65.2%** |
| combined | 38 | 25 | 65.8% |

Within 1.5 pp on identical wrapper, identical model, identical seed.
The strategy is doing real work, not memorising the 15 it was evolved
on. The `mod` family in particular generalises cleanly: it activates
on 5 medium theorems (3 from subset + 2 new), wins 4 (2 inherited + 2
new — `Nat.mod_two_ne_one`, `Nat.mod_two_ne_zero`).

## Remaining limitations

1. **`div` family: 0 wins.** Activates on all 6 div theorems in the
   medium set, advances state on multiple (`rw [Nat.div_eq_of_lt]`
   modifies the goal), but no follow-up closes. The remaining 5 div
   theorems (`Nat.div_le_div_right`, `Nat.div_lt_iff_lt_mul'`,
   `Nat.div_lt_one_iff`, `Nat.div_pos`, `Nat.div_pos_iff`,
   `Nat.dvd_iff_div_mul_eq`) likely need either premise retrieval
   (`Nat.div_le_div_right_aux` and the matching elimination lemmas) or
   an `induction` invocation properly scoped to the div recursion.
2. **`Nat.AM_GM` unresolved.** This Lean environment doesn't ship the
   tactics that ordinarily close this kind of goal (`nlinarith`,
   `ring_nf`, `positivity`, `nlinarith [sq_nonneg (a - b)]` all report
   `unknown tactic`). v3.5 stripped them from the AM_GM family. To
   actually close `Nat.AM_GM` we'd need to import the corresponding
   Mathlib tactic packages, which is outside the wrapper's scope.
3. **`Nat.add_mod_eq_ite` and the residual REPL crash.** v3.4 traced a
   `DojoCrashError` on a `<;>`-chained `by_cases + simp_all + omega`
   tactic against this theorem's ite-shaped goal. v3.5 removed all
   `<;>` combinators, but observed the same crash on plain
   `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]` — that exact tactic still
   wins `Nat.add_mod_eq_add_mod_left/right` cleanly, so it cannot be
   removed globally. **v3.6** introduces a per-theorem deny-list that
   filters this specific `(theorem, tactic)` pair only. With that, the
   theorem fails normally instead of crashing.
4. **No premise retrieval.** The wrapper has no access to the Mathlib
   library beyond the lemma names baked into the fallback / family
   tactics. A semantic premise retriever (the deferred component) would
   likely unlock the div cluster.
5. **No retraining.** Every win on `nat_defs_medium` was obtained from
   the *same `gen_v5` checkpoint* used by the 3/38 baseline. The
   improvement is purely from search-time strategy, not learned weights.

## Method invariants — preserved across every iteration

- No checkpoint changes; `gen_v5` was the model at every stage.
- Lean (via LeanDojo) is the sole grader.
- Same `top_k = 8`, `max_steps = 8` as the baseline run.
- Decoding is deterministic (beam search).
- The evolutionary mutator is deterministic (RNG seeded from
  `(parent.name, generation, index)`).
- The evolved seed for each stage is the proved-10/15 ordering inherited
  from the prior stage's best candidate.

## Artifacts

- `project/evolve/runs/evolve-20260521-093316-8d8595/` — v3.4 subset
  (10/15, first run with family tactics, residual `<;>` crash)
- `project/evolve/runs/evolve-20260521-181556-250a59/` — v3.5 subset
  (10/15, AM_GM cleanup applied)
- `project/evolve/runs/evolve-20260521-182223-1f6a34/` — v3.5 medium
  (25/38, hybrid_evolved, residual `DojoCrashError` on `Nat.add_mod_eq_ite`)
- `project/evolve/runs/evolve-20260521-184742-70ac3e/` — **v3.6 medium**
  (25/38, hybrid_evolved + per-theorem deny-list, 0 crashes — the
  reproducible published run)
- `/tmp/gen_v5_baseline_medium_v3_6/` — paired gen_v5 plain baseline
  (3/38) used to generate the v3.6 report
- Generated report: `project/evolve/reports/nat_defs_medium_v3_6.md`

## v4.1 / v4.2 — Div-family premise retrieval

v4.1 wired a static premise retriever into the `hybrid_evolved` wrapper so
that `Nat.div_*` / `Nat.dvd_*` theorems get a curated `Nat.div` lemma
bucket converted to `rw / simp / exact / apply` tactic candidates. The
plumbing landed in `premise_retriever.retrieve_for_state`,
`StrategyWrapperPolicy.retrieval_*`, and the eval / candidate / evolve
config surfaces. **Outcome**: 25/38 preserved exactly, with retrieval
activated on all 6 div theorems but **0 new closures**. The diagnostic
report identified three structural failures: target-theorem
self-retrieval, lemmas unavailable in the eval-env import closure (200
`unknown constant` errors), and tactic-form shotgun firing.

v4.2 shipped the hygiene layer over v4.1. Three filters added to
`retrieve_for_state`:

* **Self-filter**: target theorem excluded from its own retrieved set.
* **Static unavailability denylist** (`_UNAVAILABLE_LEMMAS`): 7 lemmas
  empirically observed to produce `unknown constant` — 2 genuinely
  outside the import closure (`Nat.div_eq_zero_iff`,
  `Nat.div_le_iff_le_mul`) plus 5 forward-reference target theorems that
  are unknown at the proof position of other in-file targets
  (`Nat.div_le_div_right`, `Nat.div_lt_one_iff`, `Nat.div_pos`,
  `Nat.div_pos_iff`, `Nat.dvd_iff_div_mul_eq`).
* **Tactic-form ablation**: default forms shrunk to `["rw","simp","apply"]`
  (dropped `exact`, which produced zero wins and many type-mismatch errors
  in v4.1).

**Outcome (v4.2)**: 25/38 preserved, **zero `unknown constant` errors**
(200 → 0), retrieval attempts **−61 %** (783 → 303), wallclock −1 min
vs v4.1. Baseline B (v4.2 wrapper, retrieval off) confirms 25/38 with
identical `proved_by_origin` — retrieval is a clean addition. The
remaining 16 retrieved-tactic state advances all hit one pathological
pattern (`apply Nat.lt_of_lt_of_le` growing the goal count by 2 per
step on `Nat.div_le_div_right` / `Nat.div_lt_one_iff`) that v4.3 will
need a goal-shape filter to suppress.

- `project/evolve/runs/evolve-20260521-233937-cf2370/` — v4.1 medium
  (25/38, retrieval plumbing, 200 unknown-constant errors,
  pre-hygiene baseline)
- `project/evolve/runs/evolve-20260522-025521-bc3e5a/` — **v4.2 medium**
  (25/38, retrieval + self/unavailable filters, **0 unknown-constant
  errors**, current published run)
- `/tmp/v4_2_baseline_B/` — Baseline B paired run (v4.2 wrapper, retrieval
  off): 25/38, same proved_by_origin — confirms retrieval is a clean
  addition rather than a contributing factor to the 25 wins
- Generated reports:
  `project/evolve/reports/nat_defs_medium_v4_1.md`,
  `project/evolve/reports/nat_defs_medium_v4_2.md`

## v4.3 — Goal-shape filter on retrieved `apply` tactics

v4.1/v4.2 surfaced one structural waste pattern that the hygiene layer
did not address: the only retrieved tactic that ever produced a state
advance in v4.2 was `apply Nat.lt_of_lt_of_le`, and every one of those
16 advances strictly grew the open-goal count by 2 (1→3→5→7→9→11) on
`Nat.div_le_div_right` / `Nat.div_lt_one_iff` / `Nat.div_pos_iff`. The
extra subgoals could never be closed by the wrapper, so each accepted
bloat advance just added 27 wasted Lean roundtrips at the next step.

v4.3 adds a per-theorem goal-shape filter to `rollout_one_theorem`:

* When a retrieved `apply LEMMA` produces a TacticState transition with
  `num_goals_after > num_goals_before`, the trace is written with
  `bloat_rejected=True`, the lemma joins a per-theorem
  `bloating_apply_lemmas` set, and the advance is **not taken** (the
  rollout continues at the same state).
* Subsequent retrieved `apply LEMMA` candidates on the same theorem
  emit `SkippedBloatingApply` traces and are skipped before Lean is
  invoked.
* The lemma is NOT globally banned — `rw [LEMMA]` / `simp [LEMMA]` for
  the same lemma still flow.
* Toggled by `SearchCandidate.retrieval_skip_bloating_apply` (default
  True). The default seed enables it.

**Outcome (Experiment B, the new default)**: 25/38 preserved (zero
regressions), `proved_by_origin` bit-identical to v3.6. The 16
pathological apply chains in v4.2 collapse to 3 first-time observations
(one per affected theorem) which are then rejected; the chain is killed
before it eats Lean budget. Retrieval attempts drop from v4.2's 303 to
**121** (-60%). Wallclock drops from 4m19s to **3m42s** (-37s).
Errored/Exhausted distribution restored to v3.6's shape (10/3) — the
v4.1/v4.2 EXH @8 status on 3 div theorems was an artifact of the
bloating chain extending the rollout, not real progress.

Two ablations confirm:

* **Experiment A** (bloat filter OFF, forms `rw/simp/apply`) reproduces
  v4.2 (303 attempts, 16 bloat advances, 25/38, errored 7 / exhausted 6)
  — confirming the v4.3 code path is behavior-equivalent to v4.2 when
  the filter is off.
* **Experiment C** (bloat filter ON, forms `rw/simp` only) — same 25/38,
  73 attempts, 3m28s, but loses the `apply` diagnostic surface.

Default kept as B (apply enabled, bloat-filtered) because the
diagnostic value of seeing which lemmas bloat outweighs the 0.2 min /
48-roundtrip saving of dropping `apply` entirely.

- `project/evolve/runs/evolve-20260522-034524-d10acf/` — **v4.3 / B
  medium** (25/38, retrieval + bloat filter, 16 → 3 bloat events,
  current published default)
- `/tmp/v4_3_exp_A/`, `/tmp/v4_3_exp_C/` — ablation runs
- Generated report: `project/evolve/reports/nat_defs_medium_v4_3.md`

## v4.4 — Shape-aware retrieval ranking

v4.3 killed the symptom (apply-bloat) but identified the deeper cause:
4 of the 6 unproved div theorems have `iff` goals, 1 has `le`, 1 has
`lt` — yet v4.3 default emitted `apply LEMMA` against every retrieved
lemma regardless of goal shape, guaranteed to `failed to unify` on iff
goals.

v4.4 classifies the goal's head connective and gates which forms each
retrieved lemma emits per the `(goal_shape, lemma_shape)` pair:

* `premise_retriever.classify_goal_shape(state_pp)` returns one of
  `eq` / `iff` / `lt` / `le` / `dvd` / `and` / `or` / `unknown` from
  the `⊢` line.
* `lemma_shape_from_name(name)` heuristically tags catalog lemmas
  (16/16 match their conclusion shape on the `Nat.div` bucket).
* `_SHAPE_FORM_ALLOW[(goal, lemma)] → set[form]` whitelist: iff×iff
  emits rw/simp/apply; iff×eq emits rw/simp; le×le emits apply/exact;
  iff×lt emits rw/simp (no apply); etc. Default fallback for unlisted
  pairs is `{rw, simp}`.

The retriever also adds a shape-bonus to scoring (+1.5 for exact
match, smaller bonuses for compatible cross-shape pairs, −0.5
penalty for distant mismatch with no token overlap). Wrapper now
exposes `last_retrieved_shapes` / `last_goal_shape` /
`last_shape_mismatch_filtered_count`; trace records tagged with
`goal_shape` / `tactic_retrieved_shape` / `shape_match`.

**Outcome**: 25/38 preserved with `proved_by_origin` bit-identical to
v3.6. **213 shape-mismatched forms suppressed** across 6 div theorems
(63 emissions on average, 6+ forms suppressed per `rank_tactics` call).
**Zero `unknown constant` errors** in retrieved traces (v4.2 hygiene
preserved). **3 div theorems escape the v4.3 ERR @early failure mode**
and now run to EXH @8 — confirming shape-aware emission unblocks
exploration. Wallclock 4m 21s (vs v4.3's 3m 42s; the +0.6 min is paid
to run those theorems deeper, not on wasted Lean roundtrips).

**Still no new proofs.** The remaining 6 div theorems all fail for
structural reasons the form filter cannot fix: hypothesis-chaining gaps
(`Nat.div_pos_iff.mpr ⟨_, _⟩` needs a term builder) and induction-
template gaps (`Nat.div_le_div_right` needs induction on the `a ≤ b`
hypothesis). v4.5 → induction templates.

- `project/evolve/runs/evolve-20260522-044315-1c0395/` — v4.4 medium
  (25/38, retrieval + shape filter, 213 shape-mismatch forms suppressed)
- Generated report: `project/evolve/reports/nat_defs_medium_v4_4.md`

## v4.5 — Structured div-family templates

v4.4's report flagged hypothesis-chaining and induction templates as
the remaining wall. v4.5 attacks the induction half: add 8 structured
templates to `theorem_family_tactics["div"]` covering iff-constructor
splits, induction on `≤`-hypotheses, and `rw [Nat.div_lt_iff_lt_mul hb,
Nat.mul_one]`-style chains. The wrapper gains hypothesis placeholders
`{hyp_le}` / `{hyp_pos}` / `{hyp_ne_zero}` so templates render against
the actual hypothesis names in scope (`h`, `hba`, `hb` etc.) and skip
silently when the required hypothesis is absent.

* `_extract_hypotheses(state_pp)` scans hypothesis lines (stops at `⊢`)
  and returns a dict of {placeholder → name|None}.
* `_render_template(template, nat_vars, hypotheses)` accepts the dict;
  templates whose placeholders have no backing hypothesis render to
  the empty list and are dropped.
* Templates 7–8 (`induction {hyp_le} with | refl => ... | step _ ih => ...`)
  test the multi-line `induction with` form for the first time in
  this codebase.
* `family_budgets["div"]` bumped 12 → 20 to fit the 8 new templates.

**Outcome**: 25/38 preserved with `proved_by_origin` bit-identical to
v3.6. **Zero new closures.** All 8 templates emit correctly (hypothesis
substitution traced as expected — `h`, `hba`, `hb`) but every template
hits one of three failure modes:

1. **Lemma name/signature wrong for actual mathlib** — `Nat.le_refl`
   produces type-mismatch (probable rename); `rw [Nat.div_lt_iff_lt_mul
   hb, ...]` produces "did not find instance of the pattern" (argument
   order or LHS/RHS swap).
2. **`simp`/`simp_all` cannot fire hypothesis-conditioned lemmas**
   without explicit hypothesis hints — every `simp [Nat.div_lt_iff_lt_mul,
   Nat.mul_one]` produces "made no progress" across all 6 div theorems.
3. **`omega` and `simp_all` don't understand `Nat.div`** — `constructor
   <;> intro h_split <;> omega` correctly splits iff goals but neither
   omega nor simp_all can close the resulting subgoals with `a/b` in
   them.

One small win: `Nat.dvd_iff_div_mul_eq` moved from v4.4's ERR @3 to
v4.5's EXH @8 — `constructor <;> intro h_split <;> simp_all` advances
state where every prior tactic errored. No proof, but more exploration.

No crashes, no regressions, no new `unknown constant`. The
infrastructure for hypothesis-aware templates is solid; the *content*
of the new templates needs verification against the actual mathlib
declarations the eval environment uses. v4.6 should write a
template-side analog of v4.2's `_UNAVAILABLE_LEMMAS` (a `#check`
pass at run-evolve setup time) before adding more templates.

- `project/evolve/runs/evolve-20260522-050325-0fe236/` — **v4.5 medium**
  (25/38, retrieval pipeline + 8 structured div templates, current
  published default)
- Generated report: `project/evolve/reports/nat_defs_medium_v4_5.md`

---

## v4.6 — controlled template-variant sweep + verifier (2026-05-22)

Built and shipped `evolve/template_verifier.py` — the template-side
analog of `premise_retriever._UNAVAILABLE_LEMMAS` recommended by v4.5.
It statically filters templates referencing constants in either a
known-unavailable set (`Nat.div_le_div_right`, `Nat.div_le_iff_le_mul`,
`Nat.left_comm`) or a known-type-mismatch set (`Nat.le_refl`,
`Nat.div_le_succ_div`). On the v4.5 div family this drops 4 of 19
templates with zero regression — they never advanced any goal across
v4.1-v4.5.

Added `--template-variant` CLI to `run_evolve.py` with 6 presets:
`v45`, `verified`, `constructor`, `div-rewrite`, `mixed-small`,
`verified-no-rw-eq`. The seed candidate's `theorem_family_tactics["div"]`
and `family_budgets["div"]` are rewritten from the variant spec at
run-evolve startup.

Ran the full sweep on nat_defs_medium:

  | variant            | proved   |   delta vs v4.5 |
  |--------------------|---------:|----------------:|
  | v45 (reference)    | 25 / 38  |              0 |
  | verified           | 25 / 38  |              0 |
  | **constructor**    | **26/38**|             **+1** |
  | div-rewrite        | 25 / 38  |              0 |
  | mixed-small        | 25 / 38  |              0 |
  | verified-no-rw-eq  | 25 / 38  |              0 |

**Constructor variant closed `Nat.div_lt_iff_lt_mul'` — first div-family
closure since v3.6.** The mechanism: removing all rw-style family
tactics lets retrieval emit `rw [Nat.div_lt_iff_lt_mul]` (no prime)
as the first advancing tactic; generative `simp_all` closes step 2.
The hypothesis-confirmation variant (`verified-no-rw-eq`) shows that
dropping `rw [Nat.div_eq_of_lt]` alone is *not* sufficient — multiple
rewrite templates have to be removed for retrieval to get first shot.

The other four variants produce byte-identical metrics (proved=25,
retrieval=354/0/0, mismatch=238). Variant-level changes that don't
affect the first family tactic to fire on any theorem do not affect
the trajectory.

Adopt constructor as the new seed; verifier on by default. v4.7
direction: small evolution sweep on the constructor seed, or term-mode
proof builder for the three div theorems still erroring at step 3-5
(`Nat.div_pos`, `Nat.div_pos_iff`, `Nat.dvd_iff_div_mul_eq`).

- `project/evolve/runs/evolve-20260522-061049-9be813/` — v4.6 v45 baseline
- `project/evolve/runs/evolve-20260522-061553-122264/` — v4.6 verified
- `project/evolve/runs/evolve-20260522-062048-3673c6/` — **v4.6 constructor (26/38)**
- `project/evolve/runs/evolve-20260522-062457-6e9f3e/` — v4.6 div-rewrite
- `project/evolve/runs/evolve-20260522-062940-b62ae9/` — v4.6 mixed-small
- `project/evolve/runs/evolve-20260522-064654-332077/` — v4.6 verified-no-rw-eq
- Generated reports: `project/evolve/reports/nat_defs_medium_v4_6_overnight.md`,
  `project/evolve/reports/v4_6_template_failure_diagnostics.md`
