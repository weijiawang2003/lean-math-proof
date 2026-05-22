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
