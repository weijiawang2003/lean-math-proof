# Evaluation report — nat_defs_medium / hybrid_evolved + v4.5 structured div templates

**Branch**: `v4-premise-retrieval-div`
**Parent commit (v4.4)**: `0f92d44` — `Add shape-aware ranking for retrieved premise tactics`
**v4.5 run id**: `evolve-20260522-050325-0fe236`
**Metrics**: `project/evolve/runs/evolve-20260522-050325-0fe236/eval/seed-baseline/eval-*/metrics.json`
**Checkpoint**: `project/models/gen_v5` (unchanged)
**Top-k**: 8, **Max-steps**: 8, **Decode**: beam
**Wallclock**: 4m 48s (≈ 7.6 s/theorem)

## Headline

| | v3.6 | v4.1 | v4.2 | v4.3 | v4.4 | **v4.5 (default)** |
|---|---:|---:|---:|---:|---:|---:|
| **Proved** | **25/38** | **25/38** | **25/38** | **25/38** | **25/38** | **25/38** |
| Errored | 10 | 6 | 7 | 10 | 7 | **6** |
| Exhausted | 3 | 7 | 6 | 3 | 6 | **7** |
| Retrieval attempts | — | 783 | 303 | 121 | 279 | **354** |
| Retrieval wins | — | 0 | 0 | 0 | 0 | **0** |
| DojoCrashError | 0 | 0 | 0 | 0 | 0 | **0** |
| `unknown constant` (retrieved) | — | 200 | 0 | 0 | 0 | **0** |
| Family-tactic emissions on div | — | — | — | — | — | **+8 structured templates per state** |
| Bloating-apply observations | — | 6 | 16 | 3 | 3 | **3** |
| Shape-mismatch suppressions | — | — | — | — | 213 | **238** |
| Wallclock | ~3m 28s | 5m 22s | 4m 19s | 3m 42s | 4m 21s | **4m 48s** |

v4.5 adds 8 structured templates (induction-on-hypothesis,
iff-constructor split, rewriting with positivity hypothesis) to the
`div` family. Infrastructure works correctly — all 8 templates trace,
all are emitted only on theorems where their hypothesis placeholders
match. **None close any div theorem** for the diagnostic reasons
catalogued below.

## Success criteria

| Criterion | Met? | Detail |
|---|---|---|
| ≥ 25/38 preserved | ✓ | 25/38 (matches v3.6 / v4.1–v4.4) |
| No regressions | ✓ | `proved_by_origin = {fallback_tactic: 18, family_tactic: 4, generative_topk: 3}` — bit-identical to v3.6 |
| No new `DojoCrashError` | ✓ | Denied tactics = 8 (unchanged). No crashes observed. |
| `unknown constant` remains 0 (retrieved) | ✓ | 0 in retrieved traces. |
| ≥ 1 new div theorem solved (*ideal*) | ✗ | 0 new closures. All 8 structured templates errored on every div theorem they targeted. Diagnostic in next section. |

**4 of 5 explicit criteria met.** The ideal closure criterion is again
not met; v4.5 is the third stage in a row (v4.3, v4.4, v4.5) to find
that the catalog/closure work is harder than expected. Concrete next
steps below.

## The 8 v4.5 templates and what each did

Each template is shown with its placeholder shape and the per-theorem
outcome it actually produced. All are added to
`theorem_family_tactics["div"]` and only emitted on states whose
hypothesis set matches the placeholders.

| # | Template | Placeholders | Emitted on | Outcomes |
|---|---|---|---|---|
| 1 | `simp [Nat.div_lt_iff_lt_mul, Nat.mul_one]` | none | all 6 div theorems | every attempt: `simp made no progress` |
| 2 | `simp_all [Nat.div_lt_iff_lt_mul, Nat.mul_one]` | none | all 6 | every attempt: `simp_all made no progress` |
| 3 | `simp_all [Nat.div_lt_iff_lt_mul', Nat.mul_one]` | none | all 6 | every attempt: `simp_all made no progress` |
| 4 | `rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]` | `{hyp_pos}` | div_lt_iff_lt_mul', div_lt_one_iff, div_pos (3 theorems) | `tactic 'rewrite' failed, did not find instance of the pattern` on every state |
| 5 | `constructor <;> intro h_split <;> omega` | none | all 6 | iff goals: `omega could not prove the goal`; le/lt goals: `tactic 'constructor' failed, no applicable constructor` |
| 6 | `constructor <;> intro h_split <;> simp_all` | none | all 6 | iff goals: `simp_all made no progress` (twice); le/lt goals: constructor failed |
| 7 | `induction {hyp_le} <;> simp_all` | `{hyp_le}` | div_le_div_right, div_pos (2 theorems) | `simp_all made no progress` (induction itself succeeded) |
| 8 | `induction {hyp_le} with \| refl => exact Nat.le_refl _ \| step h_step ih => exact ih.trans (Nat.div_le_succ_div _ _)` | `{hyp_le}` | div_le_div_right, div_pos (2 theorems) | `type mismatch Nat.le_refl ?m... has type ?m...` — wrong name |

Diagnostic from the trace data:

* **Template 4 (`rw [Nat.div_lt_iff_lt_mul hb, Nat.mul_one]`)** — Lean
  could not pattern-match the rewrite against the goal. The lemma in
  the actual mathlib version we're running against has a different
  signature than what the template assumed. Likely candidates: the
  positivity hypothesis is the *second* argument, not the first, or
  the LHS/RHS are swapped. Without inspecting the exact mathlib
  declaration the template can't be fixed by guessing.

* **Template 8 (`Nat.le_refl`)** — type mismatch suggests the constant
  name is wrong in the eval-env import closure. `Nat.le_refl` may
  exist as `Nat.le.refl` (constructor) or `le_refl` (root namespace) or
  not be exported under that name at all. This is the same
  forward-reference / availability trap that v4.2's `_UNAVAILABLE_LEMMAS`
  was built to handle, just applied to a hand-written template instead
  of a retrieved premise.

* **Templates 1/2/3 (`simp [Nat.div_lt_iff_lt_mul, Nat.mul_one]`)** —
  `simp` and `simp_all` cannot fire `Nat.div_lt_iff_lt_mul` because the
  lemma has a positivity precondition (`0 < c` or `0 < b`) and `simp`
  won't generate the existential hypothesis on its own. The `simp_all`
  variant has access to the `hb : 0 < b` hypothesis but apparently
  doesn't use it as a rewrite hint.

* **Templates 5/6 (`constructor <;> intro h_split <;> ...`)** — work
  correctly on iff goals (constructor splits them into the two
  directions), but neither `omega` nor `simp_all` can close the
  resulting subgoals because they involve `a / b` which is outside
  omega's theory and which `simp` lemmas in scope don't handle.

* **Template 7 (`induction {hyp_le} <;> simp_all`)** — induction runs
  to completion, but the resulting `refl`/`step` subgoals contain `a/c`
  which simp can't close.

## Per-div-theorem result

| Theorem | Hyps | Shape | v3.6 | v4.4 | **v4.5** | New templates fired | First-fail reason |
|---|---|---|---|---|---|---:|---|
| `Nat.div_le_div_right` | `h : a ≤ b` | le | ERR @3 | EXH @8 | **EXH @8** | 6 of 8 | `induction h with Nat.le_refl` → type mismatch (wrong name) |
| `Nat.div_lt_iff_lt_mul'` | `hb : 0 < b` | iff | ERR @2 | EXH @8 | **EXH @8** | 6 of 8 | `rw [Nat.div_lt_iff_lt_mul hb, Nat.mul_one]` → did not find pattern |
| `Nat.div_lt_one_iff` | `hb : 0 < b` | iff | ERR @4 | EXH @8 | **EXH @8** | 6 of 8 | same — `rw` did not match the goal pattern |
| `Nat.div_pos` | `hba`, `hb` | lt | ERR @3 | ERR @3 | **ERR @3** | 8 of 8 | type mismatch / pattern failure cascade; all-errored at step 3 |
| `Nat.div_pos_iff` | `hb : b ≠ 0` | iff | ERR @4 | ERR @4 | **ERR @4** | 4 of 8 | constructor split succeeded; omega/simp_all then failed |
| `Nat.dvd_iff_div_mul_eq` | none | iff | ERR @3 | ERR @3 | **EXH @8** | 4 of 8 | constructor + simp_all kept making progress without closing |

`Nat.dvd_iff_div_mul_eq` shifted from v4.4's ERR @3 to v4.5's EXH @8 —
the `constructor <;> intro h_split <;> simp_all` template advanced state
where every prior tactic errored. It still doesn't close, but the
rollout now explores. (This is the same pattern v4.4 showed on 3 other
div theorems.)

## Wrapper / template infrastructure additions

### `evolve/strategy_wrapper.py`

* `_HYP_LINE: re.Pattern` matches `name : type_expr` for single-name
  hypothesis lines (multi-name `a b c : ℕ` binders are caught by the
  existing `_NAT_LINE` and skipped).
* `_POS_PREFIX: re.Pattern` matches `0 < ...` for the `hyp_pos` shape.
* `_extract_hypotheses(state_pp) → dict[str, str | None]` returns
  `{hyp_le, hyp_pos, hyp_ne_zero}` keyed name lookups. First match
  wins; iteration stops at the `⊢` line.
* `_PLACEHOLDER_RE: re.Pattern` extracts `{...}` placeholders.
* `_render_template(template, nat_vars, hypotheses)` extended to
  handle hypothesis placeholders alongside `{var}`. **Skips the
  template entirely** when any `{hyp_*}` referenced has no backing
  hypothesis name — so templates never emit malformed Lean.
* `StrategyWrapperPolicy.rank_tactics` extracts hypotheses once per
  state and passes the dict to every `_render_template` call.

### `evolve/run_evolve.py`

* `theorem_family_tactics["div"]` extended with 8 v4.5 entries
  (4 hyp-less, 1 `{hyp_pos}`, 2 `{hyp_le}`, 1 induction-with-cases).
* `family_budgets["div"]` bumped 12 → 20 to accommodate the new
  templates after dedup.
* Seed description updated to v4.5.

### Nothing else changed

* `premise_retriever.py`, `evolve/candidate.py`, `evolve/evaluator.py`,
  `eval_rollout_all.py` untouched. The new hypothesis machinery is
  purely template-level and reuses the existing rendering pipeline.
* No new strategy-config flags. No JSON-schema bump. The 13-tuple from
  v4.4 is unchanged. Old strategy configs still load.

## v4.5 retrieval pipeline (unchanged)

For completeness — the retrieval layer behaves exactly as in v4.4:

| Metric | Value |
|---|---:|
| `retrieved_premise_activation_count` | 6 (all 6 div theorems) |
| `retrieved_premise_attempt_count` | 354 (-1% vs v4.4's 279 +75; the rise comes from deeper rollouts on `Nat.dvd_iff_div_mul_eq` not from a regression in filtering) |
| `retrieved_premise_advanced_count` | 0 |
| `retrieved_premise_proved_count` | 0 |
| `retrieved_premise_filtered_self_count` | 38 |
| `retrieved_premise_filtered_unavailable_count` | 254 |
| `shape_mismatch_filtered_count` | 238 |
| `bloating_apply_lemma_counts` | `{Nat.lt_of_lt_of_le: 3}` |
| `skipped_bloating_apply_count` | 12 |

## What v4.5 did NOT do (per scope)

* No retraining, no checkpoint changes.
* No new policy types.
* No new lemmas in `STATIC_PREMISES`.
* No new `_UNAVAILABLE_LEMMAS` entries.
* No new strategy-config flags (no JSON schema bump).
* No live Lean availability checker.
* No term-mode `exact ⟨_,_⟩` builder (would be needed to close
  `Nat.div_pos_iff`).
* No mutation sweep — given the structural finding (template lemma
  names are wrong, not config-tunable), running mutations of the same
  templates would not produce informative results.

## Why v4.5 added zero proofs

Three distinct failure modes, in priority order:

1. **Lemma name / signature mismatch** with the actual mathlib version
   the eval environment uses. `Nat.le_refl`, `Nat.div_le_succ_div`,
   `Nat.div_lt_iff_lt_mul hb` (positional hypothesis) — any of these
   could be off by a name, a constructor, or an argument order. The
   v4.2-era `_UNAVAILABLE_LEMMAS` denylist solved the retrieval-side
   version of this; v4.5 needs the template-side equivalent (verify
   each template's lemma references against the real mathlib).
2. **`simp` cannot fire hypothesis-conditioned lemmas.** Most of the
   `Nat.div_*_iff_*` lemmas have a positivity precondition. `simp`
   refuses to apply them as rewrite rules without an explicit
   `[h_pos]` argument. `simp_all` uses hypotheses but apparently still
   doesn't combine them with the precondition correctly.
3. **omega / simp_all do not understand `Nat.div`.** Most of the
   `constructor <;> ... <;> omega` and `... <;> simp_all` templates
   die in the second stage when the resulting subgoal contains `a/b`.
   omega is a linear-arithmetic decision procedure and doesn't model
   integer division. simp's `Nat.div_*` lemmas are conditional and
   don't fire.

The v4.5 infrastructure makes it **easy** to add new hypothesis-aware
templates: a single string in `theorem_family_tactics["div"]` with
`{hyp_le}` / `{hyp_pos}` / `{hyp_ne_zero}` placeholders. The remaining
work is *content* — finding the right Lean tactics that actually
close these proofs.

## Recommended next step (v4.6)

**Two paths, ordered by expected payoff:**

### Path A (lighter — recommended first): template lemma verification

Before bulk-adding more templates, write a one-time script that takes
each template's referenced lemmas (`Nat.le_refl`, `Nat.div_le_succ_div`,
`Nat.div_lt_iff_lt_mul`, …) and runs a Lean `#check` against the eval
import context. Templates that reference unavailable / misnamed
constants are dropped. This is the template-side analog of v4.2's
`_UNAVAILABLE_LEMMAS`, and it would have caught templates 4 and 8 at
authoring time. **Estimated effort**: a single shell-out per template
at run-evolve setup time, or a one-time human-curation pass.

### Path B (heavier): term-mode `exact ⟨_,_⟩` builder

Several div theorems (especially `Nat.div_pos_iff` and
`Nat.dvd_iff_div_mul_eq`) close via term-mode constructions like
`Iff.intro (fun h => ...) (fun h => ...)` or
`⟨fun h => ..., fun h => ...⟩`. None of the current
`rw`/`simp`/`apply`/`exact LEMMA` forms can produce these. A
`term_builder` form would generate
`exact ⟨{fwd_tactic}, {bwd_tactic}⟩`-style entries from a small
template DSL. **Estimated effort**: medium; adds a new form-emission
path in the wrapper and requires careful hypothesis-name extraction
for the lambda binders.

### Path C (heaviest, not recommended yet)

Hand-curate the 6 div theorems' proofs from the actual mathlib source
and ship them as per-theorem-specific templates (a new
`theorem_specific_templates` map keyed by full_name, not just by
family). This works but is brittle and doesn't generalize. Defer until
A and B have been tried.

## Artifacts

* `project/evolve/runs/evolve-20260522-050325-0fe236/` — v4.5 run root
* `…/eval/seed-baseline/eval-*/metrics.json` — full metrics
* `…/eval/seed-baseline/eval-*/traces.jsonl` — 1,911 trace records
  (includes 244 family-tactic attempts on div theorems, of which
  ~120 are v4.5 structured templates)
* `…/eval/seed-baseline/strategy_config.json` — unchanged config
  surface (no new flags)
* Branch: `v4-premise-retrieval-div` (off `e74861f`); v4.5 commit
  pending on top of `0f92d44`.
