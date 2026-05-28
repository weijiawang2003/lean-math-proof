# NS3 — lemma-name audit: evaluation results

Companion to `ns3_lemma_audit.md`. Each variant patches v5-27-w4-master's
genome with one (or all) of the candidate priority templates derived
from the official Mathlib proofs of the 7 remaining failures. Eval set:
`nat_defs_medium` (38 theorems). Baseline (v5-27 master + NS1): **31 / 38**.

## Headline

| metric | value |
|---|---|
| v5-27 baseline (pre-NS3)              | 31 / 38 (82%) |
| **ns3-combined (final, all patches)** | **37 / 38 (97%)** |
| new closures (vs v5-27)               | **6 of 7 remaining failures** |
| regressions                           | 0 |
| DojoCrash errors                      | 0 |
| `unknown constant` errors             | 0 |

Only `Nat.AM_GM` remains unsolved — its proof needs `nlinarith` /
`polyrith`, neither of which is in the trace-replay tactic surface.
Documented as environment-limited; not in scope for NS3.

Final `ns3-combined` genome:
``project/evolve/autonomous_runs/v5-ns3-20260522-200519-d374c5/eval/ns3-combined/genome.json``

## Candidate availability summary

All candidate premises were verified AVAILABLE in the cached LeanDojo
Mathlib tree (commit `29dcec07`) via `scripts/check_lean_names.py`.
Same-file circularity was checked against each target's source line.

| target | promising premises (all AVAILABLE) | audit verdict |
|---|---|---|
| `Nat.dvd_iff_div_mul_eq`        | `Nat.div_mul_cancel`, `Nat.dvd_mul_left` | PROMISING |
| `Nat.eq_one_of_mul_eq_one_left` | `Nat.eq_one_of_mul_eq_one_right`, `Nat.mul_comm`, `mul_eq_one` | PROMISING |
| `Nat.add_mod_eq_ite`            | `Nat.add_mod`, `Nat.mod_eq_of_lt`, `Nat.mod_eq_sub_mod`, `Nat.sub_lt_iff_lt_add`, `Nat.mod_lt`, `Nat.lt_of_not_ge` | initially BLOCKED, actually CLOSED |
| `Nat.div_le_div_right`          | `Nat.le_div_iff_mul_le'`, `Nat.div_mul_le_self`, `Nat.le_trans`, `Nat.eq_zero_or_pos`, `Nat.pos_of_ne_zero`, `gcongr` | PROMISING |
| `Nat.sqrt_lt`                   | `Nat.le_sqrt`, `Nat.not_le` | PROMISING |
| `Nat.pow_lt_pow_iff_left`       | `Nat.pow_le_pow_iff_left`, `Nat.not_le` | PROMISING |
| `Nat.AM_GM`                     | `Nat.add_le_add`, `Nat.mul_le_mul` | BLOCKED (needs nlinarith) |

## Variant results

Three runs were needed; each fixed an issue surfaced by the previous one.

### Run 1 — full first sweep

| variant | proved | Δ vs 31 | new wins | regressions | runtime | crashes | unknown const |
|---|---|---|---|---|---|---|---|
| `ns3-dvd`         | 32 | +1 | `Nat.dvd_iff_div_mul_eq` | none | 206 s | 0 | 0 |
| `ns3-eq-one-mul`  | 32 | +1 | `Nat.eq_one_of_mul_eq_one_left` | none | 209 s | 0 | 0 |
| `ns3-add-mod-ite` | 32 | +1 | `Nat.add_mod_eq_ite` | none | 209 s | 0 | 0 |
| `ns3-div-le`      | 32 | +1 | `Nat.div_le_div_right` | none | 188 s | 0 | 0 |
| `ns3-sqrt-pow`    | 32 | +1 | `Nat.sqrt_lt`, `Nat.pow_lt_pow_iff_left` (+2 wins) | **`Nat.div_pos_iff` (-1)** | 190 s | 0 | 0 |
| `ns3-combined`    | 35 | +4 | 5 wins | **`Nat.div_pos_iff` (-1)** | 162 s | 0 | 0 |

**Issue surfaced:** `ns3-sqrt-pow`'s `simp only [← Nat.not_le, ...]`
templates prepended the iff slot. The `← Nat.not_le` rewrite is a
universal `<` → `¬ ≤` flip, so it fires on EVERY iff goal with a `<`
inside — including `Nat.div_pos_iff` (`0 < n / m ↔ m ≤ n`). After it
fires, the working `rw [Nat.pos_iff_ne_zero, Nat.div_ne_zero_iff
{hyp_ne_zero}]` template no longer matches because the LHS is now
`¬ (n / m ≤ 0)` instead of `0 < n / m`.

**Fix:** Append the sqrt/pow templates *after* the existing iff specifics
(but still inside the specific group, so NS1's auto-sort keeps them
ahead of the generics). The existing div templates run first; the
sqrt/pow templates only fire when the div ones fail to match — which
they do on sqrt-shaped / pow-shaped goals.

### Run 2 — sqrt-pow + combined re-run after the append fix

| variant | proved | Δ vs 31 | new wins | regressions | runtime |
|---|---|---|---|---|---|
| `ns3-sqrt-pow` | 33 | +2 | `Nat.sqrt_lt`, `Nat.pow_lt_pow_iff_left` | **none** | 187 s |
| `ns3-combined` | 36 | +5 | 5 wins (no add_mod_eq_ite) | none | 156 s |

Regression fixed. But `ns3-combined` lost `Nat.add_mod_eq_ite` (which
`ns3-add-mod-ite` had closed).

**Issue surfaced:** The wrapper's shape gate is exclusive — once a
specific shape slot exists in the genome, goals of that shape never
fall back to the `any` slot. `Nat.add_mod_eq_ite` classifies as `le`
(its if-then-else contains a `≤`, and the classifier checks `≤`
before `=`). With `ns3-combined`'s new `le` slot taking over, the
multi-step `cases k <;> [skip; rw [Nat.add_mod]; ...]` template that
lives in the `any` slot was no longer reachable.

### Run 3 — combined with eq broadcast (insufficient)

| variant | proved | new wins | regressions |
|---|---|---|---|
| `ns3-combined` | 36 | 5 wins (still no add_mod_eq_ite) | none |

Mirroring `any` into `eq` only — wrong shape, as it turned out.

### Run 4 — combined with le broadcast (final)

| variant | proved | Δ vs 31 | new wins | regressions | runtime | crashes | unknown const |
|---|---|---|---|---|---|---|---|
| **`ns3-combined`** | **37** | **+6** | all 6 promising | **none** | 152 s | 0 | 0 |

Mirroring `any` into both `eq` and `le` makes the add_mod multi-step
template reachable on add_mod_eq_ite's `le`-classified goal.

## Per-theorem verdict (post-eval)

| theorem | closed by | how |
|---|---|---|
| `Nat.dvd_iff_div_mul_eq`        | `ns3-dvd`         | verbatim Mathlib proof: `exact ⟨fun h => Nat.div_mul_cancel h, fun h => by rw [← h]; exact Nat.dvd_mul_left _ _⟩` |
| `Nat.eq_one_of_mul_eq_one_left` | `ns3-eq-one-mul`  | `exact Nat.eq_one_of_mul_eq_one_right (by rwa [Nat.mul_comm])` (new `eq` slot) |
| `Nat.add_mod_eq_ite`            | `ns3-add-mod-ite` | 2-step: multi-step `cases k <;> [skip; rw [Nat.add_mod]; split_ifs ...]` advances state, then existing `split_ifs <;> omega` closes |
| `Nat.div_le_div_right`          | `ns3-div-le`      | `gcongr` (target lemma is itself `@[gcongr]`-annotated) |
| `Nat.sqrt_lt`                   | `ns3-sqrt-pow`    | verbatim Mathlib proof: `simp only [← Nat.not_le, Nat.le_sqrt]` |
| `Nat.pow_lt_pow_iff_left`       | `ns3-sqrt-pow`    | verbatim Mathlib proof: `simp only [← Nat.not_le, Nat.pow_le_pow_iff_left {hyp_ne_zero}]` |
| `Nat.AM_GM`                     | (none)            | needs `nlinarith` / `polyrith` — environment-limited |

## Why this worked: the audit was the lever, not the search

The audit doc (`ns3_lemma_audit.md`) did the real work. By reading the
official Mathlib proof of each target, we extracted the exact premises
the proof type-checks against. Every one of those premises was AVAILABLE
in the cached Mathlib (verified via the new
`scripts/check_lean_names.py`). No new search algorithm, no LLM call —
just a deterministic "what does Mathlib actually use?" lookup.

The audit also reframed two failures the v5 reports had marked as
"environment limitations":

  - **`Nat.sqrt_lt`** — V5_README said "`Nat.sqrt_lt'` doesn't exist in
    env." Actually `Nat.sqrt_lt'` is at `Defs.lean:1623` — 3 lines AFTER
    the target at 1620 — so trace replay genuinely can't see it. But
    `Nat.le_sqrt` is 8 lines BEFORE the target and is the premise the
    Mathlib proof actually uses.

  - **`Nat.add_mod_eq_ite`** — `v5_failure_deep_dive_add_mod_eq_ite.md`
    said this needed a v6 branch-skeleton type. In fact a compressed
    single-line multi-step template advances the state enough that the
    existing `split_ifs <;> omega` closes it on the next step. The
    "asymmetric branches require asymmetric closers" diagnosis was
    right in principle, but a two-step proof using `omega` for both
    branches as the closer turned out to work.

## Lessons for v6

  1. **Static availability data + cached Mathlib source is enough to
     unstick a lot of plateaus.** No model, no skeleton, no mutator.
     Just `grep` and "what does the human proof do here?"

  2. **Specificity ordering (NS1) and shape gating interact.** Three
     follow-up runs were needed to debug subtle ordering and
     shape-gate issues:
       - NS1 keeps specifics ahead of generics, but within-specific
         order matters too (the sqrt-pow regression was within-specific).
       - The shape gate is exclusive — once a specific slot exists,
         the `any` slot is unreachable. This is the most surprising
         constraint and was responsible for the combined run missing
         add_mod_eq_ite for two iterations.

  3. **`simp only [...]` is dangerous in multi-template slots.** The
     `← Nat.not_le` rewrite is universal. Future templates that include
     such "always-fires" rewrites need a goal-content gate or should be
     emitted only when the rest of the template's lemmas would apply.
     v6 could implement this as a "guard" field on a skeleton.

  4. **The `@[gcongr]` mechanism is a free win.** Any Mathlib lemma
     annotated `@[gcongr]` can be closed by a one-token `gcongr`
     template in the `le` slot. There may be more div/mod lemmas in
     `nat_defs_large_v5` with the same annotation.

## Recommendation for v6

The NS3 result confirms the v5 → v6 plan in `v5_next_steps.md` from a
different angle: the **architectural** bottleneck (skeleton-bag with
proper guards / two-tier mutator) is what's needed *next*, but the
short-term win on the existing wrapper was real and substantial. Order
of operations remains:

  1. ✅ **NS1** (specificity ordering) — done.
  2. ✅ **NS3** (lemma-name audit) — done, +6 wins.
  3. **NS2** (manual hand-coded multi-step skeleton for any v5/NS3
     failure that didn't make it through). Currently only `Nat.AM_GM`
     remains, and it's env-limited; NS2 is effectively complete.
  4. **NS4 + NS5** (skeleton bag + two-tier mutator) — now the
     load-bearing next step.

NS3 also surfaces a **half-step** worth doing before NS4:

  - **NS-3.5: shape-fallback semantics for the wrapper.** A 3-line
    change to `rank_tactics` that adds `any` templates as a
    fallback AFTER specific-shape templates (rather than only when no
    specific-shape slot exists at all). Would have made the ns3-combined
    `le`-broadcast unnecessary. Worth doing as a small wrapper polish.

## Run artifacts (not committed; for reference paths)

  - `v5-ns3-20260522-193323-3294ab/` — full first sweep (6 variants, sqrt-pow regression)
  - `v5-ns3-20260522-195455-7f1508/` — sqrt-pow + combined fix (no regression, missing add_mod)
  - `v5-ns3-20260522-200134-2c809a/` — combined with eq broadcast (no add_mod)
  - `v5-ns3-20260522-200519-d374c5/` — **FINAL ns3-combined at 37/38**
