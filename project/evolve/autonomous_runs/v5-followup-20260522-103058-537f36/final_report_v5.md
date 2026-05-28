# v5 followup-loop final report

**Run id:** `v5-followup-20260522-103058-537f36`
**Theorem set:** `nat_defs_medium`
**Mechanism under test:** `priority_templates` (new genome slot
emitting templates BEFORE generative_topk).

## Result

Best v5 candidate: **v5-18-prio-kitchen** at **29 / 38**.

| baseline | best v5 |     Δ     |
|----------|---------|-----------|
| 26 / 38  | 29 / 38 | **+3**    |

Three new theorems closed, all via the priority_templates slot.
Each new closure required a different priority template:

  1. **`Nat.div_lt_one_iff`** — `rw [Nat.div_lt_iff_lt_mul hb, Nat.one_mul]`
  2. **`Nat.mul_eq_left`**    — `exact ⟨Nat.eq_of_mul_eq_mul_left ..., simp [h]⟩`
  3. **`Nat.mul_eq_right`**   — `exact ⟨Nat.eq_of_mul_eq_mul_right ..., simp [h]⟩`

## Why these and not the other 9 failing theorems

| theorem | v5 attempt | outcome |
|---|---|---|
| `Nat.div_lt_one_iff`   | priority rw with `{hyp_pos}` and `Nat.one_mul` | **closed** |
| `Nat.mul_eq_left`      | priority term-mode with `Nat.eq_of_mul_eq_mul_left` | **closed** |
| `Nat.mul_eq_right`     | priority term-mode with `Nat.eq_of_mul_eq_mul_right` | **closed** |
| `Nat.div_le_div_right` | no template tried — `le` shape not in priority dict | unsolved |
| `Nat.AM_GM`            | no template — `∀ {a b}` needs intros first | unsolved |
| `Nat.add_mod_eq_ite`   | split_ifs advanced once but no closing inner tactic | unsolved |
| `Nat.eq_one_of_mul_eq_one_left` | `eq` shape, no template covers | unsolved |
| `Nat.div_pos`          | priority `lt` shape but lemma chain not in env  | unsolved |
| `Nat.div_pos_iff`      | priority iff fired but lemmas mismatch  | unsolved |
| `Nat.sqrt_lt`          | `Nat.sqrt_lt'` doesn't exist in this Lean      | unsolved |
| `Nat.pow_lt_pow_iff_left` | self-reference; no alt lemma | unsolved |
| `Nat.dvd_iff_div_mul_eq`  | dvd template tried but wrong shape detection | unsolved |

The unsolved 9 break into three classes:

  - **Needs a new tactic the env doesn't have** (Nat.AM_GM, sqrt forms,
    nlinarith-class). No code change can fix.
  - **Needs the right Mathlib lemma name in the priority list** (div_pos
    chain, dvd witness, pow_lt_iff non-prime form). Tractable with
    wave 4 / wave 5.
  - **Needs an inner-tactic stronger than `omega`/`simp_all` after the
    split** (add_mod_eq_ite after split_ifs). Tractable with smarter
    skeleton-aware split.

## Time budget

> _Total followup runtime: TBD when all 11 cycles complete._

## Lesson

The `priority_templates` slot is the single change that broke the
26/38 plateau. The lesson generalizes: in a ranked-list wrapper
architecture, the *position* of a tactic in the ranked list is as
important as its *contents*. Empirically, generative_topk's
"weak-simp advances state but doesn't close" behavior shadowed
every downstream specifically-crafted template until priority_templates
moved a few hand-curated templates ahead of the model output.

For v6: the architecture should make this finding default. Either
templates are skeleton-bag entries that emit BEFORE the model, or
the model's output is gated by an "is_progress_toward_closure"
filter that suppresses weak-simp advances when a more specific
template exists.
