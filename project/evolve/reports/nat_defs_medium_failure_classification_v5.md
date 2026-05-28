# nat_defs_medium — v5 failure classification

Source: v4.7 constructor-default seed baseline,
`project/evolve/runs/evolve-20260522-072211-b7f1fc/eval/seed-baseline/eval-430bc699/metrics.json`.

26 / 38 proved. The 12 unsolved theorems are classified below by the kind of
proof structure they need, not by what the wrapper happens to try.

## Per-theorem breakdown

| # | theorem                           | shape    | status | reason wrapper fails                                                                 | what would close it (sketch)                                       |
|---|-----------------------------------|----------|--------|--------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| 1 | `Nat.div_le_div_right`            | le       | EXH/8  | needs lemma not in retrieval bucket; `Nat.div_le_div_right` itself is the target     | `Nat.div_le_div_left h _ _` or `Nat.div_le_div_iff_right`         |
| 2 | `Nat.div_lt_one_iff`              | iff      | EXH/8  | `rw [Nat.div_lt_iff_lt_mul]` fails without `hb`; bare form unifies but pattern wrong | `rw [Nat.div_lt_iff_lt_mul hb]; simp` — **constructor variant lost the {hyp_pos} template** |
| 3 | `Nat.AM_GM`                       | unknown  | ERR/1  | goal is `∀ {a b}`; first tactic must `intro`; without `nlinarith` the rest fails     | `intro a b; nlinarith [sq_nonneg (a-b)]` — `nlinarith` unavailable |
| 4 | `Nat.add_mod_eq_ite`              | unknown  | EXH/8  | ite head; deny-listed simp_all crashes Dojo; nothing splits the if-then-else        | `split_ifs <;> omega` — `split_ifs` not tried                      |
| 5 | `Nat.mul_eq_left`                 | iff      | ERR/2  | `a*b = a ↔ b = 1` (`ha : a ≠ 0`); no `mul` family; iff split fails on forward         | `Nat.eq_one_of_mul_eq_self_right ha` / `Nat.mul_left_cancel`       |
| 6 | `Nat.mul_eq_right`                | iff      | ERR/2  | symmetric to above                                                                   | `Nat.eq_one_of_mul_eq_self_left hb`                                |
| 7 | `Nat.eq_one_of_mul_eq_one_left`   | unknown  | ERR/1  | `m * n = 1 → n = 1`; needs case analysis on m, n                                     | `match m, n, H with | 1, 1, _ => rfl | …`                          |
| 8 | `Nat.div_pos`                     | lt       | ERR/5  | `0 < a/b` with `b ≤ a, 0 < b`; needs `Nat.div_pos hba hb` (self) or compose          | `(Nat.one_le_div_iff_le hb).mpr hba`                               |
| 9 | `Nat.div_pos_iff`                 | iff      | ERR/4  | both directions need div-arithmetic reasoning                                        | `constructor <;> intro h; · by_contra hc; simp [Nat.div_eq_of_lt …] at h; · …` |
| 10 | `Nat.sqrt_lt`                     | unknown  | EXH/8  | iff with sqrt; no sqrt lemmas in retrieval                                          | `Nat.sqrt_lt'` or rewrite via `Nat.lt_succ_sqrt`                   |
| 11 | `Nat.pow_lt_pow_iff_left`         | unknown  | EXH/8  | `a^n < b^n ↔ a < b` (`hn : n ≠ 0`); needs `Nat.pow_lt_pow_iff_*`                     | `Nat.pow_lt_pow_iff_left hn`                                       |
| 12 | `Nat.dvd_iff_div_mul_eq`          | iff      | ERR/3  | dvd witness; iff between `d ∣ n` and `n/d * d = n`                                   | `constructor; · intro ⟨k, hk⟩; ...; · intro h; exact ⟨n/d, h.symm⟩`|

## Buckets

### Bucket I — "missing-template" (tractable tonight)
Templates the seed could ship but doesn't, that would close the goal in 1-2 steps.

  - **#2 `Nat.div_lt_one_iff`** — `rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]` exists in v45 variant but is **missing from constructor variant**. Restore it. *Highest-confidence win.*
  - **#11 `Nat.pow_lt_pow_iff_left`** — `rw [Nat.pow_lt_pow_iff_left {hyp_ne_zero}]` (or simp form). Retrieval doesn't catalog pow.

### Bucket II — "term-mode-or-shape-mini-solver" (Direction A/B)
Iff goals where the two directions need *different* inner tactics, so `<;>` is too coarse.

  - **#5 `Nat.mul_eq_left`** and **#6 `Nat.mul_eq_right`** — need asymmetric iff split. Forward direction uses mul-cancellation; backward is `subst`.
  - **#9 `Nat.div_pos_iff`** — same, both directions different.
  - **#12 `Nat.dvd_iff_div_mul_eq`** — also asymmetric (one direction unpacks `∃`, the other constructs it).

### Bucket III — "needs new tactic" (intractable without code)
Tactics that don't exist in this Lean env or require deeper reasoning.

  - **#3 `Nat.AM_GM`** — `nlinarith [sq_nonneg (a-b)]` is the canonical close; `nlinarith` reports `unknown tactic` per v3.4 trace census. No code-only workaround.
  - **#4 `Nat.add_mod_eq_ite`** — `split_ifs <;> omega` likely works but `split_ifs` not tried; **adding it is a one-line genome change**.

### Bucket IV — "induction-or-cases" (Direction C-ish)
Theorems where the missing primitive is small-case enumeration.

  - **#7 `Nat.eq_one_of_mul_eq_one_left`** — needs `cases m; cases n` or `match` term-mode. Tractable with `rcases` + omega-style closure.
  - **#1 `Nat.div_le_div_right`** — needs the right lemma. Retrieval bucket may not contain it. Add it.

## Implications for v5 directions

1. **First win attempt** — restore the v45 `{hyp_pos}` div templates to the constructor variant. Expect 1-2 wins (#2, possibly #8).
2. **Direction A (term_builder)** — focus on iff-split with asymmetric inner tactics. Target #5, #6, #9, #12.
3. **Direction B (mini-solvers)** — add `mul` family (covers #5, #6, possibly #7) and `pow` family (#11). Add a `split_ifs` template (#4).
4. **Direction C (skeleton mutation)** — given a candidate proof skeleton, mutate the inner tactic positions (omega ↔ simp_all ↔ rfl ↔ trivial) and the hypothesis substitutions.

Tracking metric: **proved_count out of 38** plus **buckets-cleared count out of bucket sizes** (I: 2, II: 4, III: 2, IV: 2). Negative-result documentation is acceptable per task spec.

## Wave-by-wave outcomes (filled after run)

### v5 first-pass (12 variants, Directions A/B/C without priority_templates)
**No new wins.** Confirmed v3 → v4 plateau at 26/38. Twelve different
ways of adding templates, families, term-mode skeletons, and inner-tactic
mutations all stalled at the same number because the wrapper's
generative-first ordering shadowed every new template.

### v5 followup-pass (11 variants, with `priority_templates`)
**+3 wins, best variant 29/38.**
  - Bucket I closed (`Nat.div_lt_one_iff`) — v5-12 priority div-hyp-pos.
  - Bucket II partially closed (`Nat.mul_eq_left`, `Nat.mul_eq_right`)
    — v5-15 priority mul-specific term-mode skeleton.
  - Surprise: Bucket III partially closed
    (`Nat.div_pos`, `Nat.div_pos_iff`) via v5-20 — turned out to be
    Bucket I after all (just needed the right Mathlib lemma).

### Remaining failures (7 of original 12)
  - `Nat.div_le_div_right` (le) — no working template in v5
  - `Nat.AM_GM` (∀-quantified) — needs `nlinarith`-class tactic absent in env
  - `Nat.add_mod_eq_ite` — `split_ifs` advances but no inner closer
  - `Nat.eq_one_of_mul_eq_one_left` (eq) — needs case analysis
  - `Nat.sqrt_lt`, `Nat.pow_lt_pow_iff_left` — env lemmas don't exist or self-ref
  - `Nat.dvd_iff_div_mul_eq` — asymmetric dvd-iff, no template tried yet

### Wave 4 (in progress)
Targeted variants for the 7 remaining failures. v5-28 super-kitchen
unions every confirmed-working priority template — expected **31/38**.
