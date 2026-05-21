# Evaluation report — nat_defs_medium / hybrid_evolved

**Run id**: `eval-11df704d`  
**Metrics**: `/Users/weijiawang/dev/dojo_sandbox/project/evolve/runs/evolve-20260521-184742-70ac3e/eval/seed-baseline/eval-11df704d/metrics.json`  
**Checkpoint**: `project/models/gen_v5`  
**Top-k**: 8, **Max-steps**: 8, **Decode**: beam

## Summary

- **Proved**: **25/38** (65.8%) of 38 theorems  (errored 10, exhausted 3, skipped 0)
- **Wallclock**: 208s (3m 28s), ~5.5s/theorem
- **Denied tactics filtered**: 8 (per-theorem deny-list)
- **Anti-loop**: enabled=False, loops detected=18, skipped repeats=0, unseen advances=24

**Proved by origin**

- `fallback_tactic`: 18
- `family_tactic`: 4
- `generative_topk`: 3

**Family activations**

| Family | Activated on | Wins | Theorems |
|---|---|---|---|
| `AM_GM` | 1 | 0 | `Nat.AM_GM` |
| `div` | 6 | 0 | `Nat.div_le_div_right`, `Nat.div_lt_iff_lt_mul'`, `Nat.div_lt_one_iff`, `Nat.div_pos`, `Nat.div_pos_iff`, `Nat.dvd_iff_div_mul_eq` |
| `mod` | 5 | 4 | `Nat.add_mod_eq_add_mod_left`, `Nat.add_mod_eq_add_mod_right`, `Nat.add_mod_eq_ite`, `Nat.mod_two_ne_one`, `Nat.mod_two_ne_zero` |

## Comparison with baseline

- **Hybrid**:  25/38
- **Baseline**: 3/38
- **Δ = +22** (gains: 22, regressions: 0)

### Gains over baseline (22)

| Theorem | Hybrid origin | Family | Winning tactic |
|---|---|---|---|
| `Nat.add_eq_left` | fallback_tactic |  | `omega` |
| `Nat.add_eq_max_iff` | fallback_tactic |  | `omega` |
| `Nat.add_eq_min_iff` | fallback_tactic |  | `omega` |
| `Nat.add_eq_one_iff` | fallback_tactic |  | `omega` |
| `Nat.add_eq_right` | fallback_tactic |  | `omega` |
| `Nat.add_eq_zero` | fallback_tactic |  | `omega` |
| `Nat.add_mod_eq_add_mod_left` | family_tactic | mod | `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]` |
| `Nat.add_mod_eq_add_mod_right` | family_tactic | mod | `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]` |
| `Nat.add_pos_iff_pos_or_pos` | fallback_tactic |  | `omega` |
| `Nat.eq_zero_of_double_le` | fallback_tactic |  | `omega` |
| `Nat.half_le_of_sub_le_half` | fallback_tactic |  | `omega` |
| `Nat.le_add_one_iff` | fallback_tactic |  | `omega` |
| `Nat.le_and_le_add_one_iff` | fallback_tactic |  | `omega` |
| `Nat.le_one_iff_eq_zero_or_eq_one` | fallback_tactic |  | `omega` |
| `Nat.le_or_le_of_add_eq_add_pred` | fallback_tactic |  | `omega` |
| `Nat.lt_one_iff` | fallback_tactic |  | `omega` |
| `Nat.mod_two_ne_one` | family_tactic | mod | `omega` |
| `Nat.mod_two_ne_zero` | family_tactic | mod | `omega` |
| `Nat.one_add_le_iff` | fallback_tactic |  | `omega` |
| `Nat.sub_lt_iff_lt_add` | fallback_tactic |  | `omega` |
| `Nat.sub_lt_iff_lt_add'` | fallback_tactic |  | `omega` |
| `Nat.two_mul_ne_two_mul_add_one` | fallback_tactic |  | `omega` |

### Regressions

None — every baseline win is also a hybrid win.

### Wins on both sides (3)

- `Nat.lt_iff_add_one_le` (hybrid: `simp_arith`)
- `Nat.pred_eq_of_eq_succ` (hybrid: `simp_all`)
- `Nat.succ_succ_ne_one` (hybrid: `simp [Nat.mul_zero]`)

## Proved theorems

| Theorem | Status | Steps | Origin | Family | Winning Tactic |
|---|---|---|---|---|---|
| `Nat.add_eq_left` | PROVED | 1 | fallback_tactic |  | `omega` |
| `Nat.add_eq_max_iff` | PROVED | 1 | fallback_tactic |  | `omega` |
| `Nat.add_eq_min_iff` | PROVED | 1 | fallback_tactic |  | `omega` |
| `Nat.add_eq_one_iff` | PROVED | 1 | fallback_tactic |  | `omega` |
| `Nat.add_eq_right` | PROVED | 1 | fallback_tactic |  | `omega` |
| `Nat.add_mod_eq_add_mod_left` | PROVED | 1 | family_tactic | mod | `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]` |
| `Nat.add_mod_eq_add_mod_right` | PROVED | 1 | family_tactic | mod | `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]` |
| `Nat.half_le_of_sub_le_half` | PROVED | 2 | fallback_tactic |  | `omega` |
| `Nat.le_and_le_add_one_iff` | PROVED | 1 | fallback_tactic |  | `omega` |
| `Nat.le_or_le_of_add_eq_add_pred` | PROVED | 1 | fallback_tactic |  | `omega` |
| `Nat.add_eq_zero` | PROVED | 1 | fallback_tactic |  | `omega` |
| `Nat.add_pos_iff_pos_or_pos` | PROVED | 2 | fallback_tactic |  | `omega` |
| `Nat.lt_iff_add_one_le` | PROVED | 1 | generative_topk |  | `simp_arith` |
| `Nat.lt_one_iff` | PROVED | 1 | fallback_tactic |  | `omega` |
| `Nat.le_one_iff_eq_zero_or_eq_one` | PROVED | 1 | fallback_tactic |  | `omega` |
| `Nat.le_add_one_iff` | PROVED | 1 | fallback_tactic |  | `omega` |
| `Nat.one_add_le_iff` | PROVED | 1 | fallback_tactic |  | `omega` |
| `Nat.eq_zero_of_double_le` | PROVED | 1 | fallback_tactic |  | `omega` |
| `Nat.sub_lt_iff_lt_add` | PROVED | 1 | fallback_tactic |  | `omega` |
| `Nat.sub_lt_iff_lt_add'` | PROVED | 1 | fallback_tactic |  | `omega` |
| `Nat.mod_two_ne_one` | PROVED | 1 | family_tactic | mod | `omega` |
| `Nat.mod_two_ne_zero` | PROVED | 1 | family_tactic | mod | `omega` |
| `Nat.succ_succ_ne_one` | PROVED | 1 | generative_topk |  | `simp [Nat.mul_zero]` |
| `Nat.pred_eq_of_eq_succ` | PROVED | 1 | generative_topk |  | `simp_all` |
| `Nat.two_mul_ne_two_mul_add_one` | PROVED | 2 | fallback_tactic |  | `omega` |

## Failed theorems

| Theorem | Status | Steps | Origin | Family | Winning Tactic |
|---|---|---|---|---|---|
| `Nat.div_le_div_right` | ERROR | 3 |  |  | `All top-20 tactics errored at step 3` |
| `Nat.div_lt_iff_lt_mul'` | ERROR | 2 |  |  | `All top-20 tactics errored at step 2` |
| `Nat.div_lt_one_iff` | ERROR | 4 |  |  | `All top-20 tactics errored at step 4` |
| `Nat.AM_GM` | ERROR | 1 |  |  | `All top-16 tactics errored at step 1` |
| `Nat.add_mod_eq_ite` | EXHAUSTED | 8 |  |  | `` |
| `Nat.mul_eq_left` | ERROR | 2 |  |  | `All top-18 tactics errored at step 2` |
| `Nat.mul_eq_right` | ERROR | 2 |  |  | `All top-18 tactics errored at step 2` |
| `Nat.eq_one_of_mul_eq_one_left` | ERROR | 1 |  |  | `All top-18 tactics errored at step 1` |
| `Nat.div_pos` | ERROR | 3 |  |  | `All top-20 tactics errored at step 3` |
| `Nat.div_pos_iff` | ERROR | 4 |  |  | `All top-20 tactics errored at step 4` |
| `Nat.sqrt_lt` | EXHAUSTED | 8 |  |  | `` |
| `Nat.pow_lt_pow_iff_left` | EXHAUSTED | 8 |  |  | `` |
| `Nat.dvd_iff_div_mul_eq` | ERROR | 3 |  |  | `All top-20 tactics errored at step 3` |
