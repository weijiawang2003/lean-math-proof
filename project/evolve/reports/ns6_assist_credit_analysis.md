# NS6 — assist-credit analysis

Per-skeleton credit accounting over the per-step traces below.
`direct_wins` = closed the proof; `advances` = produced a new
state without closing; `assist_wins_kN` = advanced, and within
the next N accepted proof steps a *different* tactic closed the
proof.

## Sources

- `project/evolve/ns6_runs/baseline/medium/eval-85ba1889/traces.jsonl`
- `project/evolve/ns6_runs/baseline/large/eval-079b6e55/traces.jsonl`

- skeletons observed: **52**
- total direct wins: **79**
- total advances: **15**
- total assist@k3: **4**

## Per-skeleton credit table

| skeleton | shape | family | origin | attempts | direct_wins | advances | assist@1 | assist@2 | assist@3 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| pt_iff_8 | iff | - | priority_template | 62 | 40 | 0 | 0 | 0 | 0 |
| fb_22 | any | - | fallback_tactic | 22 | 13 | 0 | 0 | 0 | 0 |
| fam_mod_19 | any | mod | family_tactic | 5 | 4 | 0 | 0 | 0 | 0 |
| pt_iff_6 | iff | - | priority_template | 74 | 2 | 4 | 0 | 0 | 0 |
| pt_iff_7 | iff | - | priority_template | 12 | 2 | 4 | 0 | 0 | 0 |
| pt_any_12 | any | - | priority_template | 66 | 2 | 0 | 0 | 0 | 0 |
| pt_eq_14 | eq | - | priority_template | 24 | 2 | 0 | 0 | 0 | 0 |
| pt_iff_0 | iff | - | priority_template | 84 | 2 | 0 | 0 | 0 | 0 |
| pt_iff_1 | iff | - | priority_template | 6 | 2 | 0 | 0 | 0 | 0 |
| pt_iff_3 | iff | - | priority_template | 18 | 2 | 0 | 0 | 0 | 0 |
| pt_iff_4 | iff | - | priority_template | 16 | 2 | 0 | 0 | 0 | 0 |
| pt_iff_5 | iff | - | priority_template | 14 | 2 | 0 | 0 | 0 | 0 |
| pt_le_17 | le | - | priority_template | 7 | 2 | 0 | 0 | 0 | 0 |
| pt_lt_11 | lt | - | priority_template | 2 | 2 | 0 | 0 | 0 | 0 |
| pt_any_13 | any | - | priority_template | 68 | 0 | 2 | 2 | 2 | 2 |
| retrieved:Nat.div_lt_iff_lt_mul:rw | iff | div | retrieved_premise | 8 | 0 | 2 | 2 | 2 | 2 |
| retrieved:Nat.div_eq_of_lt:rw | iff | div | retrieved_premise | 7 | 0 | 3 | 0 | 0 | 0 |
| fam_div_18 | any | div | family_tactic | 9 | 0 | 0 | 0 | 0 | 0 |
| fam_mod_20 | any | mod | family_tactic | 1 | 0 | 0 | 0 | 0 | 0 |
| fam_mod_21 | any | mod | family_tactic | 1 | 0 | 0 | 0 | 0 | 0 |
| fb_23 | any | - | fallback_tactic | 14 | 0 | 0 | 0 | 0 | 0 |
| fb_24 | any | - | fallback_tactic | 13 | 0 | 0 | 0 | 0 | 0 |
| pt_eq_15 | eq | - | priority_template | 22 | 0 | 0 | 0 | 0 | 0 |
| pt_iff_2 | iff | - | priority_template | 4 | 0 | 0 | 0 | 0 | 0 |
| pt_le_16 | le | - | priority_template | 14 | 0 | 0 | 0 | 0 | 0 |
| pt_lt_10 | lt | - | priority_template | 5 | 0 | 0 | 0 | 0 | 0 |
| pt_lt_9 | lt | - | priority_template | 5 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.div_eq_of_lt:apply | eq | div | retrieved_premise | 2 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.div_eq_of_lt:simp | lt | div | retrieved_premise | 4 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.div_lt_iff_lt_mul':apply | iff | div | retrieved_premise | 2 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.div_lt_iff_lt_mul':rw | iff | div | retrieved_premise | 6 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.div_lt_iff_lt_mul':simp | iff | div | retrieved_premise | 6 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.div_lt_iff_lt_mul:apply | iff | div | retrieved_premise | 2 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.div_lt_iff_lt_mul:simp | iff | div | retrieved_premise | 6 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.div_mul_cancel:apply | eq | div | retrieved_premise | 2 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.div_mul_cancel:rw | lt | div | retrieved_premise | 4 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.div_mul_cancel:simp | lt | div | retrieved_premise | 4 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.lt_of_lt_of_le:apply | lt | div | retrieved_premise | 2 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.lt_of_lt_of_le:rw | lt | div | retrieved_premise | 4 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.lt_of_lt_of_le:simp | lt | div | retrieved_premise | 4 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.mod_add_div:apply | lt | div | retrieved_premise | 4 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.mod_add_div:rw | lt | div | retrieved_premise | 4 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.mod_add_div:simp | lt | div | retrieved_premise | 4 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.mod_eq_of_lt:apply | eq | div | retrieved_premise | 2 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.mod_eq_of_lt:rw | eq | div | retrieved_premise | 2 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.mod_eq_of_lt:simp | eq | div | retrieved_premise | 2 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.mul_div_cancel:apply | eq | div | retrieved_premise | 2 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.mul_div_cancel:rw | lt | div | retrieved_premise | 4 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.mul_div_cancel:simp | lt | div | retrieved_premise | 4 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.pos_of_ne_zero:apply | lt | div | retrieved_premise | 2 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.pos_of_ne_zero:rw | lt | div | retrieved_premise | 2 | 0 | 0 | 0 | 0 | 0 |
| retrieved:Nat.pos_of_ne_zero:simp | lt | div | retrieved_premise | 2 | 0 | 0 | 0 | 0 | 0 |

## Zero-win skeletons with assist credit (MUST-PROTECT)

These skeletons never closed a proof but advanced state into a form a
later tactic closed within K≤3 steps. NS5's wins-only `disable_dead_skeleton`
would prune them — NS6's safe pruning rule must protect them.

| skeleton | shape | origin | advances | assist@1 | assist@2 | assist@3 | assisted theorems |
|---|---|---|---:|---:|---:|---:|---|
| pt_any_13 | any | priority_template | 2 | 2 | 2 | 2 | Nat.add_mod_eq_ite |
| retrieved:Nat.div_lt_iff_lt_mul:rw | iff | retrieved_premise | 2 | 2 | 2 | 2 | Nat.div_lt_iff_lt_mul' |

## Truly dead skeletons (safe to prune)

`attempts >= 5` AND `direct_wins = advances = assist_wins_k3 = 0`.

| skeleton | shape | origin | attempts |
|---|---|---|---:|
| pt_eq_15 | eq | priority_template | 22 |
| fb_23 | any | fallback_tactic | 14 |
| pt_le_16 | le | priority_template | 14 |
| fb_24 | any | fallback_tactic | 13 |
| fam_div_18 | any | family_tactic | 9 |
| retrieved:Nat.div_lt_iff_lt_mul':rw | iff | retrieved_premise | 6 |
| retrieved:Nat.div_lt_iff_lt_mul':simp | iff | retrieved_premise | 6 |
| retrieved:Nat.div_lt_iff_lt_mul:simp | iff | retrieved_premise | 6 |
| pt_lt_10 | lt | priority_template | 5 |
| pt_lt_9 | lt | priority_template | 5 |

## Protection summary

| category | count |
|---|---:|
| direct-win skeletons (protected) | 14 |
| zero-win assist@3 skeletons (must-protect) | 2 |
| advance-only skeletons (review) | 1 |
| low-attempt skeletons (insufficient signal) | 25 |
| truly dead (attempts≥5, no signal) | 10 |
