# v5 autonomous research final report — v5-wave5-20260522-115150-c14a93

- theorem set: nat_defs_medium
- total runtime: 0.31h
- cycles: 5

## scoreboard
| # | variant | dir | proved | Δ | prog | err | tb attempts/adv/won | newly proved |
|---|---------|-----|--------|----|------|-----|---------------------|--------------|
| 1 | v5-29-w5-le-shape | B+priority | 31 | +5 | 4 | 3 | 0/0/0 | Nat.div_lt_one_iff, Nat.div_pos, Nat.div_pos_iff, Nat.mul_eq_left, Nat.mul_eq_right |
| 2 | v5-30-w5-add-mod-ite | B+priority | 31 | +5 | 4 | 3 | 0/0/0 | Nat.div_lt_one_iff, Nat.div_pos, Nat.div_pos_iff, Nat.mul_eq_left, Nat.mul_eq_right |
| 3 | v5-31-w5-iff-reorder | C+priority | 27 | +1 | 5 | 6 | 0/0/0 | Nat.div_pos |
| 4 | v5-32-w5-dvd-specific | A+priority | 31 | +5 | 4 | 3 | 0/0/0 | Nat.div_lt_one_iff, Nat.div_pos, Nat.div_pos_iff, Nat.mul_eq_left, Nat.mul_eq_right |
| 5 | v5-33-w5-eq-one-of-mul | B+priority | 31 | +5 | 4 | 3 | 0/0/0 | Nat.div_lt_one_iff, Nat.div_pos, Nat.div_pos_iff, Nat.mul_eq_left, Nat.mul_eq_right |

## best candidate

- name: `v5-29-w5-le-shape`
- direction: B+priority
- proved: **31**  (Δ +5)
- description: wave5: master + le-shape for div_le_div_right
- newly proved: ['Nat.div_lt_one_iff', 'Nat.div_pos', 'Nat.div_pos_iff', 'Nat.mul_eq_left', 'Nat.mul_eq_right']
