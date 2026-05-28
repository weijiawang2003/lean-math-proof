# v5 autonomous research final report — v5-wave6-20260522-121607-e99584

- theorem set: nat_defs_medium
- total runtime: 0.29h
- cycles: 5

## scoreboard
| # | variant | dir | proved | Δ | prog | err | tb attempts/adv/won | newly proved |
|---|---------|-----|--------|----|------|-----|---------------------|--------------|
| 1 | v5-34-w6-dvd-alt | A+priority | 31 | +5 | 4 | 3 | 0/0/0 | Nat.div_lt_one_iff, Nat.div_pos, Nat.div_pos_iff, Nat.mul_eq_left, Nat.mul_eq_right |
| 2 | v5-35-w6-add-mod-ite | B+priority | 31 | +5 | 4 | 3 | 0/0/0 | Nat.div_lt_one_iff, Nat.div_pos, Nat.div_pos_iff, Nat.mul_eq_left, Nat.mul_eq_right |
| 3 | v5-36-w6-eq-one-alt | B+priority | 31 | +5 | 4 | 3 | 0/0/0 | Nat.div_lt_one_iff, Nat.div_pos, Nat.div_pos_iff, Nat.mul_eq_left, Nat.mul_eq_right |
| 4 | v5-37-w6-div-le-div | B+priority | 31 | +5 | 4 | 3 | 0/0/0 | Nat.div_lt_one_iff, Nat.div_pos, Nat.div_pos_iff, Nat.mul_eq_left, Nat.mul_eq_right |
| 5 | v5-38-w6-combined | all+priority | 31 | +5 | 4 | 3 | 0/0/0 | Nat.div_lt_one_iff, Nat.div_pos, Nat.div_pos_iff, Nat.mul_eq_left, Nat.mul_eq_right |

## best candidate

- name: `v5-34-w6-dvd-alt`
- direction: A+priority
- proved: **31**  (Δ +5)
- description: wave6: dvd term-mode alternatives
- newly proved: ['Nat.div_lt_one_iff', 'Nat.div_pos', 'Nat.div_pos_iff', 'Nat.mul_eq_left', 'Nat.mul_eq_right']
