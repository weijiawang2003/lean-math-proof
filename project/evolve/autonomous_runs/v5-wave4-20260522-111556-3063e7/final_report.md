# v5 autonomous research final report — v5-wave4-20260522-111556-3063e7

- theorem set: nat_defs_medium
- total runtime: 0.40h
- cycles: 6

## scoreboard
| # | variant | dir | proved | Δ | prog | err | tb attempts/adv/won | newly proved |
|---|---------|-----|--------|----|------|-----|---------------------|--------------|
| 1 | v5-23-w4-split-ifs | B+priority | 26 | +0 | 5 | 7 | 0/0/0 | — |
| 2 | v5-24-w4-dvd-iff | A+priority | 26 | +0 | 5 | 7 | 0/0/0 | — |
| 3 | v5-25-w4-div-pos | A+priority | 28 | +2 | 5 | 5 | 0/0/0 | Nat.div_pos, Nat.div_pos_iff |
| 4 | v5-26-w4-sqrt-pow | B+priority | 26 | +0 | 5 | 7 | 0/0/0 | — |
| 5 | v5-27-w4-master | all+priority | 31 | +5 | 4 | 3 | 0/0/0 | Nat.div_lt_one_iff, Nat.div_pos, Nat.div_pos_iff, Nat.mul_eq_left, Nat.mul_eq_right |
| 6 | v5-28-w4-super-kitchen | all+priority | 31 | +5 | 4 | 3 | 0/0/0 | Nat.div_lt_one_iff, Nat.div_pos, Nat.div_pos_iff, Nat.mul_eq_left, Nat.mul_eq_right |

## best candidate

- name: `v5-27-w4-master`
- direction: all+priority
- proved: **31**  (Δ +5)
- description: wave4: master combo of all v5 wins + new attempts
- newly proved: ['Nat.div_lt_one_iff', 'Nat.div_pos', 'Nat.div_pos_iff', 'Nat.mul_eq_left', 'Nat.mul_eq_right']
