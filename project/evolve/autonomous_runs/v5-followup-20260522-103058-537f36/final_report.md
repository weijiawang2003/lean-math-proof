# v5 autonomous research final report — v5-followup-20260522-103058-537f36

- theorem set: nat_defs_medium
- total runtime: 0.74h
- cycles: 11

## scoreboard
| # | variant | dir | proved | Δ | prog | err | tb attempts/adv/won | newly proved |
|---|---------|-----|--------|----|------|-----|---------------------|--------------|
| 1 | v5-12-prio-div-hyp | B+priority | 27 | +0 | 4 | 7 | 0/0/0 | — |
| 2 | v5-13-prio-iff-constructor | A+priority | 26 | -1 | 5 | 7 | 0/0/0 | — |
| 3 | v5-14-prio-combo | A+B+priority | 27 | +0 | 4 | 7 | 32/0/0 | — |
| 4 | v5-15-prio-mul-specific | B+priority | 28 | +1 | 5 | 5 | 0/0/0 | Nat.mul_eq_left, Nat.mul_eq_right |
| 5 | v5-16-prio-sqrt-pow | B+priority | 26 | -1 | 5 | 7 | 0/0/0 | — |
| 6 | v5-17-prio-term-iff | A+priority | 26 | -1 | 5 | 7 | 0/0/0 | — |
| 7 | v5-18-prio-kitchen | all+priority | 29 | +2 | 4 | 5 | 8/0/0 | Nat.mul_eq_left, Nat.mul_eq_right |
| 8 | v5-19-prio-split-ifs | B+priority | 26 | -1 | 5 | 7 | 0/0/0 | — |
| 9 | v5-20-prio-div-pos | B+priority | 28 | +1 | 5 | 5 | 0/0/0 | Nat.div_pos, Nat.div_pos_iff |
| 10 | v5-21-prio-iff-basic | A+priority | 26 | -1 | 5 | 7 | 0/0/0 | — |
| 11 | v5-22-deny-derailers | B+priority+deny | 29 | +2 | 4 | 5 | 0/0/0 | Nat.mul_eq_left, Nat.mul_eq_right |

## best candidate

- name: `v5-18-prio-kitchen`
- direction: all+priority
- proved: **29**  (Δ +2)
- description: kitchen sink: every priority template
- newly proved: ['Nat.mul_eq_left', 'Nat.mul_eq_right']
