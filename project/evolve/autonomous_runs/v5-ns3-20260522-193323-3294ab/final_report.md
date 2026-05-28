# v5 autonomous research final report — v5-ns3-20260522-193323-3294ab

- theorem set: nat_defs_medium
- total runtime: 0.32h
- cycles: 6

## scoreboard
| # | variant | dir | proved | Δ | prog | err | tb attempts/adv/won | newly proved |
|---|---------|-----|--------|----|------|-----|---------------------|--------------|
| 1 | ns3-dvd | A+priority | 32 | +1 | 4 | 2 | 0/0/0 | Nat.dvd_iff_div_mul_eq |
| 2 | ns3-eq-one-mul | A+priority | 32 | +1 | 4 | 2 | 0/0/0 | Nat.eq_one_of_mul_eq_one_left |
| 3 | ns3-add-mod-ite | C+priority | 32 | +1 | 3 | 3 | 0/0/0 | Nat.add_mod_eq_ite |
| 4 | ns3-div-le | A+priority | 32 | +1 | 3 | 3 | 0/0/0 | Nat.div_le_div_right |
| 5 | ns3-sqrt-pow | A+priority | 32 | +1 | 2 | 4 | 0/0/0 | Nat.pow_lt_pow_iff_left, Nat.sqrt_lt |
| 6 | ns3-combined | all+priority | 35 | +4 | 1 | 2 | 0/0/0 | Nat.div_le_div_right, Nat.dvd_iff_div_mul_eq, Nat.eq_one_of_mul_eq_one_left, Nat.pow_lt_pow_iff_left, Nat.sqrt_lt |

## best candidate

- name: `ns3-combined`
- direction: all+priority
- proved: **35**  (Δ +4)
- description: NS3: all promising patches stacked
- newly proved: ['Nat.div_le_div_right', 'Nat.dvd_iff_div_mul_eq', 'Nat.eq_one_of_mul_eq_one_left', 'Nat.pow_lt_pow_iff_left', 'Nat.sqrt_lt']
- regressions: ['Nat.div_pos_iff']
