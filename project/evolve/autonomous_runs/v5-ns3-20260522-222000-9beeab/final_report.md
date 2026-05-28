# v5 autonomous research final report — v5-ns3-20260522-222000-9beeab

- theorem set: nat_defs_medium
- total runtime: 0.09h
- cycles: 2

## scoreboard
| # | variant | dir | proved | Δ | prog | err | tb attempts/adv/won | newly proved |
|---|---------|-----|--------|----|------|-----|---------------------|--------------|
| 1 | ns3-combined | all+priority | 37 | +6 | 0 | 1 | 0/0/0 | Nat.add_mod_eq_ite, Nat.div_le_div_right, Nat.dvd_iff_div_mul_eq, Nat.eq_one_of_mul_eq_one_left, Nat.pow_lt_pow_iff_left, Nat.sqrt_lt |
| 2 | ns3-combined-mirrored | all+priority | 37 | +6 | 0 | 1 | 0/0/0 | Nat.add_mod_eq_ite, Nat.div_le_div_right, Nat.dvd_iff_div_mul_eq, Nat.eq_one_of_mul_eq_one_left, Nat.pow_lt_pow_iff_left, Nat.sqrt_lt |

## best candidate

- name: `ns3-combined`
- direction: all+priority
- proved: **37**  (Δ +6)
- description: NS3: all promising patches stacked (no manual mirror)
- newly proved: ['Nat.add_mod_eq_ite', 'Nat.div_le_div_right', 'Nat.dvd_iff_div_mul_eq', 'Nat.eq_one_of_mul_eq_one_left', 'Nat.pow_lt_pow_iff_left', 'Nat.sqrt_lt']
