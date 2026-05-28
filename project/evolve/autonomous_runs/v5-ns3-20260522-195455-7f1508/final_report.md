# v5 autonomous research final report — v5-ns3-20260522-195455-7f1508

- theorem set: nat_defs_medium
- total runtime: 0.10h
- cycles: 2

## scoreboard
| # | variant | dir | proved | Δ | prog | err | tb attempts/adv/won | newly proved |
|---|---------|-----|--------|----|------|-----|---------------------|--------------|
| 1 | ns3-sqrt-pow | A+priority | 33 | +2 | 2 | 3 | 0/0/0 | Nat.pow_lt_pow_iff_left, Nat.sqrt_lt |
| 2 | ns3-combined | all+priority | 36 | +5 | 1 | 1 | 0/0/0 | Nat.div_le_div_right, Nat.dvd_iff_div_mul_eq, Nat.eq_one_of_mul_eq_one_left, Nat.pow_lt_pow_iff_left, Nat.sqrt_lt |

## best candidate

- name: `ns3-combined`
- direction: all+priority
- proved: **36**  (Δ +5)
- description: NS3: all promising patches stacked
- newly proved: ['Nat.div_le_div_right', 'Nat.dvd_iff_div_mul_eq', 'Nat.eq_one_of_mul_eq_one_left', 'Nat.pow_lt_pow_iff_left', 'Nat.sqrt_lt']
