# NS3 — v5-ns3-20260522-222000-9beeab

- variants: 2
- baseline (v5-27 master): 31/38

## cycle 1 — ns3-combined
- NS3: all promising patches stacked (no manual mirror)
- proved: 37 (Δ +6)
- origins: {'tactic_template': 28, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- NEW WINS: ['Nat.add_mod_eq_ite', 'Nat.div_le_div_right', 'Nat.dvd_iff_div_mul_eq', 'Nat.eq_one_of_mul_eq_one_left', 'Nat.pow_lt_pow_iff_left', 'Nat.sqrt_lt']

## cycle 2 — ns3-combined-mirrored
- NS3: pre-NS3.5 form with manual eq/le mirror
- proved: 37 (Δ +6)
- origins: {'tactic_template': 28, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- NEW WINS: ['Nat.add_mod_eq_ite', 'Nat.div_le_div_right', 'Nat.dvd_iff_div_mul_eq', 'Nat.eq_one_of_mul_eq_one_left', 'Nat.pow_lt_pow_iff_left', 'Nat.sqrt_lt']

## complete — 0.09h, 2 cycles
