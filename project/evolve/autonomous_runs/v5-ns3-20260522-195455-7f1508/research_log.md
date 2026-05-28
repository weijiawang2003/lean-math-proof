# NS3 — v5-ns3-20260522-195455-7f1508

- variants: 2
- baseline (v5-27 master): 31/38

## cycle 1 — ns3-sqrt-pow
- NS3: fix sqrt_lt + pow_lt_pow_iff_left
- proved: 33 (Δ +2)
- origins: {'tactic_template': 24, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- NEW WINS: ['Nat.pow_lt_pow_iff_left', 'Nat.sqrt_lt']

## cycle 2 — ns3-combined
- NS3: all promising patches stacked
- proved: 36 (Δ +5)
- origins: {'tactic_template': 27, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- NEW WINS: ['Nat.div_le_div_right', 'Nat.dvd_iff_div_mul_eq', 'Nat.eq_one_of_mul_eq_one_left', 'Nat.pow_lt_pow_iff_left', 'Nat.sqrt_lt']

## complete — 0.10h, 2 cycles
