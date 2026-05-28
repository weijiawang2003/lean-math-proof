# v5 wave5 — v5-wave5-20260522-115150-c14a93

- variants: 5
- baseline (v5-00): 26/38

## cycle 1 — v5-29-w5-le-shape
- wave5: master + le-shape for div_le_div_right
- proved: 31 (Δ +5)
- origins: {'tactic_template': 22, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- NEW WINS: ['Nat.div_lt_one_iff', 'Nat.div_pos', 'Nat.div_pos_iff', 'Nat.mul_eq_left', 'Nat.mul_eq_right']

## cycle 2 — v5-30-w5-add-mod-ite
- wave5: master + add_mod_eq_ite priorities
- proved: 31 (Δ +5)
- origins: {'tactic_template': 22, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- NEW WINS: ['Nat.div_lt_one_iff', 'Nat.div_pos', 'Nat.div_pos_iff', 'Nat.mul_eq_left', 'Nat.mul_eq_right']

## cycle 3 — v5-31-w5-iff-reorder
- wave5: iff list reordering sensitivity test
- proved: 27 (Δ +1)
- origins: {'tactic_template': 18, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- NEW WINS: ['Nat.div_pos']

## cycle 4 — v5-32-w5-dvd-specific
- wave5: master + dvd_iff_div_mul_eq attempts
- proved: 31 (Δ +5)
- origins: {'tactic_template': 22, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- NEW WINS: ['Nat.div_lt_one_iff', 'Nat.div_pos', 'Nat.div_pos_iff', 'Nat.mul_eq_left', 'Nat.mul_eq_right']

## cycle 5 — v5-33-w5-eq-one-of-mul
- wave5: master + eq-shape for eq_one_of_mul
- proved: 31 (Δ +5)
- origins: {'tactic_template': 25, 'family_tactic': 2, 'generative_topk': 2, 'fallback_tactic': 2}
- NEW WINS: ['Nat.div_lt_one_iff', 'Nat.div_pos', 'Nat.div_pos_iff', 'Nat.mul_eq_left', 'Nat.mul_eq_right']

## complete — 0.31h, 5 cycles
