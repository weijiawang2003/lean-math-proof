# v5 wave4 — v5-wave4-20260522-111556-3063e7

- variants: 6
- baseline (from v5-00): 26/38

## cycle 1 — v5-23-w4-split-ifs
- wave4: split_ifs across non-iff shapes
- proved: 26 (Δ +0)
- origins: {'fallback_tactic': 18, 'family_tactic': 4, 'generative_topk': 4}

## cycle 2 — v5-24-w4-dvd-iff
- wave4: dvd_iff_div_mul_eq specific
- proved: 26 (Δ +0)
- origins: {'fallback_tactic': 18, 'family_tactic': 4, 'generative_topk': 4}

## cycle 3 — v5-25-w4-div-pos
- wave4: div_pos and div_pos_iff
- proved: 28 (Δ +2)
- origins: {'fallback_tactic': 18, 'family_tactic': 4, 'generative_topk': 4, 'tactic_template': 2}
- NEW WINS: ['Nat.div_pos', 'Nat.div_pos_iff']

## cycle 4 — v5-26-w4-sqrt-pow
- wave4: sqrt_lt and pow_lt forms
- proved: 26 (Δ +0)
- origins: {'fallback_tactic': 18, 'family_tactic': 4, 'generative_topk': 4}

## cycle 5 — v5-27-w4-master
- wave4: master combo of all v5 wins + new attempts
- proved: 31 (Δ +5)
- origins: {'tactic_template': 22, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- NEW WINS: ['Nat.div_lt_one_iff', 'Nat.div_pos', 'Nat.div_pos_iff', 'Nat.mul_eq_left', 'Nat.mul_eq_right']

## cycle 6 — v5-28-w4-super-kitchen
- wave4: SUPER-KITCHEN — every confirmed priority win
- proved: 31 (Δ +5)
- origins: {'tactic_template': 22, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- NEW WINS: ['Nat.div_lt_one_iff', 'Nat.div_pos', 'Nat.div_pos_iff', 'Nat.mul_eq_left', 'Nat.mul_eq_right']

## complete — 0.40h, 6 cycles
