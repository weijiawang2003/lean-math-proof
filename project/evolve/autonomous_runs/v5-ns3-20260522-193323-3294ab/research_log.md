# NS3 — v5-ns3-20260522-193323-3294ab

- variants: 6
- baseline (v5-27 master): 31/38

## cycle 1 — ns3-dvd
- NS3: fix dvd_iff_div_mul_eq template
- proved: 32 (Δ +1)
- origins: {'tactic_template': 23, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- NEW WINS: ['Nat.dvd_iff_div_mul_eq']

## cycle 2 — ns3-eq-one-mul
- NS3: new eq slot for eq_one_of_mul_eq_one_left
- proved: 32 (Δ +1)
- origins: {'tactic_template': 23, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- NEW WINS: ['Nat.eq_one_of_mul_eq_one_left']

## cycle 3 — ns3-add-mod-ite
- NS3: multi-step add_mod_eq_ite template
- proved: 32 (Δ +1)
- origins: {'tactic_template': 23, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- NEW WINS: ['Nat.add_mod_eq_ite']

## cycle 4 — ns3-div-le
- NS3: new le slot for div_le_div_right (gcongr)
- proved: 32 (Δ +1)
- origins: {'tactic_template': 23, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- NEW WINS: ['Nat.div_le_div_right']

## cycle 5 — ns3-sqrt-pow
- NS3: fix sqrt_lt + pow_lt_pow_iff_left
- proved: 32 (Δ +1)
- origins: {'tactic_template': 23, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- NEW WINS: ['Nat.pow_lt_pow_iff_left', 'Nat.sqrt_lt']
- regressions: ['Nat.div_pos_iff']

## cycle 6 — ns3-combined
- NS3: all promising patches stacked
- proved: 35 (Δ +4)
- origins: {'tactic_template': 26, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- NEW WINS: ['Nat.div_le_div_right', 'Nat.dvd_iff_div_mul_eq', 'Nat.eq_one_of_mul_eq_one_left', 'Nat.pow_lt_pow_iff_left', 'Nat.sqrt_lt']
- regressions: ['Nat.div_pos_iff']

## complete — 0.32h, 6 cycles
