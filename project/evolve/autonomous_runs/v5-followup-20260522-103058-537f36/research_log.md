# v5 followup run — v5-followup-20260522-103058-537f36

- variants queued: 11

## cycle 1 — v5-12-prio-div-hyp  [B+priority]
- priority div rewrites with hyp_pos
- elapsed: 0.00h  remaining: 3.00h
- proved: 27 (Δ +0)  prog: 4  err: 7
- origins: {'fallback_tactic': 18, 'family_tactic': 4, 'generative_topk': 4, 'tactic_template': 1}

## cycle 2 — v5-13-prio-iff-constructor  [A+priority]
- priority iff constructor split
- elapsed: 0.07h  remaining: 2.93h
- proved: 26 (Δ -1)  prog: 5  err: 7
- origins: {'tactic_template': 16, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 5}
- REGRESSIONS: ['Nat.div_lt_one_iff']

## cycle 3 — v5-14-prio-combo  [A+B+priority]
- priority + family + term + split_ifs
- elapsed: 0.14h  remaining: 2.86h
- proved: 27 (Δ +0)  prog: 4  err: 7
- origins: {'tactic_template': 18, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- term_builder: 32/0/0

## cycle 4 — v5-15-prio-mul-specific  [B+priority]
- priority mul_eq term-mode skeletons
- elapsed: 0.20h  remaining: 2.80h
- proved: 28 (Δ +1)  prog: 5  err: 5
- origins: {'fallback_tactic': 18, 'family_tactic': 4, 'generative_topk': 4, 'tactic_template': 2}
- NEW WINS: ['Nat.mul_eq_left', 'Nat.mul_eq_right']
- REGRESSIONS: ['Nat.div_lt_one_iff']

## cycle 5 — v5-16-prio-sqrt-pow  [B+priority]
- priority sqrt and pow
- elapsed: 0.27h  remaining: 2.73h
- proved: 26 (Δ -1)  prog: 5  err: 7
- origins: {'fallback_tactic': 18, 'family_tactic': 4, 'generative_topk': 4}
- REGRESSIONS: ['Nat.div_lt_one_iff']

## cycle 6 — v5-17-prio-term-iff  [A+priority]
- priority term_builder iff (advanced)
- elapsed: 0.34h  remaining: 2.66h
- proved: 26 (Δ -1)  prog: 5  err: 7
- origins: {'tactic_template': 17, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- REGRESSIONS: ['Nat.div_lt_one_iff']

## cycle 7 — v5-18-prio-kitchen  [all+priority]
- kitchen sink: every priority template
- elapsed: 0.41h  remaining: 2.59h
- proved: 29 (Δ +2)  prog: 4  err: 5
- origins: {'tactic_template': 20, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- term_builder: 8/0/0
- NEW WINS: ['Nat.mul_eq_left', 'Nat.mul_eq_right']

## cycle 8 — v5-19-prio-split-ifs  [B+priority]
- priority split_ifs under 'any' key
- elapsed: 0.47h  remaining: 2.53h
- proved: 26 (Δ -1)  prog: 5  err: 7
- origins: {'fallback_tactic': 18, 'family_tactic': 4, 'generative_topk': 4}
- REGRESSIONS: ['Nat.div_lt_one_iff']

## cycle 9 — v5-20-prio-div-pos  [B+priority]
- priority div_pos and div_pos_iff
- elapsed: 0.55h  remaining: 2.45h
- proved: 28 (Δ +1)  prog: 5  err: 5
- origins: {'fallback_tactic': 18, 'family_tactic': 4, 'tactic_template': 3, 'generative_topk': 3}
- NEW WINS: ['Nat.div_pos', 'Nat.div_pos_iff']
- REGRESSIONS: ['Nat.div_lt_one_iff']

## cycle 10 — v5-21-prio-iff-basic  [A+priority]
- priority iff term_builder basic (minimal)
- elapsed: 0.62h  remaining: 2.38h
- proved: 26 (Δ -1)  prog: 5  err: 7
- origins: {'tactic_template': 17, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- REGRESSIONS: ['Nat.div_lt_one_iff']

## cycle 11 — v5-22-deny-derailers  [B+priority+deny]
- deny derailing simps + priority kitchen
- elapsed: 0.68h  remaining: 2.32h
- proved: 29 (Δ +2)  prog: 4  err: 5
- origins: {'tactic_template': 20, 'family_tactic': 2, 'generative_topk': 3, 'fallback_tactic': 4}
- NEW WINS: ['Nat.mul_eq_left', 'Nat.mul_eq_right']

## complete — 0.74h, 11 cycles, best=v5-18-prio-kitchen proved=29
