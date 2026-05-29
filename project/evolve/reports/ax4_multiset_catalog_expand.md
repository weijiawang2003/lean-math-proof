# AX4 Stage 1 — Multiset catalog expansion (broader discovered frontier)

AX3 consumed the confirmed-available Multiset surface (260 available, 259 already in prior sets). To reach Green we mine the broader discovered catalog.

- discovered Multiset names: **573**
- minus available-260 (consumed) + prior sets + labeled
- → **frontier = 313** availability-unconfirmed candidates

## Buckets (induction-likelihood)

| bucket | n |
|---|---:|
| high | 71 |
| medium | 119 |
| hard | 45 |
| negative | 78 |

- cross-surface (non-Basic file): **50**, basic: 263
- difficulty: {'medium': 281, 'easy': 5, 'hard': 20, '?': 7}

## Frontier by file

| file | n |
|---|---:|
| `Mathlib/Data/Multiset/Basic.lean` | 263 |
| `Mathlib/Data/Multiset/Dedup.lean` | 16 |
| `Mathlib/Data/Multiset/Lattice.lean` | 14 |
| `Mathlib/Data/Multiset/Bind.lean` | 13 |
| `Mathlib/Data/Finset/Card.lean` | 5 |
| `Mathlib/Data/Finset/Image.lean` | 1 |
| `Mathlib/Data/Finset/Lattice.lean` | 1 |

## Caveat

Frontier availability is UNCONFIRMED — these names were discovered by source scan but never probe-loaded. Stage 3 confirms availability at mine time (LeanDojo load), timeout-guarded per the AX3 REPL-hang incident. Expect attrition: some names will be unavailable / private / deprecated and silently drop out.
