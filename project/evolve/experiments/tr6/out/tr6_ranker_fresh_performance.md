# TR6 ranker fresh-frontier performance

## Decision: `RANKER_GENERALIZES_TO_FRESH_FRONTIER`

- fresh failures searched: 137
- programs attempted: B5 654 + B10 622 + B20 1182 = **2458** (+540 controls)
- credited wins by budget: B5 9, B10 11, B20 18
- **fresh credited total: 18** (13 non-Set)
- success/probe: **0.0073** (TR5 reference 0.0161)
- mean first-success rank: 8.28 | no-win rate: 0.8321
- unknown-name failures encountered: 42

## By namespace

| ns | searched | credited |
|---|---|---|
| Finset | 30 | 2 |
| List | 29 | 1 |
| Multiset | 22 | 9 |
| Set | 21 | 5 |
| Nat | 14 | 1 |
|  | 10 | 0 |
| Option | 3 | 0 |
| AntitoneOn | 2 | 0 |
| MonotoneOn | 2 | 0 |
| Equiv | 1 | 0 |
| IsGLB | 1 | 0 |
| IsLUB | 1 | 0 |
| PLift | 1 | 0 |

## By winning family

{'d2_simp_aesop': 9, 'd1_simp_lemma': 3, 'd1_tauto': 3, 'd2_rw_aesop': 1, 'd1_exact': 1, 'def_unfold_simp': 1}
