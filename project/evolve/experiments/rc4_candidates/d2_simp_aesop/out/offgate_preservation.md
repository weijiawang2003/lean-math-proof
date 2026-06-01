# RC4C off-gate & preservation

- off-gate emissions (all, must be 0): **0**
- off-gate emissions (nonoverlap, must be 0): **0**
- emitted-and-failed: 19/42 (rate 0.452) — honest negatives
- regressions: 0 (additive evaluator)
- verdict: **OFFGATE_CLEAN**

## Gate emissions per set

| set | n | emit_all | emit_nonoverlap | must_not_fire | ns_split |
|---|---|---|---|---|---|
| known_wins_all | 12 | 12 | 6 | False | {'Finset': 1, 'List': 1, 'Multiset': 3, 'Set': 7} |
| known_wins_nonoverlap | 4 | 4 | 4 | False | {'Finset': 1, 'List': 1, 'Multiset': 1, 'Set': 1} |
| fresh_holdout_all | 30 | 30 | 29 | False | {'Finset': 3, 'List': 4, 'Multiset': 21, 'Set': 2} |
| fresh_holdout_nonoverlap | 20 | 20 | 20 | False | {'Finset': 3, 'List': 4, 'Multiset': 13} |
| negative_controls | 18 | 0 | 0 | True | {} |
| namespace_negative_controls | 20 | 0 | 0 | True | {} |
| canonical_smoke | 45 | 0 | 0 | True | {} |

## Canonical floors (literal RC2)

| floor | n | rc2_solved | gate_fires |
|---|---|---|---|
| demo_v1 | 15 | 12 | 0 |
| nat_defs_medium | 15 | 14 | 0 |
| nat_defs_large_v5 | 15 | 14 | 0 |
