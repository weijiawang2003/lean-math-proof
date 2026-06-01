# RC4B off-gate & preservation

- off-gate emissions (must be 0): **0**
- negative-control emitted-and-failed: 0
- regressions: 0 (additive evaluator)
- verdict: **OFFGATE_CLEAN**

## Gate emissions per set

| set | n | gate_emissions | Set | Multiset | Other | must_not_fire |
|---|---|---|---|---|---|---|
| known_wins | 11 | 11 | 7 | 4 | 0 | False |
| fresh_holdout_set | 8 | 8 | 8 | 0 | 0 | False |
| fresh_holdout_multiset | 20 | 20 | 0 | 20 | 0 | False |
| disjoint_negative_controls | 15 | 0 | 0 | 0 | 0 | True |
| namespace_negative_controls | 20 | 0 | 0 | 0 | 0 | True |
| canonical_smoke | 45 | 0 | 0 | 0 | 0 | True |

## Canonical floors (literal RC2)

| floor | n | rc2_solved | gate_fires |
|---|---|---|---|
| demo_v1 | 15 | 12 | 0 |
| nat_defs_medium | 15 | 14 | 0 |
| nat_defs_large_v5 | 15 | 14 | 0 |
