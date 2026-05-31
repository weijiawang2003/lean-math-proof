# RC4A off-gate & preservation

- off-gate emissions (must be 0): **0**
- regressions: 0 (additive evaluator)
- verdict: **OFFGATE_CLEAN**

## Gate emissions per set

| set | n | gate_emissions | must_not_fire |
|---|---|---|---|
| known_wins | 5 | 5 | False |
| same_cluster_holdout | 3 | 3 | False |
| fresh_frontier_holdout | 3 | 3 | False |
| negative_controls | 20 | 0 | True |
| canonical_smoke | 45 | 0 | True |

## Canonical floors (literal RC2)

| floor | n | rc2_solved | gate_fires |
|---|---|---|---|
| demo_v1 | 15 | 12 | 0 |
| nat_defs_medium | 15 | 14 | 0 |
| nat_defs_large_v5 | 15 | 14 | 0 |
