# RC4D off-gate & preservation

- off-gate emissions (must be 0): **0**
- RC4C_residue off-gate emissions (must be 0): **0**
- emitted-and-failed by component: {'RC4A': {'fired': 12, 'failed': 2, 'rate': 0.167}, 'RC4B': {'fired': 39, 'failed': 18, 'rate': 0.462}, 'RC4C_residue': {'fired': 34, 'failed': 18, 'rate': 0.529}}
- regressions: 0 (additive evaluator)
- verdict: **OFFGATE_CLEAN**

## Gate emissions per set

| set | n | emit | must_not_fire | component_split |
|---|---|---|---|---|
| rc4a_known_wins | 5 | 5 | False | {'RC4A': 5} |
| rc4b_known_wins | 16 | 16 | False | {'RC4B': 16, 'RC4C_residue': 9} |
| rc4c_residue_known_wins | 7 | 7 | False | {'RC4C_residue': 7, 'RC4B': 5} |
| component_overlap_controls | 5 | 5 | False | {'RC4B': 5, 'RC4C_residue': 5} |
| composition_fresh_holdout | 34 | 34 | False | {'RC4B': 23, 'RC4C_residue': 23, 'RC4A': 7} |
| negative_controls | 24 | 0 | True | {} |
| namespace_negative_controls | 20 | 0 | True | {} |
| canonical_smoke | 30 | 0 | True | {} |

## Canonical floors (literal RC2)

| floor | n | rc2_solved | gate_fires |
|---|---|---|---|
| demo_v1 | 15 | 12 | 0 |
| nat_defs_medium | 15 | 14 | 0 |
