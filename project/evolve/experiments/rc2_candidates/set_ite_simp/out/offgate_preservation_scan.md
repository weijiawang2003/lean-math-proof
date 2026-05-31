# RC2 — SET_ITE_SIMP Off-Gate / Preservation Scan

- scanned=25 | gate emissions=2 | **off-gate emissions=0** | positive Set controls fired=2/2 | sanity_ok=True

| surface | is_set | count | fired |
|---|---|---|---|
| nat_only | False | 4 | 0 |
| int_only | False | 2 | 0 |
| multiset | False | 2 | 0 |
| demo_v1 | False | 3 | 0 |
| nat_defs_medium | False | 2 | 0 |
| set_positive_control | True | 2 | 2 |
| set_ite_negative_controls (name-only) | False | 5 | 0 |
| set_ite_canonical_smoke (name-only) | False | 5 | 0 |

> Gate is a pure name+goal predicate; dry scan is deterministic and sufficient. Live eval on canonical surfaces unnecessary (gate cannot fire without 'Set' in the name). SET_ITE_SIMP is off-by-default in prod.