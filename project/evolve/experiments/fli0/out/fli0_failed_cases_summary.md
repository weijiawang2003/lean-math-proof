# FLI0 failed-case extraction summary

- **total failures: 455** | **clean failures: 327** (fresh: 327)
- by source stage: {'RC5V2': 141, 'RC5V3': 314} | clean: {'RC5V2': 133, 'RC5V3': 194}
- RC5V2∩RC5V3 overlap: 0
- by dynamic result: {'failed': 328, 'unknown_name': 15, 'infra_error': 112}
- by failure reason: {'all attempts unknown_name': 15, 'setup/network error, no live attempt': 112}

## Clean failures by namespace

| namespace | clean | all |
|---|---|---|
| Finset | 107 | 120 |
| Multiset | 35 | 104 |
| Nat | 87 | 99 |
| List | 41 | 71 |
| Set | 57 | 61 |

## Clean failures by feature

| feature | clean |
|---|---|
| has_eq | 185 |
| has_nat_arith | 134 |
| has_map_filter | 125 |
| has_singleton | 122 |
| has_mem | 77 |
| has_card | 76 |
| has_order | 75 |
| has_iff | 61 |
| has_union_inter | 58 |
| has_subset | 43 |
| has_disjoint | 4 |
