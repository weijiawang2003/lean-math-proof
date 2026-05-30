# MX2 preservation matrix

Broad Set-aesop config (`mx2_set_aesop_safe`). aesop gate: `['Set.']`.

| set | n | non-Set | non-Set aesop-admissible | live A | live E | live regr | NS9 floor |
|---|---|---|---|---|---|---|---|
| demo_v1 | 15 | 4 | 0 | — | — | — | 11/15 |
| nat_defs_medium | 38 | 38 | 0 | — | — | — | 37/38 |
| nat_defs_large_v5 | 65 | 65 | 0 | — | — | — | 49/65 |
| ns17_set_extra | 30 | 0 | 0 | — | — | — | — |
| ns17_finset_extra | 30 | 30 | 0 | — | — | — | — |
| ns14_set_finset_extra | 20 | 10 | 0 | — | — | — | — |
| wx2_list_cases_easy | 40 | 40 | 0 | — | — | — | — |
| ax4_multiset_induction_heldout | 45 | 45 | 0 | — | — | — | — |

**Non-Set aesop-admissible total: 0** (PASS — zero).
**Live regressions total: 0.**

The aesop name-gate forbids any aesop tactic on a non-Set theorem (static guarantee). aesop is additive to the ranked list, so it cannot remove a production win (0 live regressions).
