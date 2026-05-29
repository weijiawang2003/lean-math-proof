# SX1 preservation matrix

Combined SX1 sequence config with the flag **ON**, run over each preservation set's initial states. Gated namespaces: ['List', 'Multiset', 'Option'].

| set | n | gated ns? | emissions (gated) | emissions (off-gate) | NS9 floor |
|---|---|---|---|---|---|
| demo_v1 | 15 | False | 0 | 0 | 11/15 |
| nat_defs_medium | 38 | False | 0 | 0 | 37/38 |
| nat_defs_large_v5 | 64 | False | 0 | 0 | 49/65 |
| ns17_set_extra | 30 | False | 0 | 0 | — |
| ns17_finset_extra | 30 | False | 0 | 0 | — |
| wx2_list_cases_easy | 38 | True | 38 | 0 | — |
| wx2_list_cases_medium | 33 | True | 27 | 0 | — |
| wx2_list_induction | 12 | True | 12 | 0 | — |
| wx3_multiset_induction_heldout | 45 | True | 34 | 0 | — |

**Off-gate emissions total: 0** (PASS — zero).

NS9 canonical floors are preserved: the genome is byte-identical to `ns9_best_genome.json`, sequence plans are additive to the ranked list, and the planner emits nothing on the Nat/demo surfaces. Floors: medium 37/38, large 49/65, demo 11/15.
