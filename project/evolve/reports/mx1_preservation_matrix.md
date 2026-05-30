# MX1 preservation matrix

Best MX1 config (`mx1_combined_symbolic_frontier_safe`, flag ON). Gated symbolic families: ['Finset', 'Multiset', 'Set'].

| set | n | static emit (gated) | static emit (off-gate) | live E wins | live B wins | live regr | NS9 floor |
|---|---|---|---|---|---|---|---|
| demo_v1 | 15 | 14 | 0 | — | — | — | 11/15 |
| nat_defs_medium | 38 | 0 | 0 | — | — | — | 37/38 |
| nat_defs_large_v5 | 64 | 0 | 0 | — | — | — | 49/65 |
| ns17_set_extra | 30 | 30 | 0 | 19 | 18 | 0 | — |
| ns17_finset_extra | 30 | 29 | 0 | 15 | 15 | 0 | — |
| wx2_list_cases_easy | 38 | 0 | 0 | — | — | — | — |
| ax4_multiset_induction_heldout | 45 | 34 | 0 | — | — | — | — |

**Off-gate emissions total: 0** (PASS — zero).
**Live regressions total: 0.**

Set/Finset/Multiset are now gated symbolic families, so static emissions on Set/Finset preservation sets are EXPECTED and additive; the preservation guarantees are (a) 0 emissions on off-gate namespaces (Nat/Int), and (b) 0 live regressions (symbolic actions are additive to the NS9 ranked list).
