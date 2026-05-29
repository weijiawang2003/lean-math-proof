# AX1 — WX2 vs symbolic-action equivalence

Does the AX1 symbolic-action wrapper reproduce the WX2 custom cases wrapper? Same sets, `ns24_router`, top-k 8 max-steps 8.

| set | ns class | WX2 | AX1 | Δ | ax1_only | wx2_only | symbolic emit |
|---|---|---:|---:|---:|---:|---:|---:|
| wx2_list_cases_easy | List | 9 | 9 | +0 | 0 | 0 | 164 |
| wx2_list_cases_medium | List | 17 | 17 | +0 | 0 | 0 | 60 |
| cx3_option_simp_easy | Option | 26 | 26 | +0 | 0 | 0 | 15 |
| cx3_option_cases_medium | Option | 11 | 11 | +0 | 0 | 0 | 6 |
| cx3_bool_option_mixed | Option | 22 | 22 | +0 | 0 | 0 | 14 |
| demo_v1 | mixed-Nat | 11 | 11 | +0 | 0 | 0 | 0 |

- **AX1 reproduces WX2: True** (max per-set |Δ| = 0, threshold 2).
- Total wins: WX2 96 vs AX1 96.
- Symbolic emissions outside gated Option/List namespaces: 0 (demo control).
