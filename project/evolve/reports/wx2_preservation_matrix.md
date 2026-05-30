# WX2 — broad preservation matrix

NS9 wrapper vs **WX2-promoted** (Option-only state-aware cases) on `ns24_router`. Promoted config = NS9 genome + the WX1-validated Option cases block (2 tactics).

| set | ns class | NS9 | WX2-prom | Δ | regress | option_cases emit |
|---|---|---:|---:|---:|---:|---:|
| cx3_option_simp_easy | Option | 15 | 26 | +11 | 0 | 19 |
| cx3_option_cases_medium | Option | 7 | 11 | +4 | 0 | 7 |
| cx3_bool_option_mixed | Option/Bool | 18 | 22 | +4 | 0 | 18 |
| cx3_bool_decide_easy | Bool | 2 | 2 | +0 | 0 | 0 |
| demo_v1 | mixed-Nat | 11 | 11 | +0 | 0 | 0 |
| nat_defs_medium | Nat | 37 | 37 | +0 | 0 | 0 |
| ns17_set_extra | Set | 18 | 18 | +0 | 0 | 0 |
| ns17_finset_extra | Finset | 15 | 15 | +0 | 0 | 0 |

- **Option-surface delta vs NS9: +19** (WX1 gain retained by the promoted 2-tactic config).
- **Non-Option regressions: 0.**
- **`wrapper_option_cases` emissions outside Option: 0.** (Namespace gate holds; preservation by construction confirmed empirically.)
- `nat_defs_large_v5` / `ns14_set_finset_extra`: unchanged by the same gate (ranked-list identity), not re-run.
