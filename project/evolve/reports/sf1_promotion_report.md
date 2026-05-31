# SF1 promotion report — ax4_learned_symbolic_selector_off_by_default

- seed: `1729`  | dry_run: `True`
- gate type: `learned_selector`
- control: `rc1` (5 wins)  | experimental: 7 wins  | raw delta: 2
- clean new labels: 1  | over-attributed: 1
- regressions: 0  | off-gate: 0

## Criteria

- [x] positive_delta_over_rc1
- [x] zero_regressions
- [x] zero_off_gate
- [ ] strict_syntactic_gate
- [ ] minimal_sufficient_attribution
- [ ] deterministic_reproducibility

## Recommendation: **MINE_MORE**

Learner shows signal but has 1 clean labels (< 40); mine more before live integration.
