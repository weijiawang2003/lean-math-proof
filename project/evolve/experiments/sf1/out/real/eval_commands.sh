#!/usr/bin/env bash
set -uo pipefail

# batch=sf1_multiset_holdout set=sf1_multiset_holdout_runnable policy=rc1 supported=True size=3
/opt/anaconda3/bin/python3 scripts/sf1_run_eval.py --theorem-set-file project/evolve/experiments/sf1/theorem_sets/sf1_multiset_holdout_runnable.json --out-dir project/evolve/experiments/sf2/out/frontier_expansion/eval/rc1_smoke_sf1_multiset_holdout_runnable --top-k 8 --max-steps 8 --policy-type hybrid_evolved --route-config project/evolve/routing/ns24_router.json --strategy-config project/evolve/experiments/rc1/rc1_production_wrapper.json

# batch=sf1_balanced_mini set=sf1_balanced_mini_runnable policy=rc1 supported=True size=6
/opt/anaconda3/bin/python3 scripts/sf1_run_eval.py --theorem-set-file project/evolve/experiments/sf1/theorem_sets/sf1_balanced_mini_runnable.json --out-dir project/evolve/experiments/sf2/out/frontier_expansion/eval/rc1_smoke_sf1_balanced_mini_runnable --top-k 8 --max-steps 8 --policy-type hybrid_evolved --route-config project/evolve/routing/ns24_router.json --strategy-config project/evolve/experiments/rc1/rc1_production_wrapper.json

# batch=sf1_frontier_all set=sf1_frontier_runnable_subset policy=rc1 supported=True size=20
/opt/anaconda3/bin/python3 scripts/sf1_run_eval.py --theorem-set-file project/evolve/experiments/sf1/theorem_sets/sf1_frontier_runnable_subset.json --out-dir project/evolve/experiments/sf2/out/frontier_expansion/eval/rc1_smoke_sf1_frontier_runnable_subset --top-k 8 --max-steps 8 --policy-type hybrid_evolved --route-config project/evolve/routing/ns24_router.json --strategy-config project/evolve/experiments/rc1/rc1_production_wrapper.json

