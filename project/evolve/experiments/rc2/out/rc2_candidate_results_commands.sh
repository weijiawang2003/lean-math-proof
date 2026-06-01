#!/usr/bin/env bash
# RC2 benchmark — policy=rc2_candidate. RC1/NS24 untouched.
set -uo pipefail
cd /Users/weijiawang/dev/dojo_sandbox

# demo_v1 (canonical_floor)
/opt/anaconda3/bin/python3 eval_rollout_all.py --theorem-set demo_v1 --policy-type hybrid_evolved --route-config project/evolve/routing/ns24_router.json --strategy-config project/evolve/experiments/rc2/rc2_candidate_wrapper.json --top-k 8 --max-steps 8 --out-dir project/evolve/experiments/rc2/out/rc2_candidate_runs/demo_v1

# nat_defs_medium (canonical_floor)
/opt/anaconda3/bin/python3 eval_rollout_all.py --theorem-set nat_defs_medium --policy-type hybrid_evolved --route-config project/evolve/routing/ns24_router.json --strategy-config project/evolve/experiments/rc2/rc2_candidate_wrapper.json --top-k 8 --max-steps 8 --out-dir project/evolve/experiments/rc2/out/rc2_candidate_runs/nat_defs_medium

# nat_defs_large_v5 (canonical_floor)
/opt/anaconda3/bin/python3 eval_rollout_all.py --theorem-set nat_defs_large_v5 --policy-type hybrid_evolved --route-config project/evolve/routing/ns24_router.json --strategy-config project/evolve/experiments/rc2/rc2_candidate_wrapper.json --top-k 8 --max-steps 8 --out-dir project/evolve/experiments/rc2/out/rc2_candidate_runs/nat_defs_large_v5

# set_ite_known_wins (candidate_validation)
/opt/anaconda3/bin/python3 scripts/sf1_run_eval.py --theorem-set-file project/evolve/experiments/rc2_candidates/set_ite_simp/theorem_sets/set_ite_known_wins.json --register-name rc2bench_set_ite_known_wins -- --policy-type hybrid_evolved --route-config project/evolve/routing/ns24_router.json --strategy-config project/evolve/experiments/rc2/rc2_candidate_wrapper.json --top-k 8 --max-steps 8 --out-dir project/evolve/experiments/rc2/out/rc2_candidate_runs/set_ite_known_wins

# set_ite_selected_failures (candidate_validation)
/opt/anaconda3/bin/python3 scripts/sf1_run_eval.py --theorem-set-file project/evolve/experiments/rc2_candidates/set_ite_simp/theorem_sets/set_ite_selected_failures.json --register-name rc2bench_set_ite_selected_failures -- --policy-type hybrid_evolved --route-config project/evolve/routing/ns24_router.json --strategy-config project/evolve/experiments/rc2/rc2_candidate_wrapper.json --top-k 8 --max-steps 8 --out-dir project/evolve/experiments/rc2/out/rc2_candidate_runs/set_ite_selected_failures

# set_ite_fresh_holdout (candidate_validation)
/opt/anaconda3/bin/python3 scripts/sf1_run_eval.py --theorem-set-file project/evolve/experiments/rc2_candidates/set_ite_simp/theorem_sets/set_ite_fresh_holdout.json --register-name rc2bench_set_ite_fresh_holdout -- --policy-type hybrid_evolved --route-config project/evolve/routing/ns24_router.json --strategy-config project/evolve/experiments/rc2/rc2_candidate_wrapper.json --top-k 8 --max-steps 8 --out-dir project/evolve/experiments/rc2/out/rc2_candidate_runs/set_ite_fresh_holdout

# sf1_frontier_runnable_subset (fresh_frontier)
/opt/anaconda3/bin/python3 scripts/sf1_run_eval.py --theorem-set-file project/evolve/experiments/sf1/theorem_sets/sf1_frontier_runnable_subset.json --register-name rc2bench_sf1_frontier_runnable_subset -- --policy-type hybrid_evolved --route-config project/evolve/routing/ns24_router.json --strategy-config project/evolve/experiments/rc2/rc2_candidate_wrapper.json --top-k 8 --max-steps 8 --out-dir project/evolve/experiments/rc2/out/rc2_candidate_runs/sf1_frontier_runnable_subset

# set_ite_negative_controls (negative_control)
/opt/anaconda3/bin/python3 scripts/sf1_run_eval.py --theorem-set-file project/evolve/experiments/rc2_candidates/set_ite_simp/theorem_sets/set_ite_negative_controls.json --register-name rc2bench_set_ite_negative_controls -- --policy-type hybrid_evolved --route-config project/evolve/routing/ns24_router.json --strategy-config project/evolve/experiments/rc2/rc2_candidate_wrapper.json --top-k 8 --max-steps 8 --out-dir project/evolve/experiments/rc2/out/rc2_candidate_runs/set_ite_negative_controls
