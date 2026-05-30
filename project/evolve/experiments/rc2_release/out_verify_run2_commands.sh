#!/usr/bin/env bash
# RC2 benchmark — policy=rc2_candidate. RC1/NS24 untouched.
set -uo pipefail
cd /Users/weijiawang/dev/dojo_sandbox

# set_ite_known_wins (candidate_validation)
/opt/anaconda3/bin/python3 scripts/sf1_run_eval.py --theorem-set-file project/evolve/experiments/rc2_candidates/set_ite_simp/theorem_sets/set_ite_known_wins.json --register-name rc2bench_set_ite_known_wins -- --policy-type hybrid_evolved --route-config project/evolve/routing/ns24_router.json --strategy-config project/evolve/experiments/rc2_release/rc2_production_wrapper.json --top-k 8 --max-steps 8 --out-dir project/evolve/experiments/rc2_release/rc2_candidate_runs/set_ite_known_wins
