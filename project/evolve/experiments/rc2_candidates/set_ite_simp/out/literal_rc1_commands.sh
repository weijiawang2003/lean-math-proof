#!/usr/bin/env bash
# Literal RC1 validation — replayable commands. RC1/NS24 untouched.
set -uo pipefail
cd /Users/weijiawang/dev/dojo_sandbox

# set_ite_known_wins
/opt/anaconda3/bin/python3 scripts/sf1_run_eval.py --theorem-set-file project/evolve/experiments/rc2_candidates/set_ite_simp/theorem_sets/set_ite_known_wins.json --register-name rc2_set_ite_known_wins -- --policy-type hybrid_evolved --route-config project/evolve/routing/ns24_router.json --strategy-config project/evolve/experiments/rc1/rc1_production_wrapper.json --top-k 8 --max-steps 8 --out-dir project/evolve/experiments/rc2_candidates/set_ite_simp/out/literal_rc1/set_ite_known_wins

# set_ite_selected_failures
/opt/anaconda3/bin/python3 scripts/sf1_run_eval.py --theorem-set-file project/evolve/experiments/rc2_candidates/set_ite_simp/theorem_sets/set_ite_selected_failures.json --register-name rc2_set_ite_selected_failures -- --policy-type hybrid_evolved --route-config project/evolve/routing/ns24_router.json --strategy-config project/evolve/experiments/rc1/rc1_production_wrapper.json --top-k 8 --max-steps 8 --out-dir project/evolve/experiments/rc2_candidates/set_ite_simp/out/literal_rc1/set_ite_selected_failures

# set_ite_fresh_holdout
/opt/anaconda3/bin/python3 scripts/sf1_run_eval.py --theorem-set-file project/evolve/experiments/rc2_candidates/set_ite_simp/theorem_sets/set_ite_fresh_holdout.json --register-name rc2_set_ite_fresh_holdout -- --policy-type hybrid_evolved --route-config project/evolve/routing/ns24_router.json --strategy-config project/evolve/experiments/rc1/rc1_production_wrapper.json --top-k 8 --max-steps 8 --out-dir project/evolve/experiments/rc2_candidates/set_ite_simp/out/literal_rc1/set_ite_fresh_holdout
