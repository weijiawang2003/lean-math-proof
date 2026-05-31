#!/usr/bin/env bash
# Literal RC2 baseline through the production eval harness. RC1/RC2-release/NS24 untouched.
set -uo pipefail
cd /Users/weijiawang/dev/dojo_sandbox
/opt/anaconda3/bin/python3 scripts/rc3_run_literal_validation.py \
  --manifest project/evolve/experiments/rc3_validation/validation_manifest.json \
  --policy rc2 \
  --strategy-config project/evolve/experiments/rc2_release/rc2_production_wrapper.json \
  --route-config project/evolve/routing/ns24_router.json \
  --top-k 8 --max-steps 8 \
  --out-dir project/evolve/experiments/rc3_validation/out/literal_rc2_runs \
  --out-json project/evolve/experiments/rc3_validation/out/literal_rc2_results.json
