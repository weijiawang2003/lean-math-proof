#!/usr/bin/env bash
# Literal RC3 candidate (RC2 ⊕ SX3_SET_ITE_AESOP) through the production eval harness.
set -uo pipefail
cd /Users/weijiawang/dev/dojo_sandbox
/opt/anaconda3/bin/python3 scripts/rc3_run_literal_validation.py \
  --manifest project/evolve/experiments/rc3_validation/validation_manifest.json \
  --policy rc3_candidate \
  --strategy-config project/evolve/experiments/rc3_validation/rc3_candidate_wrapper.json \
  --route-config project/evolve/routing/ns24_router.json \
  --top-k 8 --max-steps 8 \
  --out-dir project/evolve/experiments/rc3_validation/out/literal_rc3_runs \
  --out-json project/evolve/experiments/rc3_validation/out/literal_rc3_results.json
