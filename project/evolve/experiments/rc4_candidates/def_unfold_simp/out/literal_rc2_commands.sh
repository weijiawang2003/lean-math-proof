#!/bin/sh
# Literal RC2 baseline command (per chunk; identical config to SF4/TR2/TR3):
# strategy-config=project/evolve/experiments/rc2_release/rc2_production_wrapper.json route-config=project/evolve/routing/ns24_router.json
# policy-type=hybrid_evolved top-k=8 max-steps=8
# python3 eval_rollout_all.py --theorem-set <registered> --policy-type hybrid_evolved \
#   --route-config project/evolve/routing/ns24_router.json --strategy-config project/evolve/experiments/rc2_release/rc2_production_wrapper.json \
#   --top-k 8 --max-steps 8 --out-dir <out>
