# RC2 Reproduction Commands

RC1 (`rc1_production_wrapper.json`) and the NS24 router are never modified.
macOS has no `timeout` → use `scripts/run_with_timeout.py`. Authoritative solved
flag = per-theorem `finished` in each run's `metrics.json`.

## RC2 production eval (registered theorem set)
```
python3 eval_rollout_all.py --theorem-set <set> \
  --policy-type hybrid_evolved \
  --route-config project/evolve/routing/ns24_router.json \
  --strategy-config project/evolve/experiments/rc2_release/rc2_production_wrapper.json \
  --top-k 8 --max-steps 8 --out-dir <run-dir>
```

## RC2 production eval (file-based theorem set, runtime-registered)
```
python3 scripts/sf1_run_eval.py \
  --theorem-set-file <set>.json --register-name <name> \
  -- --policy-type hybrid_evolved \
     --route-config project/evolve/routing/ns24_router.json \
     --strategy-config project/evolve/experiments/rc2_release/rc2_production_wrapper.json \
     --top-k 8 --max-steps 8 --out-dir <run-dir>
```

## Canonical preservation floors
```
for S in demo_v1 nat_defs_medium nat_defs_large_v5; do
  python3 eval_rollout_all.py --theorem-set $S --policy-type hybrid_evolved \
    --route-config project/evolve/routing/ns24_router.json \
    --strategy-config project/evolve/experiments/rc2_release/rc2_production_wrapper.json \
    --top-k 8 --max-steps 8 --out-dir runs/rc2_$S
done
```
Expected: demo_v1 ≥ 11/15, nat_defs_medium ≥ 37/38, nat_defs_large_v5 ≥ 49/65.

## RC1 vs RC2 benchmark + attribution (helper scripts)
```
python3 scripts/rc2_run_benchmark.py --manifest project/evolve/experiments/rc2/rc2_benchmark_manifest.json \
  --policy rc1 --strategy-config project/evolve/experiments/rc1/rc1_production_wrapper.json \
  --route-config project/evolve/routing/ns24_router.json --out <rc1_results.json>
python3 scripts/rc2_run_benchmark.py --manifest .../rc2_benchmark_manifest.json \
  --policy rc2_candidate --strategy-config project/evolve/experiments/rc2_release/rc2_production_wrapper.json \
  --route-config project/evolve/routing/ns24_router.json --out <rc2_results.json>
python3 scripts/rc2_compare_results.py --rc1 <rc1_results.json> --rc2 <rc2_results.json> \
  --manifest .../rc2_benchmark_manifest.json --out-json <comparison.json> --out-md <comparison.md>
python3 scripts/rc2_minimal_relabel.py --comparison <comparison.json> \
  --out-json <relabel.json> --out-md <relabel.md>
python3 scripts/rc2_check_determinism.py --manifest .../rc2_benchmark_manifest.json \
  --candidate-wrapper project/evolve/experiments/rc2_release/rc2_production_wrapper.json \
  --out <determinism.json>
```

## Protected-file confirmation
```
git diff --stat HEAD -- project/evolve/experiments/rc1/rc1_production_wrapper.json \
  project/evolve/routing/ns24_router.json   # expect empty
git status --short
```
