# RC2 Candidate — Reproduction Commands (DRAFT)

All commands are non-invasive; RC1 (`rc1_production_wrapper.json`) and the NS24 router
are never modified. No commit. macOS has no `timeout` → use `scripts/run_with_timeout.py`.

## 1. Compose the RC2 candidate wrapper (RC1 ⊕ SET_ITE_SIMP, priority_templates slot)
```
python3 scripts/rc2_compose_candidate.py \
  --rc1-wrapper project/evolve/experiments/rc1/rc1_production_wrapper.json \
  --set-ite-policy project/evolve/experiments/rc2_candidates/set_ite_simp/set_ite_simp_gate_policy.json \
  --out-wrapper project/evolve/experiments/rc2/rc2_candidate_wrapper.json \
  --out-summary project/evolve/experiments/rc2/rc2_component_summary.json \
  --emit-slot priority_any
```

## 2. Build benchmark manifest
```
python3 scripts/rc2_build_benchmark_manifest.py \
  --out project/evolve/experiments/rc2/rc2_benchmark_manifest.json
```

## 3. Literal RC1 baseline (reuses prior literal-RC1 for the candidate Set sets)
```
python3 scripts/rc2_run_benchmark.py --manifest .../rc2_benchmark_manifest.json \
  --policy rc1 --route-config project/evolve/routing/ns24_router.json \
  --strategy-config project/evolve/experiments/rc1/rc1_production_wrapper.json \
  --reuse project/evolve/experiments/rc2/out/_rc1_reuse_seed.json \
  --out project/evolve/experiments/rc2/out/rc1_baseline_results.json
```

## 4. RC2 candidate (full_wrapper_eval)
```
python3 scripts/rc2_run_benchmark.py --manifest .../rc2_benchmark_manifest.json \
  --policy rc2_candidate --route-config project/evolve/routing/ns24_router.json \
  --strategy-config project/evolve/experiments/rc2/rc2_candidate_wrapper.json \
  --out project/evolve/experiments/rc2/out/rc2_candidate_results.json
```

## 5. Compare, minimal relabel, forensics, variants, ledger, preservation
```
python3 scripts/rc2_compare_results.py --rc1 .../rc1_baseline_results.json \
  --rc2 .../rc2_candidate_results.json --manifest .../rc2_benchmark_manifest.json \
  --out-json .../rc2_comparison.json --out-md .../rc2_comparison.md

python3 scripts/rc2_minimal_relabel.py --comparison .../rc2_comparison.json \
  --out-json .../rc2_minimal_relabel_results.json --out-md .../rc2_minimal_relabel_results.md

python3 scripts/rc2_forensic_trace_compare.py --rc1-results .../rc1_baseline_results.json \
  --rc2-results .../rc2_candidate_results.json --comparison .../rc2_comparison.json \
  --out-json .../rc2_hardening/out/perturbation_forensics.json --out-md ...

python3 scripts/rc2_test_integration_variants.py \
  --base-wrapper project/evolve/experiments/rc1/rc1_production_wrapper.json \
  --set-ite-policy project/evolve/experiments/rc2/rc2_set_ite_simp_gate.json \
  --manifest .../rc2_benchmark_manifest.json --out-dir .../rc2_hardening/out/variants \
  --out-json .../variant_comparison.json --out-md .../variant_comparison.md

python3 scripts/rc2_build_delta_ledger.py --comparison .../rc2_comparison.json \
  --minimal-relabel .../rc2_minimal_relabel_results.json \
  --forensics .../perturbation_forensics.json \
  --out-json .../rc2_delta_ledger.json --out-md .../rc2_delta_ledger.md

python3 scripts/rc2_preservation_hardening.py \
  --candidate-wrapper project/evolve/experiments/rc2/rc2_candidate_wrapper.json \
  --manifest .../rc2_benchmark_manifest.json \
  --out-json .../preservation_hardening.json --out-md .../preservation_hardening.md
```

## 6. Protected-file confirmation
```
git diff --stat HEAD -- project/evolve/experiments/rc1/rc1_production_wrapper.json \
  project/evolve/routing/ns24_router.json   # expect empty
git status --short
```

Expected: RC1 0/5 known + 0/12 selected + 11/20 holdout; RC2 5/5 + 4/12 + 15/20;
credited delta +5; 0 regressions; 0 off-gate; floors 11/15·37/38·49/65; deterministic.
