# REL1 — reproduction commands

All commands run from the repo root. RC1 is a *composed* config (NS9 ⊕ WX3
Multiset ⊕ MX2 narrow Set.Finite); the benchmark is derived from already-mined
arc traces plus a small live confirmation — no new full sweep is required.

## Paths

- RC1 config: `project/evolve/experiments/rc1/rc1_production_wrapper.json`
- Router: `project/evolve/routing/ns24_router.json`
- Final report: `project/evolve/reports/rc1_final_project_report.md`

## RC1 benchmark (composed; reads arc traces)

```bash
python3 scripts/rc1_compose_benchmark.py
# writes:
#   project/data/rc1_full_benchmark_meta.json
#   project/data/rc1_component_ablation_meta.json
#   project/evolve/reports/rc1_component_ablation.md
# expected: RC1 +15 vs NS9 (106 -> 121), 0 regressions
```

## RC1 preservation check (static gate logic)

```bash
python3 scripts/rc1_preservation_check.py
# writes project/data/rc1_preservation_meta.json
# expected: 0 off-gate Multiset emissions, 0 off-gate aesop emissions,
#           predictor=False, sequence=False
```

## Component ablation

The ablation is produced by `scripts/rc1_compose_benchmark.py` (above):
`project/evolve/reports/rc1_component_ablation.md`.
- expected: WX3 +12 (Multiset), MX2 +3 (Set.Finite), RC1 total +15, no
  negative interaction.

## Live RC1 confirmation (optional, ~1-2 min)

```bash
python3 scripts/run_with_timeout.py 300 python3 eval_rollout_all.py \
  --theorem-set mx2_set_aesop_known \
  --policy-type hybrid_evolved \
  --route-config project/evolve/routing/ns24_router.json \
  --strategy-config project/evolve/experiments/rc1/rc1_production_wrapper.json \
  --top-k 8 --max-steps 8 \
  --out-dir project/evolve/eval_runs/rc1_C_mx2_set_aesop_known
# expected: closes Set.Finite.toFinset_insert and Set.Finite.toFinset_offDiag via aesop
```

## Run RC1 on a Multiset / other surface

Swap `--theorem-set` (e.g. `ax4_multiset_induction_heldout`,
`ax4_multiset_induction_heldout2`); the Multiset induction oracle fires on
`Multiset.*` and aesop on `Set.Finite.`/`Set.toFinset` only.

## Final reports

- `project/evolve/reports/rc1_final_project_report.md` — full arc + lessons.
- `project/evolve/reports/rel1_executive_summary.md` — one-page summary.
- `project/evolve/reports/rc1_preservation_report.md` — preservation proof.
- `project/evolve/reports/rel1_release_checklist.md` — release checklist.
