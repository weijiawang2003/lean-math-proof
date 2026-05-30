# REL1 — release checklist

Packaging checklist for the RC1 production result. **No new experiments**; this
is reproduction/presentation only.

## Recommended config

- **Production deterministic wrapper:**
  `project/evolve/experiments/rc1/rc1_production_wrapper.json`
- Router: `project/evolve/routing/ns24_router.json` (unchanged)
- Base genome: `project/evolve/best/ns9_best_genome.json` (unchanged; RC1 is a
  separate composed config that deep-copies it)
- Experimental components are **off by default** inside the RC1 config:
  AX4 learned predictor (`symbolic_predictor.enabled = false`), SX1 sequence
  search (`symbolic_sequence_search.enabled = false`).

## Reproduction commands

```bash
# Re-derive the composed benchmark + ablation (reads already-mined arc traces;
# no new live sweep):
python3 scripts/rc1_compose_benchmark.py

# Static preservation / gate check (gate-logic over theorem-set names):
python3 scripts/rc1_preservation_check.py

# Live RC1 confirmation on a small Set.Finite set (optional; ~1-2 min):
python3 eval_rollout_all.py --theorem-set mx2_set_aesop_known \
  --policy-type hybrid_evolved \
  --route-config project/evolve/routing/ns24_router.json \
  --strategy-config project/evolve/experiments/rc1/rc1_production_wrapper.json \
  --top-k 8 --max-steps 8 --out-dir <run-dir>
```

(macOS has no coreutils `timeout`; wrap long live runs with
`python3 scripts/run_with_timeout.py <secs> <cmd>`.)

## Expected headline numbers

| metric | value |
|---|---|
| RC1 delta vs NS9 (measured surfaces) | **+15 wins** (106 → 121) |
| WX3 Multiset induction contribution | +12 |
| MX2 narrow Set.Finite aesop contribution | +3 |
| Regressions | **0** |
| Off-gate emissions (Multiset action / aesop) | **0 / 0** |
| Floors | demo 11/15, medium 37/38, large 49/65 (preserved) |
| Live RC1 confirmation | `Set.Finite.toFinset_insert` + `toFinset_offDiag` via aesop |

## Must NOT be committed

- Checkpoints / model artifacts (`project/models/`, `*.pt/*.ckpt/*.safetensors`)
- Eval run dirs and logs (`project/evolve/eval_runs/`, `*.log`)
- Raw traces and large JSONL (`*.jsonl` datasets)
- Population / training run dirs (`project/evolve/runs/`, `*_runs/`)

(All covered by `.gitignore`; verify with the sanity check below.)

## Sanity checks before sharing

```bash
# 1. NS9 genome untouched (no gates/symbolic blocks):
python3 -c "import json;g=json.load(open('project/evolve/best/ns9_best_genome.json'));print('NS9 clean:', 'theorem_name_tactic_gates' not in g and 'symbolic_actions' not in g)"

# 2. RC1 experimental components disabled:
python3 -c "import json;c=json.load(open('project/evolve/experiments/rc1/rc1_production_wrapper.json'));print('predictor off:',not c['symbolic_predictor']['enabled'],'| seq off:',not c['symbolic_sequence_search']['enabled'])"

# 3. No artifacts staged:
git diff --cached --name-only | grep -Ei '\.jsonl$|/models/|eval_runs/|\.log$|/runs/|\.ckpt|\.pt$' || echo "clean"

# 4. Working tree clean (excluding ignored eval_runs):
git status --porcelain | grep -vE 'eval_runs/' || echo "clean"
```

## Optional: tag the release

Not done automatically. To tag:

```bash
git tag -a rc1-production-stack -m "RC1 production wrapper: +15 vs NS9, 0 regressions"
```
