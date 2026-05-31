#!/usr/bin/env bash
# Part 5 — broader RC1 frontier eval (as run). Protected configs untouched.
python3 scripts/sf1_eval_matrix.py --run-real --policies rc1 \
  --out-dir project/evolve/experiments/sf2/out/frontier_expansion/eval \
  --max-batches 3 --max-theorems-per-batch 20 --top-k 8 --max-steps 8 --timeout 2800
