#!/usr/bin/env bash
# NS24 — train Int minimal-omega aggregate checkpoints.
#
# Trains from the NS23-repaired minimal-tactic labels (omega aggregate).
# Mirrors the NS22 training recipe: 3 epochs, batch 8, lr 5e-5, CodeT5
# tokenizer inherited from the init checkpoint.
#
# Variants (see scripts/build_ns24_training_data.py):
#   omega_5x              init ns22  primary
#   omega_10x             init ns22  stronger oversample
#   plus_constructor_5x   init ns22  exploratory (+1 constructor<;>omega row)
#   from_ns12             init ns12  ablation (same data as omega_5x)
#
# Usage: bash scripts/ns24_train.sh <variant>   (or "all")
set -euo pipefail

EPOCHS=3
BS=8
LR=5e-5
SEED=42
NS22=project/models/gen_v5_ns22_int_fallback_omega_5x
NS12=project/models/gen_v5_ns12_balanced

train_one() {
  local data="$1"; local init="$2"; local out="$3"
  if [[ -f "$out/model.safetensors" ]]; then
    echo "[skip] $out already trained"
    return 0
  fi
  if [[ ! -f "$data" ]]; then
    echo "[err] missing dataset $data — run scripts/build_ns24_training_data.py" >&2
    exit 1
  fi
  echo "[train] $out  (init=$init, data=$data)"
  python3 train_tactic_generator.py \
    --data "$data" \
    --model "$init" \
    --output-dir "$out" \
    --epochs "$EPOCHS" --batch-size "$BS" --lr "$LR" --seed "$SEED" \
    > "${out}_training.log" 2>&1
  echo "[done] $out"
}

VARIANT="${1:-all}"
run_variant() {
  case "$1" in
    omega_5x)
      train_one project/data/ns24_int_minimal_omega_5x.jsonl "$NS22" \
        project/models/gen_v5_ns24_int_minimal_omega_5x ;;
    omega_10x)
      train_one project/data/ns24_int_minimal_omega_10x.jsonl "$NS22" \
        project/models/gen_v5_ns24_int_minimal_omega_10x ;;
    plus_constructor_5x)
      train_one project/data/ns24_int_minimal_omega_plus_constructor_5x.jsonl "$NS22" \
        project/models/gen_v5_ns24_int_minimal_omega_plus_constructor_5x ;;
    from_ns12)
      train_one project/data/ns24_int_minimal_omega_5x.jsonl "$NS12" \
        project/models/gen_v5_ns24_int_minimal_omega_5x_from_ns12 ;;
    *) echo "unknown variant $1" >&2; exit 1 ;;
  esac
}

if [[ "$VARIANT" == "all" ]]; then
  for v in omega_5x omega_10x plus_constructor_5x from_ns12; do run_variant "$v"; done
else
  run_variant "$VARIANT"
fi
echo "[ns24_train] complete"
