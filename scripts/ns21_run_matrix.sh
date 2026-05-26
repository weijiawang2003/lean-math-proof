#!/usr/bin/env bash
# NS21 — eval matrix driver. Trimmed to the cells that genuinely inform
# the transfer-vs-memorization verdict and router selection.
set -euo pipefail

E="bash scripts/ns21_run_eval.sh"

# raw_ckpt per NS21 candidate on:
#  - Set/demo negative control + ns14_set_finset_extra (regression risk)
#  - Finset surfaces with pool & held-out coverage
CKPT_SETS=(
  demo_v1
  ns17_set_extra
  ns14_set_finset_extra
  ns17_finset_extra
  cx1_finset_image_filter
  ns20_finset_aesop_extra_medium
)

for ckpt in \
  gen_v5_ns21_finset_aesop_10x \
  gen_v5_ns21_finset_aesop_20x \
  gen_v5_ns21_finset_aesop_minimal
do
  for s in "${CKPT_SETS[@]}"; do
    $E raw_ckpt "$ckpt" "$s"
  done
done

# Baseline on the two new Finset surfaces (ns12_balanced hasn't been
# evaluated on cx1/ns20 sets). The existing pre-NS21 gen_v5_ns12_balanced
# runs cover demo_v1, ns17_set_extra, ns17_finset_extra under different
# tag names; ns21_compare_finset_transfer.py reads ns21_rawckpt_*, so
# also re-eval these under the canonical NS21 tag for apples-to-apples.
BASE_REPEAT=(
  demo_v1
  ns17_set_extra
  ns14_set_finset_extra
  ns17_finset_extra
  cx1_finset_image_filter
  ns20_finset_aesop_extra_medium
)
for s in "${BASE_REPEAT[@]}"; do
  $E raw_ckpt gen_v5_ns12_balanced "$s"
done

echo "[done] Stage 4 raw_ckpt matrix complete"

# Stage 5 — pick router and route raw eval.
python3 scripts/ns21_pick_router.py
ROUTED_SETS=(
  demo_v1
  ns14_set_finset_extra
  ns17_set_extra
  ns17_finset_extra
  nat_defs_medium
  nat_defs_large_v5
  cx1_finset_image_filter
  ns20_finset_aesop_extra_medium
)
for s in "${ROUTED_SETS[@]}"; do
  $E raw_routed ns21_router "$s"
done

echo "[done] Stage 5 routed raw matrix complete"

# Stage 6 — wrapper compat (NS9 wrap + NS21 router).
for s in nat_defs_medium nat_defs_large_v5 demo_v1 cx1_finset_image_filter; do
  $E wrap_routed ns21_router "$s"
done

echo "[done] NS21 matrix complete"
