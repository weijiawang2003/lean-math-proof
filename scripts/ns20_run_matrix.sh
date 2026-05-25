#!/usr/bin/env bash
# NS20 matrix driver — final Finset/aesop mining pass.
#
# For each new NS20 set:
#   raw NS15 routed
#   NS9 wrapper + NS15 routed
#   ns19_finset_aesop_only + NS15 routed
#
# Plus cross-domain preservation re-evals for finset_aesop_only
# on existing benchmarks (the NS19 variant evals on these are
# already cached and will [skip] under the ns20_ tag — so we re-
# run them under ns20_ to keep the NS20 comparison self-contained,
# unless they're already done.
set -euo pipefail

E="bash scripts/ns20_run_eval.sh"

# --- New surfaces ---
for S in ns20_finset_aesop_extra_easy ns20_finset_aesop_extra_medium ns20_finset_aesop_extra_hard; do
  $E raw     none "$S"
  $E ns9wrap none "$S"
  $E variant finset_aesop_only "$S"
done

# --- Preservation eval missing from NS19: nat_defs_large_v5 ---
# (NS19 already covered demo_v1, nat_defs_medium,
# ns14_set_finset_extra, ns17_set_extra, ns17_finset_extra. The
# Set-regression preservation evals re-use those NS19 results.)
$E variant finset_aesop_only nat_defs_large_v5

echo "[done] NS20 matrix complete"
