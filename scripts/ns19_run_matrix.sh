#!/usr/bin/env bash
# NS19 matrix driver — run the eval plan for both target families.
#
# Plan:
#   For the new Finset surface:
#       raw, ns9wrap, finset_aesop_only
#   For the new Nat-arith replay surface:
#       raw, ns9wrap, nat_simp_arith_targeted
#   For preservation / regression-elimination:
#       finset_aesop_only on ns17_finset_extra, ns17_set_extra,
#       nat_defs_medium, demo_v1
#
# Usage: bash scripts/ns19_run_matrix.sh
set -euo pipefail

E="bash scripts/ns19_run_eval.sh"

# --- Finset aesop surface (new) ---
$E raw      none ns19_finset_aesop_surface
$E ns9wrap  none ns19_finset_aesop_surface
$E variant  finset_aesop_only ns19_finset_aesop_surface

# --- Nat-arith replay surface (existing Nat thms not yet wrapper-only-proved) ---
$E raw      none ns19_nat_simp_arith_replay
$E ns9wrap  none ns19_nat_simp_arith_replay
$E variant  nat_simp_arith_targeted ns19_nat_simp_arith_replay

# --- Preservation / regression-elimination for finset_aesop_only ---
$E variant  finset_aesop_only ns17_finset_extra
$E variant  finset_aesop_only ns17_set_extra
$E variant  finset_aesop_only nat_defs_medium
$E variant  finset_aesop_only demo_v1
$E variant  finset_aesop_only ns14_set_finset_extra

# --- Preservation for nat_simp_arith_targeted ---
$E variant  nat_simp_arith_targeted nat_defs_medium
$E variant  nat_simp_arith_targeted demo_v1
$E variant  nat_simp_arith_targeted ns16_nat_div_mod_extra

echo "[done] NS19 matrix complete"
