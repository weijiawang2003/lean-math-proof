#!/usr/bin/env bash
# NS18 matrix driver: per-variant set list.
# Usage: bash scripts/ns18_run_matrix.sh <variant>
set -euo pipefail

VARIANT="${1:?usage: $0 <variant>}"

# Targeted set list per variant. combined_safe runs the full matrix
# (preservation + every fresh surface); other variants run on their
# most-likely-to-help surfaces.
case "$VARIANT" in
  constructor_omega)
    SETS=(ns14_nat_extra ns16_nat_iff_extra nat_defs_medium)
    ;;
  split_ifs_omega)
    SETS=(ns16_nat_div_mod_extra ns16_nat_mixed_extra ns17_nat_remaining)
    ;;
  nat_simp_arith)
    SETS=(ns16_nat_div_mod_extra ns16_nat_mixed_extra ns16_nat_order_extra)
    ;;
  aesop_wrapper)
    SETS=(ns17_set_extra ns17_finset_extra ns17_list_multiset ns17_nat_remaining ns16_nat_mixed_extra)
    ;;
  bool_option_cases)
    SETS=(ns17_list_multiset ns17_set_extra ns17_finset_extra)
    ;;
  combined_safe)
    SETS=(
      nat_defs_medium nat_defs_large_v5
      ns14_nat_extra ns14_set_finset_extra
      ns16_nat_iff_extra ns16_nat_div_mod_extra
      ns16_nat_order_extra ns16_nat_mixed_extra
      ns17_set_extra ns17_finset_extra
      ns17_list_multiset ns17_nat_remaining
    )
    ;;
  *)
    echo "unknown variant: $VARIANT" >&2; exit 1 ;;
esac

for S in "${SETS[@]}"; do
  bash scripts/ns18_run_eval.sh "$VARIANT" "$S"
done
echo "[done] $VARIANT matrix complete"
