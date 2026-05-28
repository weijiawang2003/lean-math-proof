#!/usr/bin/env bash
# CX1 Stage 5 — eval matrix driver.
#
# For each CX1 set: raw NS15 routed, NS9 wrapper, and the NS19
# finset_aesop_only variant where the set contains Finset thms.
set -euo pipefail

E="bash scripts/cx1_run_eval.sh"

# Limited probe per CX1 spec: focus on the four sets most likely to
# surface new wrapper-only signal. The two stratified mixed_* sets
# are deferred — they cover ground the four target sets already touch.

# Stage A — Finset image/filter (directly addresses NS20's gap).
$E raw     none cx1_finset_image_filter
$E ns9wrap none cx1_finset_image_filter
$E variant finset_aesop_only cx1_finset_image_filter

# Stage B — Nat gcd/dvd/mod (fresh Nat surface beyond Nat/Defs.lean).
$E raw     none cx1_nat_gcd_dvd_mod
$E ns9wrap none cx1_nat_gcd_dvd_mod

# Stage C — Bool/Option/Int (entirely new namespaces; high chance of
# new patterns since the routed base model has never seen them).
$E raw     none cx1_bool_option_int
$E ns9wrap none cx1_bool_option_int

# Stage D — List/Multiset bulk (List was 13 thms; now 100 fresh).
$E raw     none cx1_list_multiset
$E ns9wrap none cx1_list_multiset

echo "[done] CX1 matrix complete"
