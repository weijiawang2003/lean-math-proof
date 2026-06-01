#!/usr/bin/env python3
"""RC2 Part 2 — build the RC2 benchmark manifest.

Surfaces:
  canonical_floor      demo_v1, nat_defs_medium, nat_defs_large_v5 (registered names)
  candidate_validation set_ite_known_wins / selected_failures / fresh_holdout (files)
  fresh_frontier       sf1_frontier_runnable_subset (file)
  negative_control     set_ite_negative_controls (file)

Each surface: name, role, kind (registered|file), path_or_registered_name, size,
contains_set_ite (whether RC2 can differ from RC1), expected_rc1, expected_rc2_delta.
"""
from __future__ import annotations

import argparse
import json
import os

TS = "project/evolve/experiments/rc2_candidates/set_ite_simp/theorem_sets"
SF1_TS = "project/evolve/experiments/sf1/theorem_sets"


def _file_rows(path):
    o = json.load(open(path))
    rows = list(o.values())[0] if isinstance(o, dict) else o
    return rows


def _count_set_ite(rows):
    return sum(1 for r in rows
               if str(r.get("full_name") or r.get("name") or "").startswith("Set.ite"))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out",
                   default="project/evolve/experiments/rc2/rc2_benchmark_manifest.json")
    args = p.parse_args(argv)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import eval_rollout_all as E

    surfaces = []

    # canonical floors (registered names)
    floors = {"demo_v1": ">=11/15", "nat_defs_medium": ">=37/38",
              "nat_defs_large_v5": ">=49/65"}
    for name, exp in floors.items():
        try:
            size = len(E.get_theorems(name))
        except Exception:
            size = None
        surfaces.append({
            "name": name, "role": "canonical_floor", "kind": "registered",
            "path_or_registered_name": name, "size": size,
            "contains_set_ite": False,
            "expected_rc1": exp,
            "expected_rc2_delta": "0 (no Set.ite names; RC2==RC1 by construction)"})

    # candidate validation sets (files)
    cand = {
        "set_ite_known_wins": ("candidate_validation", "0/5", "+5 (all SET_ITE wins)"),
        "set_ite_selected_failures": ("candidate_validation", "0/12", "+2 (ite_empty_right, ite_right)"),
        "set_ite_fresh_holdout": ("candidate_validation", "11/20", "+3 (ite_empty, ite_empty_left, ite_left)"),
    }
    for nm, (role, erc1, edelta) in cand.items():
        path = os.path.join(TS, f"{nm}.json")
        rows = _file_rows(path)
        surfaces.append({
            "name": nm, "role": role, "kind": "file",
            "path_or_registered_name": path, "size": len(rows),
            "contains_set_ite": _count_set_ite(rows) > 0,
            "num_set_ite": _count_set_ite(rows),
            "expected_rc1": erc1, "expected_rc2_delta": edelta})

    # SF1 fresh frontier subset (file)
    fr = os.path.join(SF1_TS, "sf1_frontier_runnable_subset.json")
    if os.path.exists(fr):
        rows = _file_rows(fr)
        surfaces.append({
            "name": "sf1_frontier_runnable_subset", "role": "fresh_frontier",
            "kind": "file", "path_or_registered_name": fr, "size": len(rows),
            "contains_set_ite": _count_set_ite(rows) > 0,
            "num_set_ite": _count_set_ite(rows),
            "expected_rc1": "unknown (live)",
            "expected_rc2_delta": "Set.ite-shaped frontier wins if any"})

    # negative controls (file; mostly non-live, dry off-gate check)
    nc = os.path.join(TS, "set_ite_negative_controls.json")
    if os.path.exists(nc):
        rows = _file_rows(nc)
        surfaces.append({
            "name": "set_ite_negative_controls", "role": "negative_control",
            "kind": "file", "path_or_registered_name": nc,
            "size": len(rows), "live_runnable": sum(1 for r in rows if r.get("file_path")),
            "contains_set_ite": _count_set_ite(rows) > 0,
            "expected_rc1": "n/a", "expected_rc2_delta": "0 (gate must NOT fire; off-gate=0)"})

    manifest = {
        "candidate": "RC2 = RC1 ⊕ SET_ITE_SIMP (simp [Set.ite])",
        "rc2_wrapper": "project/evolve/experiments/rc2/rc2_candidate_wrapper.json",
        "rc1_command": {"policy_type": "hybrid_evolved",
                        "route_config": "project/evolve/routing/ns24_router.json",
                        "strategy_config": "project/evolve/experiments/rc1/rc1_production_wrapper.json",
                        "top_k": 8, "max_steps": 8},
        "reuse_note": "literal RC1 for known_wins/selected_failures/fresh_holdout may be "
                      "reused from project/evolve/experiments/rc2_candidates/set_ite_simp/"
                      "out/literal_rc1_results.json (identical command, configs, sets, "
                      "repaired finished-key).",
        "by_construction_note": "RC2 differs from RC1 ONLY on theorems whose name starts "
                                "with 'Set.ite'. Canonical floors contain none -> RC2==RC1 "
                                "there; run both to confirm preservation empirically.",
        "num_surfaces": len(surfaces),
        "surfaces": surfaces,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(manifest, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[rc2:manifest] {len(surfaces)} surfaces -> {args.out}")
    for s in surfaces:
        print(f"   {s['name']:32s} {s['role']:20s} size={s['size']} "
              f"set_ite={s.get('contains_set_ite')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
