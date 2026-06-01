#!/usr/bin/env python3
"""RC2 Part 2 — build SET_ITE_SIMP validation theorem sets from SX2 artifacts.

Emits five theorem-set files + a manifest under
project/evolve/experiments/rc2_candidates/set_ite_simp/theorem_sets/:

  1. set_ite_known_wins.json      the 5 SX2 TRUE_SET2_WIN theorems (2 selected + 3 holdout)
  2. set_ite_selected_failures.json  the 12 SF2 deep-dive selected Set failures
  3. set_ite_fresh_holdout.json   the SX2 holdout Set theorems (paths, not in mining)
  4. set_ite_negative_controls.json  Nat/Multiset/non-Set rows (gate must NOT fire)
  5. set_ite_canonical_smoke.json  small canonical non-Set sample (preservation)

Each row: {file_path, full_name, namespace, source, validation_role}.
Data-leakage notes: known_wins ⊂ (selected ∪ holdout); fresh_holdout excludes the
12 selected and (by SX2 construction) the 3 SF2 deferrals; known_wins are NOT
excluded from fresh_holdout's parent (they came FROM holdout) — flagged per row.
"""
from __future__ import annotations

import argparse
import json
import os

KNOWN_WINS = {
    "Set.ite_empty_right": "selected_reproduction",
    "Set.ite_right": "selected_reproduction",
    "Set.ite_empty": "fresh_holdout_win",
    "Set.ite_empty_left": "fresh_holdout_win",
    "Set.ite_left": "fresh_holdout_win",
}

# Negative controls + canonical smoke: synthetic non-Set rows. file_path=null means
# "gate-only / not live-runnable here"; the off-gate scan treats them as dry checks.
NEGATIVE_CONTROLS = [
    {"full_name": "Nat.add_comm", "namespace": "Nat", "file_path": None},
    {"full_name": "Nat.mul_succ", "namespace": "Nat", "file_path": None},
    {"full_name": "Int.add_mul", "namespace": "Int", "file_path": None},
    {"full_name": "Multiset.toFinset_eq_singleton_iff", "namespace": "Multiset",
     "file_path": "Mathlib/Data/Finset/Basic.lean"},
    {"full_name": "Multiset.cons_inj_left", "namespace": "Multiset", "file_path": None},
]
CANONICAL_SMOKE = [
    {"full_name": "Nat.add_zero", "namespace": "Nat", "file_path": None, "surface": "demo_v1"},
    {"full_name": "Nat.zero_add", "namespace": "Nat", "file_path": None, "surface": "demo_v1"},
    {"full_name": "Nat.succ_le_succ", "namespace": "Nat", "file_path": None, "surface": "nat_defs_medium"},
    {"full_name": "Bool.and_self", "namespace": "Bool", "file_path": None, "surface": "demo_v1"},
    {"full_name": "List.append_nil", "namespace": "List", "file_path": None, "surface": "demo_v1"},
]


def _row(full_name, file_path, namespace, source, role, extra=None):
    r = {"file_path": file_path, "full_name": full_name,
         "namespace": namespace or (full_name.split(".")[0] if "." in full_name else ""),
         "source": source, "validation_role": role}
    if extra:
        r.update(extra)
    return r


def _dump(path, name, rows):
    json.dump({name: rows}, open(path, "w"), ensure_ascii=False, indent=2)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sx2-selected",
                   default="project/evolve/experiments/sx2/out/set2_selected_eval_results.json")
    p.add_argument("--sx2-holdout",
                   default="project/evolve/experiments/sx2/out/set2_holdout_eval_results.json")
    p.add_argument("--sx2-holdout-cases",
                   default="project/evolve/experiments/sx2/out/set2_holdout_cases.json")
    p.add_argument("--selected-cases",
                   default="project/evolve/experiments/sf2/out/set_cluster_deep_dive/selected_cases.json")
    p.add_argument("--frontier",
                   default="project/evolve/experiments/sf1/out/real/frontier_with_paths.jsonl")
    p.add_argument("--out-dir",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/theorem_sets")
    p.add_argument("--manifest",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/validation_manifest.json")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    sel = json.load(open(args.sx2_selected))["results"]
    hold = json.load(open(args.sx2_holdout))["results"]
    sel_by = {r["full_name"]: r for r in sel}
    hold_by = {r["full_name"]: r for r in hold}
    # file_path lookup
    fp = {}
    for r in sel + hold:
        if r.get("file_path"):
            fp[r["full_name"]] = r["file_path"]

    # 1. known wins
    kw = []
    for fn, kind in KNOWN_WINS.items():
        kw.append(_row(fn, fp.get(fn, "Mathlib/Data/Set/Basic.lean"), "Set",
                       "sx2_true_set2_win", "known_win", {"win_kind": kind}))
    _dump(os.path.join(args.out_dir, "set_ite_known_wins.json"),
          "set_ite_known_wins", kw)

    # 2. selected failures (the 12)
    sf = []
    for r in json.load(open(args.selected_cases))["selected"]:
        sf.append(_row(r["full_name"], r["file_path"], "Set",
                       "sf2_set_cluster_deep_dive", "selected_failure",
                       {"primary_goal_shape": r.get("primary_goal_shape")}))
    _dump(os.path.join(args.out_dir, "set_ite_selected_failures.json"),
          "set_ite_selected_failures", sf)

    # 3. fresh holdout (the 20 SX2 holdout cases; flag the 3 that became known wins)
    fh = []
    for r in json.load(open(args.sx2_holdout_cases))["cases"]:
        fn = r["full_name"]
        fh.append(_row(fn, r["file_path"], r.get("namespace", "Set"),
                       "sx2_holdout", "fresh_holdout",
                       {"ite_shaped": r.get("ite_shaped"),
                        "is_known_win": fn in KNOWN_WINS}))
    _dump(os.path.join(args.out_dir, "set_ite_fresh_holdout.json"),
          "set_ite_fresh_holdout", fh)

    # 4. negative controls
    nc = [_row(r["full_name"], r.get("file_path"), r["namespace"],
               "synthetic_negative_control", "negative_control") for r in NEGATIVE_CONTROLS]
    _dump(os.path.join(args.out_dir, "set_ite_negative_controls.json"),
          "set_ite_negative_controls", nc)

    # 5. canonical smoke
    cs = [_row(r["full_name"], r.get("file_path"), r["namespace"],
               r.get("surface", "canonical"), "canonical",
               {"surface": r.get("surface")}) for r in CANONICAL_SMOKE]
    _dump(os.path.join(args.out_dir, "set_ite_canonical_smoke.json"),
          "set_ite_canonical_smoke", cs)

    manifest = {
        "candidate": "SET_ITE_SIMP (simp [Set.ite])",
        "sets": {
            "set_ite_known_wins": {"size": len(kw), "live_runnable": sum(1 for r in kw if r["file_path"]),
                                   "role": "the 5 SX2 TRUE_SET2_WIN theorems"},
            "set_ite_selected_failures": {"size": len(sf), "live_runnable": sum(1 for r in sf if r["file_path"]),
                                          "role": "12 SF2 deep-dive selected Set failures"},
            "set_ite_fresh_holdout": {"size": len(fh), "live_runnable": sum(1 for r in fh if r["file_path"]),
                                      "role": "20 SX2 holdout Set theorems (paths)"},
            "set_ite_negative_controls": {"size": len(nc), "live_runnable": sum(1 for r in nc if r["file_path"]),
                                          "role": "Nat/Int/Multiset; gate must NOT fire"},
            "set_ite_canonical_smoke": {"size": len(cs), "live_runnable": sum(1 for r in cs if r["file_path"]),
                                        "role": "canonical non-Set preservation sample"},
        },
        "data_leakage_notes": [
            "known_wins (5) is a SUBSET of selected_failures (2: ite_empty_right, ite_right) "
            "∪ fresh_holdout (3: ite_empty, ite_empty_left, ite_left).",
            "fresh_holdout excludes the 12 selected failures and the 3 SF2 deferrals "
            "(SX2 construction); it DOES contain the 3 holdout known-wins (flagged is_known_win).",
            "negative_controls / canonical_smoke are non-Set; the SET_ITE_SIMP gate "
            "cannot fire on them (requires 'Set' in name) — dry off-gate checks.",
            "literal RC1 is the unmodified rc1_production_wrapper.json; the candidate is "
            "additive (simp [Set.ite] only on RC1-failed gate-fired theorems).",
        ],
        "theorem_sets_dir": args.out_dir,
    }
    json.dump(manifest, open(args.manifest, "w"), ensure_ascii=False, indent=2)
    print(f"[rc2:prep] known_wins={len(kw)} selected={len(sf)} holdout={len(fh)} "
          f"neg={len(nc)} canonical={len(cs)} -> {args.out_dir}")
    print(f"[rc2:prep] manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
