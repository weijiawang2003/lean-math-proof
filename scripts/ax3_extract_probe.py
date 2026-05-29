"""AX3 Stage 3 — extract raw vs NS9 vs WX3-induction signal on AX3 sets.

Reads the matrix from scripts/ax3_run_matrix_parallel.sh (ns24_router):
  raw    : ax3_raw_<set>
  ns9    : ax3_ns9_<set>
  wx3ind : ax3_wx3ind_<set>   (wx3_multiset_induction_safe.json)

Computes per set: raw/ns9/wx3 wins, WX3-only-beyond-NS9, symbolic-action
wins (origin wrapper_symbolic_action), regressions, and per-win
tactic/family/var. Output: project/data/ax3_multiset_mining_probe_meta.json
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

SETS = ["ax3_multiset_induction_mine", "ax3_multiset_induction_heldout",
        "ax3_multiset_mixed_heldout", "ax3_multiset_negative_control"]
OUT = "project/data/ax3_multiset_mining_probe_meta.json"


def first(pat):
    ms = sorted(glob.glob(pat))
    return ms[0] if ms else None


def metrics(tag, s):
    f = first(f"project/evolve/eval_runs/ax3_{tag}_{s}/eval-*/metrics.json")
    return json.load(open(f)) if f else None


def pt(d):
    return {t["full_name"]: t for t in d.get("per_theorem", [])} if d else {}


def wins(d):
    return {n for n, t in pt(d).items() if t.get("finished")}


def _var_of(tac):
    if not tac:
        return None
    toks = tac.split()
    return toks[1] if len(toks) > 1 and toks[0] == "induction" else None


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import tasks
    name_to_file = {}
    for _s, thms in tasks.THEOREM_SETS.items():
        for t in thms:
            name_to_file.setdefault(t.full_name, t.file_path)

    per_set = []
    wx3_only_all = []
    symbolic_all = []
    regr_all = []
    fam_counts = Counter()
    missing = []

    for s in SETS:
        raw_m, ns9_m, wx3_m = (metrics("raw", s), metrics("ns9", s),
                               metrics("wx3ind", s))
        for need, m in (("raw", raw_m), ("ns9", ns9_m), ("wx3ind", wx3_m)):
            if m is None:
                missing.append(f"{need}_{s}")
        raw_w, ns9_w, wx3_w = wins(raw_m), wins(ns9_m), wins(wx3_m)
        wx3_ptm = pt(wx3_m)
        only = wx3_w - ns9_w
        regr = ns9_w - wx3_w
        symbolic = {n for n, t in wx3_ptm.items()
                    if t.get("finished") and
                    t.get("winning_tactic_origin") == "wrapper_symbolic_action"}
        for n in sorted(only):
            b = wx3_ptm.get(n, {})
            fam_counts[b.get("winning_tactic_family_source") or "?"] += 1
            wx3_only_all.append({
                "full_name": n, "set": s,
                "file_path": name_to_file.get(n, ""),
                "winning_tactic": b.get("winning_tactic"),
                "winning_origin": b.get("winning_tactic_origin"),
                "winning_family": b.get("winning_tactic_family_source"),
                "variable": _var_of(b.get("winning_tactic")),
            })
        for n in sorted(symbolic):
            b = wx3_ptm.get(n, {})
            symbolic_all.append({
                "full_name": n, "set": s,
                "winning_tactic": b.get("winning_tactic"),
                "winning_family": b.get("winning_tactic_family_source"),
                "variable": _var_of(b.get("winning_tactic")),
            })
        for n in sorted(regr):
            regr_all.append({"full_name": n, "set": s})
        per_set.append({
            "set": s, "available": (wx3_m or {}).get("available"),
            "total": (wx3_m or {}).get("total_theorems"),
            "raw_wins": len(raw_w), "ns9_wins": len(ns9_w),
            "wx3ind_wins": len(wx3_w),
            "wx3_only_beyond_ns9": len(only),
            "symbolic_action_wins": len(symbolic),
            "regressions_vs_ns9": len(regr),
            "wx3_only_theorems": sorted(only),
            "symbolic_theorems": sorted(symbolic),
            "regression_theorems": sorted(regr),
        })

    out = {
        "router": "ns24_router", "top_k": 8, "max_steps": 8,
        "configs": {
            "raw": "routed_generative (no wrapper)",
            "ns9": "project/evolve/best/ns9_best_genome.json",
            "wx3ind": "project/evolve/experiments/wx3/"
                      "wx3_multiset_induction_safe.json"},
        "missing_runs": sorted(set(missing)),
        "per_set_summary": per_set,
        "totals": {
            "wx3_only_beyond_ns9": len(wx3_only_all),
            "symbolic_action_wins": len(symbolic_all),
            "regressions_vs_ns9": len(regr_all),
            "family_counts": dict(fam_counts.most_common()),
        },
        "wx3_only_wins": wx3_only_all,
        "symbolic_wins": symbolic_all,
        "regressions": regr_all,
    }
    Path(OUT).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"wrote {OUT}")
    if missing:
        print("MISSING:", sorted(set(missing)))
    print(f"{'set':34s} {'raw':>4} {'ns9':>4} {'wx3':>4} {'only':>5} "
          f"{'sym':>4} {'regr':>5}")
    for r in per_set:
        print(f"{r['set']:34s} {r['raw_wins']:>4} {r['ns9_wins']:>4} "
              f"{r['wx3ind_wins']:>4} {r['wx3_only_beyond_ns9']:>5} "
              f"{r['symbolic_action_wins']:>4} {r['regressions_vs_ns9']:>5}")
    print(f"\nTOTAL wx3-only beyond NS9: {len(wx3_only_all)}  "
          f"symbolic wins: {len(symbolic_all)}  "
          f"regressions: {len(regr_all)}")
    print("families:", dict(fam_counts))


if __name__ == "__main__":
    main()
