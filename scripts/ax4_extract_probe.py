"""AX4 Stage 3 (extract) — raw vs NS9 vs WX3-induction signal on AX4 sets.

Reads the matrix produced by scripts/ax4_run_mining_guarded.sh (ns24_router):
  raw    : ax4_raw_<set>
  ns9    : ax4_ns9_<set>
  wx3ind : ax4_wx3ind_<set>   (wx3_multiset_induction_safe.json)

Robust to timeout-killed cells: prefers metrics.json (written at run end) but
falls back to reconstructing per-theorem outcomes from the incrementally
appended traces.jsonl when metrics.json is absent. A theorem is "won" if any
trace record has proof_finished True; the winning step's tactic / origin /
family are taken from that record.

Computes per set: raw/ns9/wx3 wins, WX3-only-beyond-NS9, symbolic-action wins
(origin wrapper_symbolic_action), regressions, per-win tactic/family/var.
Output: project/data/ax4_multiset_mining_probe_meta.json
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

SETS = ["ax4_multiset_induction_high_confidence",
        "ax4_multiset_cross_surface",
        "ax4_multiset_induction_heldout",
        "ax4_multiset_induction_medium_confidence",
        "ax4_multiset_induction_hard",
        "ax4_multiset_negative_control",
        "ax4_multiset_induction_heldout2"]
OUT = "project/data/ax4_multiset_mining_probe_meta.json"


def first(pat):
    ms = sorted(glob.glob(pat))
    return ms[0] if ms else None


def _from_metrics(tag, s):
    f = first(f"project/evolve/eval_runs/ax4_{tag}_{s}/eval-*/metrics.json")
    if not f:
        return None
    d = json.load(open(f))
    out = {}
    for t in d.get("per_theorem", []):
        out[t["full_name"]] = {
            "finished": bool(t.get("finished")),
            "winning_tactic": t.get("winning_tactic"),
            "winning_tactic_origin": t.get("winning_tactic_origin"),
            "winning_tactic_family_source": t.get("winning_tactic_family_source"),
        }
    return out


def _from_traces(tag, s):
    """Fallback: reconstruct per-theorem win info from traces.jsonl."""
    tfs = glob.glob(f"project/evolve/eval_runs/ax4_{tag}_{s}/eval-*/traces.jsonl")
    if not tfs:
        return None
    out = {}
    for tf in tfs:
        for line in open(tf):
            try:
                o = json.loads(line)
            except Exception:
                continue
            fn = o.get("full_name")
            if not fn:
                continue
            out.setdefault(fn, {"finished": False, "winning_tactic": None,
                                "winning_tactic_origin": None,
                                "winning_tactic_family_source": None})
            if o.get("proof_finished") and not out[fn]["finished"]:
                out[fn] = {
                    "finished": True,
                    "winning_tactic": o.get("tactic"),
                    "winning_tactic_origin": o.get("tactic_origin"),
                    "winning_tactic_family_source": o.get("tactic_family_source"),
                }
    return out


def per_theorem(tag, s):
    m = _from_metrics(tag, s)
    if m is not None:
        return m, "metrics"
    t = _from_traces(tag, s)
    if t is not None:
        return t, "traces_fallback"
    return None, "missing"


def wins(ptm):
    return {n for n, t in (ptm or {}).items() if t.get("finished")}


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

    per_set, wx3_only_all, symbolic_all, regr_all = [], [], [], []
    fam_counts = Counter()
    sources = {}

    for s in SETS:
        raw_m, raw_src = per_theorem("raw", s)
        ns9_m, ns9_src = per_theorem("ns9", s)
        wx3_m, wx3_src = per_theorem("wx3ind", s)
        sources[s] = {"raw": raw_src, "ns9": ns9_src, "wx3ind": wx3_src}
        raw_w, ns9_w, wx3_w = wins(raw_m), wins(ns9_m), wins(wx3_m)
        only = wx3_w - ns9_w
        regr = ns9_w - wx3_w
        symbolic = {n for n, t in (wx3_m or {}).items()
                    if t.get("finished") and
                    t.get("winning_tactic_origin") == "wrapper_symbolic_action"}
        for n in sorted(only):
            b = (wx3_m or {}).get(n, {})
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
            b = (wx3_m or {}).get(n, {})
            symbolic_all.append({
                "full_name": n, "set": s,
                "winning_tactic": b.get("winning_tactic"),
                "winning_family": b.get("winning_tactic_family_source"),
                "variable": _var_of(b.get("winning_tactic")),
            })
        for n in sorted(regr):
            regr_all.append({"full_name": n, "set": s})
        per_set.append({
            "set": s,
            "raw_wins": len(raw_w), "ns9_wins": len(ns9_w),
            "wx3ind_wins": len(wx3_w),
            "wx3_only_beyond_ns9": len(only),
            "symbolic_action_wins": len(symbolic),
            "regressions_vs_ns9": len(regr),
            "n_evaluated_wx3": len(wx3_m or {}),
            "wx3_only_theorems": sorted(only),
            "symbolic_theorems": sorted(symbolic),
            "regression_theorems": sorted(regr),
        })

    out = {
        "arc": "AX4", "router": "ns24_router", "top_k": 8, "max_steps": 8,
        "configs": {
            "raw": "routed_generative (no wrapper)",
            "ns9": "project/evolve/best/ns9_best_genome.json",
            "wx3ind": "project/evolve/experiments/wx3/"
                      "wx3_multiset_induction_safe.json"},
        "data_sources": sources,
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
    print(f"{'set':42s} {'raw':>4} {'ns9':>4} {'wx3':>4} {'only':>5} "
          f"{'sym':>4} {'regr':>5} {'src'}")
    for r, s in zip(per_set, SETS):
        print(f"{r['set']:42s} {r['raw_wins']:>4} {r['ns9_wins']:>4} "
              f"{r['wx3ind_wins']:>4} {r['wx3_only_beyond_ns9']:>5} "
              f"{r['symbolic_action_wins']:>4} {r['regressions_vs_ns9']:>5} "
              f"{sources[s]['wx3ind']}")
    print(f"\nTOTAL wx3-only beyond NS9: {len(wx3_only_all)}  "
          f"symbolic wins: {len(symbolic_all)}  regressions: {len(regr_all)}")
    print("families:", dict(fam_counts))


if __name__ == "__main__":
    main()
