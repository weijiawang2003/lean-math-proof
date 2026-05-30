"""WX2 Stage 6 — extract raw vs NS9 vs WX2-generalized signal on fresh sets.

Runs to read (all on ns24_router, top-k 8 max-steps 8):
  raw         : wx1_wx2raw_<set>   (routed, no wrapper)
  ns9         : wx1_wx2ns9_<set>   (NS9 best genome)
  generalized : wx1_wx2gen_<set>   (Option+Bool+List state-aware cases)

WX2-promoted (Option-only) ≡ NS9 on these non-Option sets by the
namespace gate, so it is not re-run; the generalized config is the
candidate. Computes per-set raw/ns9/generalized wins, generalized-only
wins beyond NS9, regressions, and per-win tactic origin/family.

Output: project/data/wx2_generalized_cases_probe_meta.json
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

WX2_SETS = ["wx2_list_cases_easy", "wx2_list_cases_medium",
            "wx2_list_induction", "wx2_prod_cases", "wx2_bool_cases_control"]


def first(pat):
    ms = sorted(glob.glob(pat))
    return ms[0] if ms else None


def metrics(tag, s):
    f = first(f"project/evolve/eval_runs/wx1_{tag}_{s}/eval-*/metrics.json")
    return json.load(open(f)) if f else None


def pt(d):
    return {t["full_name"]: t for t in d.get("per_theorem", [])} if d else {}


def wins(d):
    return {n for n, t in pt(d).items() if t.get("finished")}


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import tasks
    name_to_file = {}
    for _s, thms in tasks.THEOREM_SETS.items():
        for t in thms:
            name_to_file.setdefault(t.full_name, t.file_path)

    per_set = []
    gen_only_all = []
    regressions_all = []
    origin_counts = Counter()
    fam_counts = Counter()

    for s in WX2_SETS:
        raw_m = metrics("wx2raw", s)
        ns9_m = metrics("wx2ns9", s)
        gen_m = metrics("wx2gen", s)
        raw_w, ns9_w = wins(raw_m), wins(ns9_m)
        gen_w = wins(gen_m)
        gen_pt = pt(gen_m)
        gen_only = gen_w - ns9_w
        regress = ns9_w - gen_w

        for n in sorted(gen_only):
            b = gen_pt.get(n, {})
            origin_counts[b.get("winning_tactic_origin") or "?"] += 1
            fam_counts[b.get("winning_tactic_family_source") or "?"] += 1
            gen_only_all.append({
                "full_name": n, "file_path": name_to_file.get(n, ""),
                "namespace": n.split(".")[0], "set": s,
                "winning_tactic": b.get("winning_tactic"),
                "winning_origin": b.get("winning_tactic_origin"),
                "winning_family": b.get("winning_tactic_family_source"),
            })
        for n in sorted(regress):
            regressions_all.append({"full_name": n, "set": s})

        per_set.append({
            "set": s,
            "available": gen_m.get("available") if gen_m else None,
            "raw_wins": len(raw_w), "ns9_wins": len(ns9_w),
            "wx2_generalized_wins": len(gen_w),
            "generalized_only_beyond_ns9": len(gen_only),
            "regressions_vs_ns9": len(regress),
            "generalized_only_theorems": sorted(gen_only),
            "regression_theorems": sorted(regress),
        })

    out = {
        "router": "ns24_router",
        "configs": {
            "raw": "routed_generative (no wrapper)",
            "ns9": "NS9 best genome",
            "wx2_generalized": "project/evolve/experiments/wx2/wx2_option_list_cases_safe.json",
            "wx2_promoted_note": "Option-only; ≡ NS9 on these non-Option sets by gate (not re-run).",
        },
        "per_set_summary": per_set,
        "totals": {
            "generalized_only_beyond_ns9": len(gen_only_all),
            "regressions_vs_ns9": len(regressions_all),
        },
        "generalized_only_origin_counts": dict(origin_counts.most_common()),
        "generalized_only_family_counts": dict(fam_counts.most_common()),
        "generalized_only_by_namespace": dict(
            Counter(w["namespace"] for w in gen_only_all).most_common()),
        "regression_theorems": regressions_all,
        "generalized_only_theorems": gen_only_all,
    }
    Path("project/data/wx2_generalized_cases_probe_meta.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("wrote project/data/wx2_generalized_cases_probe_meta.json")
    print(f"{'set':24s} {'raw':>4} {'ns9':>4} {'gen':>4} {'gen-only':>8} {'regr':>5}")
    for r in per_set:
        print(f"{r['set']:24s} {r['raw_wins']:>4} {r['ns9_wins']:>4} "
              f"{r['wx2_generalized_wins']:>4} "
              f"{r['generalized_only_beyond_ns9']:>8} "
              f"{r['regressions_vs_ns9']:>5}")
    print(f"\nTOTAL gen-only beyond NS9: {len(gen_only_all)}  "
          f"regressions: {len(regressions_all)}")
    print(f"by origin: {dict(origin_counts)}")
    print(f"by namespace: {dict(Counter(w['namespace'] for w in gen_only_all))}")


if __name__ == "__main__":
    main()
