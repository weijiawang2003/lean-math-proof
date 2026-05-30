"""WX1 Stage 5 — extract raw vs NS9-wrapper vs WX1-wrapper signal.

Reuses the CX3 raw (`cx3_rawrouted_*`) and NS9-wrapper (`cx3_wraprouted_*`)
runs as baselines A and B, and the WX1 runs (`wx1_wx1_*`, `wx1_wx1b_*`) as
candidate C. For each set computes raw / NS9 / WX1 wins, WX1-only wins
(beyond NS9), regressions, and the per-theorem winning tactic + origin.

Writes:
  project/data/wx1_option_cases_probe_meta.json
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

# (set, wx1_tag) — Option surfaces + Bool control use the Option-only
# config; Bool surfaces additionally get the broader bool+option config.
WX1_SETS = [
    ("cx3_option_simp_easy", "wx1"),
    ("cx3_option_cases_medium", "wx1"),
    ("cx3_bool_option_mixed", "wx1"),
    ("cx3_bool_decide_easy", "wx1"),
]
WX1B_SETS = ["cx3_bool_decide_easy", "cx3_bool_simp_medium"]


def first(pat: str) -> str | None:
    ms = sorted(glob.glob(pat))
    return ms[0] if ms else None


def per_thm(p: str | None) -> dict[str, dict]:
    if not p:
        return {}
    return {t["full_name"]: t
            for t in json.load(open(p)).get("per_theorem", [])}


def wins(pt: dict[str, dict]) -> set[str]:
    return {n for n, t in pt.items() if t.get("finished")}


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import tasks
    name_to_file: dict[str, str] = {}
    for _s, thms in tasks.THEOREM_SETS.items():
        for t in thms:
            name_to_file.setdefault(t.full_name, t.file_path)

    per_set = []
    wx1_only_all: list[dict] = []
    regressions_all: list[dict] = []
    origin_counts: Counter = Counter()

    for s, tag in WX1_SETS:
        raw = first(f"project/evolve/eval_runs/cx3_rawrouted_ns24_router_{s}/"
                    "eval-*/metrics.json")
        ns9 = first(f"project/evolve/eval_runs/cx3_wraprouted_ns24_router_{s}/"
                    "eval-*/metrics.json")
        wx1 = first(f"project/evolve/eval_runs/wx1_{tag}_{s}/eval-*/metrics.json")
        raw_w = wins(per_thm(raw))
        ns9_w = wins(per_thm(ns9))
        wx1_pt = per_thm(wx1)
        wx1_w = wins(wx1_pt)

        wx1_only = wx1_w - ns9_w           # WX1 solves, NS9 doesn't
        regress = ns9_w - wx1_w            # NS9 solved, WX1 lost

        for n in sorted(wx1_only):
            blob = wx1_pt.get(n, {})
            tac = blob.get("winning_tactic") or ""
            origin = blob.get("winning_tactic_origin")
            origin_counts[origin or "?"] += 1
            wx1_only_all.append({
                "full_name": n,
                "file_path": name_to_file.get(n, ""),
                "namespace": n.split(".")[0],
                "set": s,
                "wx1_winning_tactic": tac,
                "wx1_winning_origin": origin,
                "wx1_winning_family": blob.get("winning_tactic_family_source"),
            })
        for n in sorted(regress):
            regressions_all.append({"full_name": n, "set": s})

        per_set.append({
            "set": s, "wx1_tag": tag,
            "raw_wins": len(raw_w), "ns9_wins": len(ns9_w),
            "wx1_wins": len(wx1_w),
            "wx1_only_beyond_ns9": len(wx1_only),
            "regressions_vs_ns9": len(regress),
            "wx1_only_theorems": sorted(wx1_only),
            "regression_theorems": sorted(regress),
        })

    # Broader bool+option config on Bool surfaces (reported separately).
    wx1b_summary = []
    for s in WX1B_SETS:
        ns9 = first(f"project/evolve/eval_runs/cx3_wraprouted_ns24_router_{s}/"
                    "eval-*/metrics.json")
        wx1b = first(f"project/evolve/eval_runs/wx1_wx1b_{s}/eval-*/metrics.json")
        if not wx1b:
            continue
        ns9_w = wins(per_thm(ns9))
        wx1b_w = wins(per_thm(wx1b))
        wx1b_summary.append({
            "set": s, "ns9_wins": len(ns9_w), "wx1b_wins": len(wx1b_w),
            "wx1b_only_beyond_ns9": len(wx1b_w - ns9_w),
            "wx1b_only_theorems": sorted(wx1b_w - ns9_w),
            "regressions_vs_ns9": len(ns9_w - wx1b_w),
        })

    by_family = Counter(w["wx1_winning_family"] or "?" for w in wx1_only_all)
    by_ns = Counter(w["namespace"] for w in wx1_only_all)

    out = {
        "router_used": "ns24_router",
        "baselines": {
            "raw": "cx3_rawrouted_ns24_router_*",
            "ns9": "cx3_wraprouted_ns24_router_* (NS9 best genome)",
        },
        "wx1_option_config": "project/evolve/experiments/wx1/wx1_option_cases_safe.json",
        "wx1_bool_option_config": "project/evolve/experiments/wx1/wx1_bool_option_cases_safe.json",
        "eval_settings": {"top_k": 8, "max_steps": 8},
        "per_set_summary": per_set,
        "wx1b_bool_summary": wx1b_summary,
        "totals": {
            "wx1_only_beyond_ns9": len(wx1_only_all),
            "regressions_vs_ns9": len(regressions_all),
        },
        "wx1_only_win_origin_counts": dict(origin_counts.most_common()),
        "wx1_only_by_family": dict(by_family.most_common()),
        "wx1_only_by_namespace": dict(by_ns.most_common()),
        "regression_theorems": regressions_all,
        "wx1_only_theorems": wx1_only_all,
    }
    Path("project/data/wx1_option_cases_probe_meta.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("wrote project/data/wx1_option_cases_probe_meta.json")
    print()
    print(f"{'set':28s} {'raw':>4} {'ns9':>4} {'wx1':>4} "
          f"{'WX1-only':>8} {'regress':>7}")
    for r in per_set:
        print(f"{r['set']:28s} {r['raw_wins']:>4} {r['ns9_wins']:>4} "
              f"{r['wx1_wins']:>4} {r['wx1_only_beyond_ns9']:>8} "
              f"{r['regressions_vs_ns9']:>7}")
    print()
    print(f"TOTAL WX1-only beyond NS9: {len(wx1_only_all)}  "
          f"regressions: {len(regressions_all)}")
    print(f"WX1-only by origin: {dict(origin_counts)}")
    print(f"WX1-only by namespace: {dict(by_ns)}")
    if wx1b_summary:
        print("\nwx1b (bool+option) on Bool surfaces:")
        for r in wx1b_summary:
            print(f"  {r['set']}: ns9={r['ns9_wins']} wx1b={r['wx1b_wins']} "
                  f"only={r['wx1b_only_beyond_ns9']}")


if __name__ == "__main__":
    main()
