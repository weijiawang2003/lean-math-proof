"""AX1 Stage 6 — WX2-vs-AX1 symbolic-config equivalence extractor.

Compares the WX2 custom cases wrapper against the AX1 symbolic-action
wrapper on the same sets. The AX1 config should reproduce WX2 wins
(within 1-2), with no regressions and no symbolic emissions outside the
gated Option/List namespaces.

WX2 baselines: List sets -> wx1_wx2gen_*, Option sets -> wx1_wx2prom_*
(Option-equivalent), demo_v1 -> wx1_wx2prom_demo_v1.

Outputs:
  project/data/ax1_symbolic_equivalence_meta.json
  project/evolve/reports/ax1_symbolic_equivalence_report.md
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

# set -> (WX2 baseline run-dir, namespace class)
PAIRS = [
    ("wx2_list_cases_easy", "wx1_wx2gen_wx2_list_cases_easy", "List"),
    ("wx2_list_cases_medium", "wx1_wx2gen_wx2_list_cases_medium", "List"),
    ("cx3_option_simp_easy", "wx1_wx2prom_cx3_option_simp_easy", "Option"),
    ("cx3_option_cases_medium", "wx1_wx2prom_cx3_option_cases_medium", "Option"),
    ("cx3_bool_option_mixed", "wx1_wx2prom_cx3_bool_option_mixed", "Option"),
    ("demo_v1", "wx1_wx2prom_demo_v1", "mixed-Nat"),
]


def first(pat):
    ms = sorted(glob.glob(pat))
    return ms[0] if ms else None


def metrics(rundir):
    f = first(f"project/evolve/eval_runs/{rundir}/eval-*/metrics.json")
    return json.load(open(f)) if f else None


def wins(d):
    return {t["full_name"] for t in d.get("per_theorem", []) if t.get("finished")} if d else set()


def symbolic_emissions(rundir):
    n = 0
    for tf in glob.glob(f"project/evolve/eval_runs/{rundir}/*/traces.jsonl"):
        for line in open(tf):
            try:
                if json.loads(line).get("tactic_origin") == "wrapper_symbolic_action":
                    n += 1
            except Exception:
                pass
    return n


def main() -> None:
    rows = []
    for s, wx2_dir, ns_class in PAIRS:
        wx2_m = metrics(wx2_dir)
        ax1_m = metrics(f"wx1_ax1_{s}")
        wx2_w = wins(wx2_m)
        ax1_w = wins(ax1_m)
        rows.append({
            "set": s, "namespace_class": ns_class,
            "wx2_baseline_run": wx2_dir,
            "wx2_wins": len(wx2_w), "ax1_wins": len(ax1_w),
            "delta_ax1_minus_wx2": len(ax1_w) - len(wx2_w),
            "ax1_only": sorted(ax1_w - wx2_w),
            "wx2_only": sorted(wx2_w - ax1_w),
            "ax1_symbolic_emissions": symbolic_emissions(f"wx1_ax1_{s}"),
            "available": ax1_m.get("available") if ax1_m else None,
        })

    total_wx2 = sum(r["wx2_wins"] for r in rows)
    total_ax1 = sum(r["ax1_wins"] for r in rows)
    max_abs_delta = max((abs(r["delta_ax1_minus_wx2"]) for r in rows), default=0)
    leak = sum(r["ax1_symbolic_emissions"] for r in rows
               if r["namespace_class"] not in ("List", "Option"))
    reproduced = (max_abs_delta <= 2 and leak == 0)

    out = {
        "comparison": "WX2 custom cases wrapper vs AX1 symbolic-action wrapper",
        "router": "ns24_router",
        "ax1_config": "project/evolve/experiments/ax1/ax1_symbolic_option_list_cases.json",
        "rows": rows,
        "summary": {
            "total_wx2_wins": total_wx2, "total_ax1_wins": total_ax1,
            "max_abs_per_set_delta": max_abs_delta,
            "symbolic_emissions_outside_gated_namespaces": leak,
            "ax1_reproduces_wx2": reproduced,
        },
    }
    Path("project/data/ax1_symbolic_equivalence_meta.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# AX1 — WX2 vs symbolic-action equivalence", ""]
    md.append("Does the AX1 symbolic-action wrapper reproduce the WX2 "
              "custom cases wrapper? Same sets, `ns24_router`, top-k 8 "
              "max-steps 8.")
    md.append("")
    md.append("| set | ns class | WX2 | AX1 | Δ | ax1_only | wx2_only | symbolic emit |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(f"| {r['set']} | {r['namespace_class']} | {r['wx2_wins']} | "
                  f"{r['ax1_wins']} | {r['delta_ax1_minus_wx2']:+d} | "
                  f"{len(r['ax1_only'])} | {len(r['wx2_only'])} | "
                  f"{r['ax1_symbolic_emissions']} |")
    md.append("")
    md.append(f"- **AX1 reproduces WX2: {reproduced}** "
              f"(max per-set |Δ| = {max_abs_delta}, threshold 2).")
    md.append(f"- Total wins: WX2 {total_wx2} vs AX1 {total_ax1}.")
    md.append(f"- Symbolic emissions outside gated Option/List namespaces: "
              f"{leak} (demo control).")
    Path("project/evolve/reports/ax1_symbolic_equivalence_report.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")

    print("wrote project/data/ax1_symbolic_equivalence_meta.json")
    print("wrote project/evolve/reports/ax1_symbolic_equivalence_report.md")
    for r in rows:
        print(f"  {r['set']:26s} {r['namespace_class']:10s} WX2={r['wx2_wins']:3d} "
              f"AX1={r['ax1_wins']:3d} Δ={r['delta_ax1_minus_wx2']:+d} "
              f"symbolic_emit={r['ax1_symbolic_emissions']}")
    print(f"\nax1_reproduces_wx2={reproduced} max|Δ|={max_abs_delta} leak={leak}")


if __name__ == "__main__":
    main()
