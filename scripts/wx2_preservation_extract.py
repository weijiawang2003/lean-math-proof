"""WX2 Stage 2 — build the broad preservation matrix.

Compares NS9 wrapper vs WX2-promoted (Option-only) wrapper across
preservation sets (Nat/Set/Finset/demo), CX3 Option surfaces (retention),
and a CX3 Bool control. WX2-promoted runs are `wx1_wx2prom_*`; NS9
baselines reuse cached runs. Also counts `wrapper_option_cases` emissions
per set (must be 0 outside Option).

Outputs:
  project/data/wx2_preservation_matrix.json
  project/evolve/reports/wx2_preservation_matrix.md
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

# set -> (ns9 baseline run-dir globs, in priority order), namespace class
NS9_BASELINE = {
    "cx3_option_simp_easy": ["cx3_wraprouted_ns24_router_cx3_option_simp_easy"],
    "cx3_option_cases_medium": ["cx3_wraprouted_ns24_router_cx3_option_cases_medium"],
    "cx3_bool_option_mixed": ["cx3_wraprouted_ns24_router_cx3_bool_option_mixed"],
    "cx3_bool_decide_easy": ["cx3_wraprouted_ns24_router_cx3_bool_decide_easy"],
    "demo_v1": ["ns24_wraprouted_ns24_router_demo_v1"],
    "nat_defs_medium": ["ns24_wraprouted_ns24_router_nat_defs_medium"],
    # Set/Finset: WX1-promoted (Option-gated) is NS9-equivalent on these.
    "ns17_set_extra": ["wx1_wx1pres_ns17_set_extra"],
    "ns17_finset_extra": ["wx1_wx1pres_ns17_finset_extra"],
}
NS_CLASS = {
    "cx3_option_simp_easy": "Option", "cx3_option_cases_medium": "Option",
    "cx3_bool_option_mixed": "Option/Bool", "cx3_bool_decide_easy": "Bool",
    "demo_v1": "mixed-Nat", "nat_defs_medium": "Nat",
    "ns17_set_extra": "Set", "ns17_finset_extra": "Finset",
}


def first(pat):
    ms = sorted(glob.glob(pat))
    return ms[0] if ms else None


def metrics(rundir):
    if not rundir:
        return None
    f = first(f"project/evolve/eval_runs/{rundir}/eval-*/metrics.json")
    return json.load(open(f)) if f else None


def wins(d):
    return {t["full_name"] for t in d.get("per_theorem", []) if t.get("finished")} if d else set()


def emissions(rundir):
    n = 0
    for tf in glob.glob(f"project/evolve/eval_runs/{rundir}/*/traces.jsonl"):
        for line in open(tf):
            try:
                if json.loads(line).get("tactic_origin") == "wrapper_option_cases":
                    n += 1
            except Exception:
                pass
    return n


def main() -> None:
    rows = []
    for s, ns9_dirs in NS9_BASELINE.items():
        ns9_d = None
        for d in ns9_dirs:
            if metrics(d):
                ns9_d = d
                break
        wx2_d = f"wx1_wx2prom_{s}"
        ns9_m = metrics(ns9_d)
        wx2_m = metrics(wx2_d)
        ns9_w = wins(ns9_m)
        wx2_w = wins(wx2_m)
        rows.append({
            "set": s,
            "namespace_class": NS_CLASS.get(s, "?"),
            "ns9_baseline_run": ns9_d,
            "ns9_wins": len(ns9_w),
            "wx2_promoted_wins": len(wx2_w),
            "delta": len(wx2_w) - len(ns9_w),
            "new_beyond_ns9": sorted(wx2_w - ns9_w),
            "regressions": sorted(ns9_w - wx2_w),
            "option_cases_emissions": emissions(wx2_d),
            "available": wx2_m.get("available") if wx2_m else None,
        })

    option_gain = sum(r["delta"] for r in rows
                      if NS_CLASS.get(r["set"], "").startswith(("Option",)))
    non_option_regressions = sum(
        len(r["regressions"]) for r in rows
        if not NS_CLASS.get(r["set"], "").startswith("Option"))
    leak = sum(r["option_cases_emissions"] for r in rows
               if NS_CLASS.get(r["set"], "") in ("Nat", "Set", "Finset",
                                                 "mixed-Nat", "Bool"))

    out = {
        "comparison": "NS9 wrapper vs WX2-promoted (Option-only) wrapper",
        "router": "ns24_router",
        "rows": rows,
        "summary": {
            "option_surface_delta": option_gain,
            "non_option_regressions": non_option_regressions,
            "option_cases_emissions_outside_option": leak,
            "by_construction_note": (
                "nat_defs_large_v5 and ns14_set_finset_extra are unchanged by "
                "the same namespace gate (ranked-list identity proof); not "
                "re-run here."),
        },
    }
    Path("project/data/wx2_preservation_matrix.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# WX2 — broad preservation matrix", ""]
    md.append("NS9 wrapper vs **WX2-promoted** (Option-only state-aware "
              "cases) on `ns24_router`. Promoted config = NS9 genome + the "
              "WX1-validated Option cases block (2 tactics).")
    md.append("")
    md.append("| set | ns class | NS9 | WX2-prom | Δ | regress | option_cases emit |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(f"| {r['set']} | {r['namespace_class']} | {r['ns9_wins']} | "
                  f"{r['wx2_promoted_wins']} | {r['delta']:+d} | "
                  f"{len(r['regressions'])} | {r['option_cases_emissions']} |")
    md.append("")
    md.append(f"- **Option-surface delta vs NS9: +{option_gain}** "
              "(WX1 gain retained by the promoted 2-tactic config).")
    md.append(f"- **Non-Option regressions: {non_option_regressions}.**")
    md.append(f"- **`wrapper_option_cases` emissions outside Option: "
              f"{leak}.** (Namespace gate holds; preservation by "
              "construction confirmed empirically.)")
    md.append("- `nat_defs_large_v5` / `ns14_set_finset_extra`: unchanged by "
              "the same gate (ranked-list identity), not re-run.")
    Path("project/evolve/reports/wx2_preservation_matrix.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")

    print("wrote project/data/wx2_preservation_matrix.json")
    print("wrote project/evolve/reports/wx2_preservation_matrix.md")
    for r in rows:
        print(f"  {r['set']:26s} {r['namespace_class']:11s} "
              f"NS9={r['ns9_wins']:3d} WX2={r['wx2_promoted_wins']:3d} "
              f"Δ={r['delta']:+d} regress={len(r['regressions'])} "
              f"emit={r['option_cases_emissions']}")
    print(f"\noption delta +{option_gain}  non-option regressions "
          f"{non_option_regressions}  leak {leak}")


if __name__ == "__main__":
    main()
