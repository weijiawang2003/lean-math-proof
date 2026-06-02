#!/usr/bin/env python3
"""FLI0 Part 2 — locate & classify RC5V2 / RC5V3 source artifacts.

Pure read-only inventory. For each expected artifact records PRESENT / MISSING / PARTIAL /
UNREADABLE / NEEDS_REVIEW, plus lightweight stats (record count, infra/setup-error fraction).
Does not fail when the RC5V3 analysis layer / final report is absent — those are simply MISSING.
"""
from __future__ import annotations

import argparse
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (logical_name, relative_path, kind) ; kind in {results, plan, report, summary, jsonl}
RC5V2 = [
    ("eval_batch", "cases/rc5v2_eval_batch.json", "summary"),
    ("rc2_baseline_results", "out/rc5v2_rc2_baseline_results.json", "results"),
    ("rc4_static_results", "out/rc5v2_static_stage_results.json", "results"),
    ("dynamic_eligible", "cases/rc5v2_dynamic_eligible.json", "results"),
    ("retrieval_results", "out/rc5v2_retrieval_results.json", "results"),
    ("safe_dynamic_plan", "out/rc5v2_safe_dynamic_plan.json", "plan"),
    ("dynamic_b5_results", "out/rc5v2_b5_dynamic_results.json", "results"),
    ("attribution", "out/rc5v2_attribution.json", "summary"),
    ("system_comparison", "out/rc5v2_system_comparison.json", "summary"),
    ("safety_audit", "out/rc5v2_safety_audit.json", "summary"),
    ("exported_examples", "data/rc5v2_dynamic_examples.jsonl", "jsonl"),
    ("final_report", "../../reports/rc5/rc5v2_hardened_hybrid_fresh_benchmark_report.md", "report"),
]
RC5V3 = [
    ("eval_batch", "cases/rc5v3_eval_batch.json", "summary"),
    ("rc2_baseline_results", "out/rc5v3_rc2_baseline_results.json", "results"),
    ("rc4_static_results", "out/rc5v3_static_stage_results.json", "results"),
    ("dynamic_eligible", "cases/rc5v3_dynamic_eligible.json", "results"),
    ("retrieval_results", "out/rc5v3_retrieval_results.json", "results"),
    ("safe_dynamic_plan", "out/rc5v3_safe_dynamic_plan.json", "plan"),
    ("dynamic_b1_results", "out/rc5v3_b1_dynamic_results.json", "results"),
    ("dynamic_b3_results", "out/rc5v3_b3_dynamic_results.json", "results"),
    ("dynamic_b5_results", "out/rc5v3_b5_dynamic_results.json", "results"),
    ("attribution", "out/rc5v3_attribution.json", "summary"),
    ("system_comparison", "out/rc5v3_system_comparison.json", "summary"),
    ("cost_curve", "out/rc5v3_cost_curve.json", "summary"),
    ("namespace_feature_yield", "out/rc5v3_namespace_feature_yield.json", "summary"),
    ("safety_audit", "out/rc5v3_safety_audit.json", "summary"),
    ("maintenance_decision", "out/rc5v3_maintenance_decision.json", "summary"),
    ("exported_examples", "data/rc5v3_dynamic_examples.jsonl", "jsonl"),
    ("final_report", "../../reports/rc5/rc5v3_hardened_hybrid_scaling_benchmark_report.md", "report"),
]


def _classify(root, rel, kind):
    path = os.path.normpath(os.path.join(root, rel))
    info = {"logical_path": rel, "abs": path, "kind": kind}
    if not os.path.exists(path):
        info["status"] = "MISSING"
        return info
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    info["size_bytes"] = size
    if kind == "report":
        info["status"] = "PRESENT" if size > 0 else "PARTIAL"
        return info
    if kind == "jsonl":
        try:
            with open(path) as f:
                n = sum(1 for line in f if line.strip())
            info["records"] = n
            info["status"] = "PRESENT" if n > 0 else "PARTIAL"
        except Exception as e:  # noqa: BLE001
            info["status"] = "UNREADABLE"
            info["error"] = str(e)[:200]
        return info
    # json
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:  # noqa: BLE001
        info["status"] = "UNREADABLE"
        info["error"] = str(e)[:200]
        return info
    recs = d.get("results") if isinstance(d, dict) else None
    if recs is None and isinstance(d, dict):
        recs = d.get("theorems")
    if isinstance(recs, list):
        info["records"] = len(recs)
        setup_err = sum(1 for r in recs if isinstance(r, dict) and r.get("setup_error"))
        if setup_err:
            info["setup_error_records"] = setup_err
            info["setup_error_fraction"] = round(setup_err / max(1, len(recs)), 3)
        # PARTIAL when a large share of a dynamic-results file is infra/setup error
        if recs and setup_err / len(recs) > 0.25:
            info["status"] = "PARTIAL"
            info["partial_reason"] = f"{setup_err}/{len(recs)} records have setup_error (infra)"
        elif len(recs) == 0:
            info["status"] = "PARTIAL"
        else:
            info["status"] = "PRESENT"
    else:
        info["status"] = "PRESENT" if d else "NEEDS_REVIEW"
    return info


def _inventory(root, spec, stage):
    items = {}
    for name, rel, kind in spec:
        items[name] = _classify(root, rel, kind)
    present = [k for k, v in items.items() if v["status"] == "PRESENT"]
    missing = [k for k, v in items.items() if v["status"] == "MISSING"]
    partial = [k for k, v in items.items() if v["status"] == "PARTIAL"]
    unreadable = [k for k, v in items.items() if v["status"] == "UNREADABLE"]
    has_report = items.get("final_report", {}).get("status") == "PRESENT"
    # raw failure data = baseline + static + eligible + at least one dynamic results file present
    dyn_keys = [k for k in items if k.startswith("dynamic_b")]
    raw_ok = (items.get("rc2_baseline_results", {}).get("status") in ("PRESENT", "PARTIAL")
              and items.get("dynamic_eligible", {}).get("status") in ("PRESENT", "PARTIAL")
              and any(items[k]["status"] in ("PRESENT", "PARTIAL") for k in dyn_keys))
    if has_report and not missing:
        stage_status = "COMPLETE"
    elif raw_ok:
        stage_status = "PARTIAL_ARTIFACTS_AVAILABLE"
    else:
        stage_status = "INSUFFICIENT"
    return {"stage": stage, "root": root, "stage_status": stage_status,
            "has_final_report": has_report, "raw_failure_data_available": raw_ok,
            "present": present, "partial": partial, "missing": missing,
            "unreadable": unreadable, "items": items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rc5v2-root", required=True)
    ap.add_argument("--rc5v3-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    v2 = _inventory(os.path.join(_REPO, args.rc5v2_root), RC5V2, "RC5V2")
    v3 = _inventory(os.path.join(_REPO, args.rc5v3_root), RC5V3, "RC5V3")
    fli0_source = "BOTH" if (v2["raw_failure_data_available"] and v3["raw_failure_data_available"]) \
        else ("RC5V2" if v2["raw_failure_data_available"] else
              ("RC5V3" if v3["raw_failure_data_available"] else "NONE"))
    out = {"generated_by": "scripts/fli0_locate_source_artifacts.py",
           "rc5v2": v2, "rc5v3": v3, "fli0_source": fli0_source,
           "note": ("RC5V3 analysis layer / final report missing is expected and non-fatal; raw "
                    "per-theorem results drive failure extraction. RC5V3 = PARTIAL.")}
    with open(os.path.join(_REPO, args.out_json), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    def block(inv):
        lines = [f"### {inv['stage']} — `{inv['stage_status']}` "
                 f"(final report: {'yes' if inv['has_final_report'] else 'no'}, "
                 f"raw failure data: {'yes' if inv['raw_failure_data_available'] else 'no'})", "",
                 "| artifact | status | records | note |", "|---|---|---|---|"]
        for name, v in inv["items"].items():
            note = v.get("partial_reason") or v.get("error") or ""
            lines.append(f"| {name} | {v['status']} | {v.get('records', '')} | {note} |")
        lines.append("")
        return lines

    md = ["# FLI0 source artifact inventory", "",
          f"**FLI0 source: `{fli0_source}`** — {out['note']}", ""]
    md += block(v2) + block(v3)
    with open(os.path.join(_REPO, args.out_md), "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"[fli0-locate] source={fli0_source} "
          f"V2={v2['stage_status']}(miss {len(v2['missing'])}) "
          f"V3={v3['stage_status']}(miss {len(v3['missing'])}, partial {v3['partial']})")


if __name__ == "__main__":
    main()
