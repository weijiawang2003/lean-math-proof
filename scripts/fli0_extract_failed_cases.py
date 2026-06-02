#!/usr/bin/env python3
"""FLI0 Part 3 — extract theorem-level failure cases from RC5V2 / RC5V3 raw artifacts.

A theorem is a FAILURE_CASE iff it was dynamic-eligible (CONFIRMED_RC2_FAILURE), RC4 static did
not solve it, and the dynamic stage did not solve it. Dynamic successes and RC2/RC4-solved
theorems are excluded. Infra (timeout / network setup error) and unknown-name-only failures are
included but tagged non-clean. CLEAN_FAILURE = a live `proof_failed` attempt with a readable
trace (no kill, no setup error, not unknown-name-only).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_KEYS = ["has_iff", "has_eq", "has_subset", "has_mem", "has_disjoint", "has_card",
                "has_map_filter", "has_tofinset", "has_nat_arith", "has_order",
                "has_singleton", "has_union_inter"]


def _p(*a):
    return os.path.join(_REPO, *a)


def _load(root, rel):
    path = _p(root, rel)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _by_name(d, key="results"):
    if not d:
        return {}
    return {r["full_name"]: r for r in d.get(key, []) if "full_name" in r}


def _retr_by_name(d):
    if not d:
        return {}
    return {r["target"]: r for r in d.get("results", []) if "target" in r}


def _merge_dynamic_v3(fn, budgets):
    """Merge B1/B3/B5 for one theorem; earliest live success wins."""
    recs = [b[fn] for b in budgets if fn in b]
    if not recs:
        return None
    succ = next((r for r in recs if r.get("success")), None)
    live = [r for r in recs if r.get("live") and r.get("programs_attempted", 0) > 0]
    killed = any(r.get("killed_by_timeout") for r in recs)
    setup_all = all(r.get("setup_error") for r in recs)
    chosen = succ or (live[0] if live else recs[0])
    # aggregate attempted tactics / outcomes across every live budget
    failures, controls = [], []
    for r in recs:
        failures += r.get("failures", []) or []
        if r.get("controls"):
            controls = r["controls"]  # controls captured once (B1); same set
    return {"chosen": chosen, "success": bool(succ), "killed": killed,
            "setup_only": setup_all and not live, "any_live": bool(live),
            "failures": failures, "controls": controls,
            "programs_attempted": sum(r.get("programs_attempted", 0) for r in recs)}


def _classify_dynamic(success, killed, setup_only, any_live, failures, programs_attempted):
    if success:
        return "solved", None
    if killed:
        return "killed", "timeout/kill"
    if setup_only or (not any_live and programs_attempted == 0):
        return "infra_error", "setup/network error, no live attempt"
    outs = [f.get("outcome") for f in failures]
    if outs := [o for o in outs if o]:
        if all(o == "unknown_name" for o in outs):
            return "unknown_name", "all attempts unknown_name"
    if programs_attempted > 0 or failures:
        return "failed", None
    return "missing", "no dynamic record"


def _build_stage(stage, root, names):
    elig = _by_name(_load(root, names["eligible"]))
    rc2 = _by_name(_load(root, names["rc2"]))
    static = _by_name(_load(root, names["static"]))
    retr = _retr_by_name(_load(root, names["retrieval"]))
    attr = _by_name(_load(root, names["attr"]), "records") if names.get("attr") else {}
    if stage == "RC5V2":
        b5 = _by_name(_load(root, names["b5"]))
        dynamic = {}
        for fn in b5:
            r = b5[fn]
            dynamic[fn] = {"chosen": r, "success": bool(r.get("success")),
                           "killed": bool(r.get("killed_by_timeout")),
                           "setup_only": bool(r.get("setup_error")) and not r.get("live"),
                           "any_live": bool(r.get("live")) and r.get("programs_attempted", 0) > 0,
                           "failures": r.get("failures", []) or [],
                           "controls": r.get("controls", []) or [],
                           "programs_attempted": r.get("programs_attempted", 0)}
    else:
        bs = [_by_name(_load(root, names[k])) for k in ("b1", "b3", "b5")]
        allnames = set().union(*[set(b) for b in bs]) if bs else set()
        dynamic = {fn: _merge_dynamic_v3(fn, bs) for fn in allnames}
        dynamic = {k: v for k, v in dynamic.items() if v}
    return {"elig": elig, "rc2": rc2, "static": static, "retr": retr,
            "attr": attr, "dynamic": dynamic, "root": root}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    V2 = "project/evolve/experiments/rc5_v2"
    V3 = "project/evolve/experiments/rc5_v3"
    stages = {
        "RC5V2": _build_stage("RC5V2", V2, {
            "eligible": "cases/rc5v2_dynamic_eligible.json",
            "rc2": "out/rc5v2_rc2_baseline_results.json",
            "static": "out/rc5v2_static_stage_results.json",
            "retrieval": "out/rc5v2_retrieval_results.json",
            "attr": "out/rc5v2_attribution.json",
            "b5": "out/rc5v2_b5_dynamic_results.json"}),
        "RC5V3": _build_stage("RC5V3", V3, {
            "eligible": "cases/rc5v3_dynamic_eligible.json",
            "rc2": "out/rc5v3_rc2_baseline_results.json",
            "static": "out/rc5v3_static_stage_results.json",
            "retrieval": "out/rc5v3_retrieval_results.json",
            "b1": "out/rc5v3_b1_dynamic_results.json",
            "b3": "out/rc5v3_b3_dynamic_results.json",
            "b5": "out/rc5v3_b5_dynamic_results.json"}),
    }

    records = []
    seen_overlap = []
    name_to_stages = {}
    for stage, S in stages.items():
        for fn in S["elig"]:
            name_to_stages.setdefault(fn, []).append(stage)

    for stage, S in stages.items():
        for fn, e in S["elig"].items():
            rc2_status = (S["rc2"].get(fn) or {}).get("status", "missing")
            rc4_rec = S["static"].get(fn) or {}
            rc4_status = rc4_rec.get("status", "missing")
            dyn = S["dynamic"].get(fn)
            if dyn is None:
                dyn_result, dyn_reason = "missing", "no dynamic record"
                failures, controls, killed, prog = [], [], False, 0
            else:
                dyn_result, dyn_reason = _classify_dynamic(
                    dyn["success"], dyn["killed"], dyn["setup_only"], dyn["any_live"],
                    dyn["failures"], dyn["programs_attempted"])
                failures, controls = dyn["failures"], dyn["controls"]
                killed, prog = dyn["killed"], dyn["programs_attempted"]
            # exclude non-failures
            if rc2_status == "solved" or rc4_status == "solved" or dyn_result == "solved":
                continue
            r = S["retr"].get(fn) or {}
            top_lemmas = [tl.get("lemma") for tl in (r.get("top_lemmas") or [])[:5]]
            retrieval_coverage = (bool(r.get("top_lemmas")) if r else None)
            attempted = [f.get("tactic") for f in failures if f.get("tactic")]
            attempted += [c.get("tactic") for c in controls if c.get("tactic")]
            errors = []
            for c in controls:
                if c.get("error") and not c.get("solved"):
                    errors.append({"tactic": c.get("tactic"), "error": c.get("error")[:160]})
            outc = Counter(f.get("outcome") for f in failures if f.get("outcome"))
            if dyn and dyn.get("chosen", {}).get("setup_error"):
                errors.append({"setup_error": dyn["chosen"]["setup_error"][:160]})
            feats = e.get("features", {}) or {}
            feature_bucket = [k for k in FEATURE_KEYS if feats.get(k)]
            # CLEAN definition
            clean = (dyn_result == "failed" and not killed
                     and any(f.get("outcome") == "proof_failed" for f in failures))
            attr_cls = (S["attr"].get(fn) or {}).get("classification")
            if attr_cls is None:
                attr_cls = {"failed": "NO_DYNAMIC_WIN", "killed": "TIMEOUT_BOUNDED",
                            "infra_error": "INFRA_ERROR", "unknown_name": "UNKNOWN_NAME_FAIL",
                            "missing": "NO_DYNAMIC_RECORD"}.get(dyn_result, "NEEDS_REVIEW")
            rec = {
                "theorem": fn,
                "namespace": e.get("namespace"),
                "source_stage": stage,
                "also_in_stages": [s for s in name_to_stages.get(fn, []) if s != stage],
                "freshness_status": e.get("freshness_status"),
                "rc2_result": rc2_status,
                "rc4_result": rc4_status,
                "dynamic_result": dyn_result,
                "dynamic_reason": dyn_reason,
                "clean_failure": clean,
                "attribution": attr_cls,
                "feature_bucket": feature_bucket,
                "eligible_for_dynamic": True,
                "retrieval_coverage": retrieval_coverage,
                "top_retrieved_lemmas": top_lemmas,
                "attempted_tactics": attempted[:12],
                "num_failure_outcomes": dict(outc),
                "errors": errors[:6],
                "runtime_flags": {"killed_by_timeout": killed, "programs_attempted": prog,
                                  "any_live": bool(dyn and dyn.get("any_live"))},
                "statement_text": e.get("statement_text"),
                "file_path": e.get("file_path"),
                "source_files": [S["root"]],
            }
            if rec["also_in_stages"] and stage == "RC5V2":
                seen_overlap.append(fn)
            records.append(rec)

    records.sort(key=lambda r: (r["source_stage"], r["namespace"] or "", r["theorem"]))
    with open(_p(args.out_jsonl), "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    clean = [r for r in records if r["clean_failure"]]
    summary = {
        "generated_by": "scripts/fli0_extract_failed_cases.py",
        "total_failures": len(records),
        "clean_failures": len(clean),
        "by_source_stage": dict(Counter(r["source_stage"] for r in records)),
        "clean_by_source_stage": dict(Counter(r["source_stage"] for r in clean)),
        "by_namespace": dict(Counter(r["namespace"] for r in records).most_common()),
        "clean_by_namespace": dict(Counter(r["namespace"] for r in clean).most_common()),
        "by_dynamic_result": dict(Counter(r["dynamic_result"] for r in records)),
        "by_failure_reason": dict(Counter(r["dynamic_reason"] for r in records if r["dynamic_reason"])),
        "by_feature": dict(Counter(f for r in records for f in r["feature_bucket"]).most_common()),
        "clean_by_feature": dict(Counter(f for r in clean for f in r["feature_bucket"]).most_common()),
        "rc5v2_rc5v3_overlap": len(set(seen_overlap)),
        "fresh_clean_failures": sum(1 for r in clean if r["freshness_status"] in ("strict_fresh", "soft_fresh")),
    }
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI0 failed-case extraction summary", "",
          f"- **total failures: {summary['total_failures']}** | "
          f"**clean failures: {summary['clean_failures']}** "
          f"(fresh: {summary['fresh_clean_failures']})",
          f"- by source stage: {summary['by_source_stage']} | clean: {summary['clean_by_source_stage']}",
          f"- RC5V2∩RC5V3 overlap: {summary['rc5v2_rc5v3_overlap']}",
          f"- by dynamic result: {summary['by_dynamic_result']}",
          f"- by failure reason: {summary['by_failure_reason']}", "",
          "## Clean failures by namespace", "", "| namespace | clean | all |", "|---|---|---|"]
    for ns in summary["by_namespace"]:
        md.append(f"| {ns} | {summary['clean_by_namespace'].get(ns, 0)} | {summary['by_namespace'][ns]} |")
    md += ["", "## Clean failures by feature", "", "| feature | clean |", "|---|---|"]
    for ft, n in summary["clean_by_feature"].items():
        md.append(f"| {ft} | {n} |")
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli0-extract] total={summary['total_failures']} clean={summary['clean_failures']} "
          f"fresh_clean={summary['fresh_clean_failures']} "
          f"stage={summary['by_source_stage']} dyn={summary['by_dynamic_result']}")


if __name__ == "__main__":
    main()
