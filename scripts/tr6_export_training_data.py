#!/usr/bin/env python3
"""TR6 Part 12 — export fresh live program-level training examples.

One attempted program (across B5/B10/B20) = one row, in the TR4 schema (so it augments
the TR4/TR5 datasets without overwriting them). Labels: FRESH_TRUE_DELTA /
FRESH_RC4A_EVIDENCE / FRESH_RC4B_EVIDENCE / FRESH_RC4C_EVIDENCE / BASELINE_DUPLICATE /
PRODUCTION_SUBSUMED / NO_WIN_UNDER_BUDGET / SOURCE_SPECIFIC / OPEN_FLAKE. Reports new
non-Set positives and whether they broaden by-namespace coverage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tr5_score as S

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _load(path):
    fp = _p(path) if not os.path.isabs(path) else path
    return json.load(open(fp)) if os.path.exists(fp) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked-plan", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--b5", default="project/evolve/experiments/tr6/out/tr6_b5_live_results.json")
    ap.add_argument("--b10", default="project/evolve/experiments/tr6/out/tr6_b10_live_results.json")
    ap.add_argument("--b20", default="project/evolve/experiments/tr6/out/tr6_b20_live_results.json")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    plan = {t["full_name"]: t for t in _load(args.ranked_plan)["theorems"]}
    attr = {r["full_name"]: r for r in _load(args.attribution)["records"]}
    prog_meta, goal_of, ns_of = {}, {}, {}
    for fn, t in plan.items():
        goal_of[fn] = None
        ns_of[fn] = t.get("namespace")
        for p in t.get("programs_ranked", []):
            prog_meta[(fn, p["tactic"])] = p
    # statement_text as goal proxy from confirmation
    conf = _load("project/evolve/experiments/tr6/out/tr6_rc2_confirmation.json")
    if conf:
        for r in conf["results"]:
            goal_of[r["full_name"]] = r.get("statement_text")

    seen = set()
    attempts = []
    for path in (args.b5, args.b10, args.b20):
        d = _load(path)
        if not d:
            continue
        for r in d["results"]:
            fn = r["full_name"]
            recs = []
            wp = r.get("winning_program")
            if wp:
                recs.append({"tactic": wp["tactic"], "rank": wp.get("rank"),
                             "family": wp.get("family"), "depth": wp.get("depth"),
                             "lemmas": wp.get("used_lemmas") or wp.get("lemmas") or [],
                             "ranker_score": wp.get("ranker_score"),
                             "outcome": "success", "solved": True})
            for f in r.get("failures", []):
                recs.append({"tactic": f["tactic"], "rank": f.get("rank"),
                             "family": f.get("family"),
                             "depth": (prog_meta.get((fn, f["tactic"])) or {}).get("depth"),
                             "lemmas": (prog_meta.get((fn, f["tactic"])) or {}).get("lemmas", []),
                             "ranker_score": (prog_meta.get((fn, f["tactic"])) or {}).get("ranker_score"),
                             "outcome": f.get("outcome"), "solved": False})
            for s in recs:
                key = (fn, s["tactic"])
                if key in seen:
                    continue
                seen.add(key)
                attempts.append((fn, s))

    rows = []
    for fn, s in attempts:
        a = attr.get(fn, {})
        cls = a.get("classification")
        solved = s["solved"]
        if solved and cls == "FRESH_TRUE_DELTA":
            if a.get("rc4a_evidence"):
                label = "FRESH_RC4A_EVIDENCE"
            elif a.get("rc4b_evidence"):
                label = "FRESH_RC4B_EVIDENCE"
            elif a.get("rc4c_evidence"):
                label = "FRESH_RC4C_EVIDENCE"
            else:
                label = "FRESH_TRUE_DELTA"
        elif solved and cls == "BASELINE_DUPLICATE":
            label = "BASELINE_DUPLICATE"
        elif solved and cls == "SOURCE_SPECIFIC":
            label = "SOURCE_SPECIFIC"
        elif solved and cls == "PRODUCTION_SUBSUMED":
            label = "PRODUCTION_SUBSUMED"
        elif not solved:
            label = "NO_WIN_UNDER_BUDGET"
        else:
            label = "OPEN_FLAKE"
        credited = label.startswith("FRESH_")
        base = S.build_row(fn, goal_of.get(fn), ns_of.get(fn), s["tactic"], s["lemmas"],
                           s.get("family"), s.get("depth") or 1, source="tr6")
        base.update({"source": "tr6", "rc2_status": "failed", "rank": s.get("rank"),
                     "ranker_score": s.get("ranker_score"), "program_family": s.get("family"),
                     "program_depth": s.get("depth") or 1, "outcome": s["outcome"],
                     "tr6_label": label, "label_success": 1 if solved else 0,
                     "label_credit": 1 if credited else 0})
        rows.append(base)

    os.makedirs(os.path.dirname(_p(args.out_jsonl)), exist_ok=True)
    with open(_p(args.out_jsonl), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    succ = sum(r["label_success"] for r in rows)
    cred = sum(r["label_credit"] for r in rows)
    by_label = Counter(r["tr6_label"] for r in rows)
    by_ns = Counter(r["namespace"] for r in rows)
    pos_by_ns = Counter(r["namespace"] for r in rows if r["label_credit"])
    nonset_pos = {ns: c for ns, c in pos_by_ns.items() if ns != "Set"}
    # TR4 had positives only in Set/Finset/List; does TR6 add new positive namespaces?
    tr4_pos_ns = {"Set", "Finset", "List"}
    new_pos_ns = {ns for ns in pos_by_ns if ns not in tr4_pos_ns and pos_by_ns[ns] > 0}
    summary = {
        "generated_by": "scripts/tr6_export_training_data.py",
        "num_examples": n, "success_positives": succ, "credit_positives": cred,
        "label_histogram": dict(by_label), "by_namespace": dict(by_ns),
        "credit_positives_by_namespace": dict(pos_by_ns),
        "nonset_credit_positives": nonset_pos,
        "new_positive_namespaces_vs_tr4": sorted(new_pos_ns),
        "helps_by_namespace_generalization": len(new_pos_ns) > 0,
        "note": "TR4/TR5 data NOT overwritten; TR6 augments. Fresh credit positives in "
                "previously-zero-positive namespaces directly address the by-namespace gap.",
    }
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR6 training delta", "",
          f"- attempted programs (examples): **{n}**",
          f"- success positives: **{succ}** | credit positives: **{cred}**",
          f"- label histogram: {dict(by_label)}",
          f"- credit positives by namespace: {dict(pos_by_ns)}",
          f"- **non-Set credit positives: {nonset_pos}**",
          f"- new positive namespaces vs TR4 (Set/Finset/List): {sorted(new_pos_ns)}",
          f"- helps by-namespace generalization: {summary['helps_by_namespace_generalization']}"]
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")
    print(f"[tr6-export] {n} examples, success={succ}, credit={cred}, "
          f"nonset_pos={nonset_pos}, new_pos_ns={sorted(new_pos_ns)}")


if __name__ == "__main__":
    main()
