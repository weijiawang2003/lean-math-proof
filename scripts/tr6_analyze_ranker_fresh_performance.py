#!/usr/bin/env python3
"""TR6 Part 10 — ranker fresh-frontier performance analysis.

Probes/wins/credit by budget, success-per-probe, first-success rank, no-win rate,
unknown-name handling, by-namespace and by-family breakdowns, and a comparison to TR5's
known-frontier efficiency. Renders a decision on whether the ranker generalizes to the
fresh frontier.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

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
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    plan = {t["full_name"]: t for t in _load(args.ranked_plan)["theorems"]}
    attr = {r["full_name"]: r for r in _load(args.attribution)["records"]}
    b5 = _load(args.b5)
    b10 = _load(args.b10)
    b20 = _load(args.b20)

    searched = [r["full_name"] for r in b5["results"]] if b5 else []
    n = len(searched)

    # programs attempted by stage
    def progs(d, field="results"):
        if not d:
            return 0
        rows = d.get(field, [])
        return sum(r.get("programs_attempted", 0) for r in rows)
    b5_progs = progs(b5)
    # b10/b20 results carry merged successes; count this-stage attempts from their checkpoints
    b10_progs = _stage_attempts(args.b10)
    b20_progs = _stage_attempts(args.b20)
    b5_controls = sum(len(r.get("controls", [])) for r in b5["results"]) if b5 else 0
    total_progs = b5_progs + b10_progs + b20_progs

    # wins by budget
    def won_by(maxb):
        return [fn for fn, r in attr.items()
                if r.get("credited") and r.get("win_budget") and r["win_budget"] <= maxb]
    cred5, cred10, cred20 = won_by(5), won_by(10), won_by(20)
    all_success = [fn for fn, r in attr.items()
                   if r.get("classification") in ("FRESH_TRUE_DELTA", "BASELINE_DUPLICATE")]

    ranks = [attr[fn]["first_success_rank"] for fn in attr
             if attr[fn].get("credited") and attr[fn].get("first_success_rank")]
    mean_rank = round(sum(ranks) / len(ranks), 2) if ranks else None
    no_win = sum(1 for fn in searched if attr.get(fn, {}).get("classification") == "NO_WIN_UNDER_BUDGET")

    # unknown-name failures encountered (live)
    unk = 0
    if b5:
        for r in b5["results"]:
            for f in r.get("failures", []):
                if f.get("outcome") == "unknown_name":
                    unk += 1

    # by namespace
    by_ns = defaultdict(lambda: {"searched": 0, "credited": 0})
    for fn in searched:
        ns = attr.get(fn, {}).get("namespace")
        by_ns[ns]["searched"] += 1
        if attr.get(fn, {}).get("credited"):
            by_ns[ns]["credited"] += 1
    # by family (of winning programs)
    by_fam = Counter(attr[fn]["winning_program"]["family"] for fn in attr
                     if attr[fn].get("credited") and attr[fn].get("winning_program"))

    succ_per_probe = round(len(cred20) / max(1, total_progs), 4)

    # TR5 comparison (known-frontier efficiency)
    tr5 = _load("project/evolve/experiments/tr5/out/tr5_vs_tr3_comparison.json")
    tr5_spp = tr5.get("success_per_probe_tr5") if tr5 else None

    # ---- decision ----
    fresh_td = len(cred20)
    nonset = sum(1 for fn in cred20 if attr[fn]["namespace"] != "Set")
    if n < 30:
        decision = "INCONCLUSIVE_TOO_FEW_FRESH_FAILURES"
    elif fresh_td == 0:
        decision = "RANKER_FAILS_ON_FRESH_FRONTIER"
    elif nonset > 0:
        decision = "RANKER_GENERALIZES_TO_FRESH_FRONTIER"
    else:
        decision = "RANKER_USEFUL_WITHIN_SEEN_FAMILIES_ONLY"

    out = {
        "generated_by": "scripts/tr6_analyze_ranker_fresh_performance.py",
        "fresh_failures_searched": n,
        "programs_attempted": {"b5": b5_progs, "b10": b10_progs, "b20": b20_progs,
                               "b5_controls": b5_controls, "total": total_progs},
        "wins_by_budget": {"b5": len(cred5), "b10": len(cred10), "b20": len(cred20)},
        "credited_total": len(cred20),
        "total_live_successes": len(all_success),
        "success_per_probe": succ_per_probe,
        "tr5_success_per_probe_reference": tr5_spp,
        "mean_first_success_rank": mean_rank,
        "no_win_rate": round(no_win / max(1, n), 4),
        "unknown_name_failures_encountered": unk,
        "by_namespace": {k: v for k, v in by_ns.items()},
        "by_winning_family": dict(by_fam),
        "nonset_credited": nonset,
        "decision": decision,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR6 ranker fresh-frontier performance", "",
          f"## Decision: `{decision}`", "",
          f"- fresh failures searched: {n}",
          f"- programs attempted: B5 {b5_progs} + B10 {b10_progs} + B20 {b20_progs} "
          f"= **{total_progs}** (+{b5_controls} controls)",
          f"- credited wins by budget: B5 {len(cred5)}, B10 {len(cred10)}, B20 {len(cred20)}",
          f"- **fresh credited total: {len(cred20)}** ({nonset} non-Set)",
          f"- success/probe: **{succ_per_probe}** (TR5 reference {tr5_spp})",
          f"- mean first-success rank: {mean_rank} | no-win rate: {out['no_win_rate']}",
          f"- unknown-name failures encountered: {unk}", "",
          "## By namespace", "", "| ns | searched | credited |", "|---|---|---|"]
    for ns, v in sorted(by_ns.items(), key=lambda x: -x[1]["searched"]):
        md.append(f"| {ns} | {v['searched']} | {v['credited']} |")
    md += ["", "## By winning family", "", f"{dict(by_fam)}"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr6-perf] decision={decision} credited={len(cred20)} nonset={nonset} "
          f"spp={succ_per_probe} total_progs={total_progs}")


def _stage_attempts(path):
    """Count programs attempted in a continuation stage from its checkpoint."""
    if not path:
        return 0
    fp = _p(path) if not os.path.isabs(path) else path
    if not os.path.exists(fp):
        return 0
    d = json.load(open(fp))
    budget = d.get("budget")
    ckpt = _p(f"project/evolve/experiments/tr6/out/b{budget}_live_checkpoint.json")
    if not os.path.exists(ckpt):
        return 0
    c = json.load(open(ckpt))
    return sum(r.get("programs_attempted", 0) for r in c.values())


if __name__ == "__main__":
    main()
