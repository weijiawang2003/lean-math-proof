#!/usr/bin/env python3
"""RC2 Hardening Part 4 — formal credited-delta ledger.

Combines the comparison new-win classification, the minimal relabel, and the
forensic probes into one authoritative per-theorem ledger. Every RC2 new win is
categorized:
  credited_SET_ITE_SIMP | SX3_sequence_candidate | search_perturbation |
  timeout_variance | excluded

Outputs:
  rc2_delta_ledger.json / .md
"""
from __future__ import annotations

import argparse
import json
import os


def _load(p):
    return json.load(open(p)) if p and os.path.exists(p) else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison",
                    default="project/evolve/experiments/rc2/out/rc2_comparison.json")
    ap.add_argument("--minimal-relabel",
                    default="project/evolve/experiments/rc2/out/rc2_minimal_relabel_results.json")
    ap.add_argument("--forensics",
                    default="project/evolve/experiments/rc2_hardening/out/perturbation_forensics.json")
    ap.add_argument("--out-json",
                    default="project/evolve/experiments/rc2_hardening/out/rc2_delta_ledger.json")
    ap.add_argument("--out-md",
                    default="project/evolve/experiments/rc2_hardening/out/rc2_delta_ledger.md")
    args = ap.parse_args(argv)

    comp = _load(args.comparison) or {}
    relabel = _load(args.minimal_relabel) or {}
    forensics = _load(args.forensics) or {}

    # collect all unique new-win theorems
    wins = {}
    for r in comp.get("new_win_classification", []):
        fn = r["full_name"]
        wins.setdefault(fn, {})["comparison_class"] = r["classification"]
        wins[fn]["rc2_winning_tactic"] = r.get("rc2_winning_tactic")
        wins[fn]["rc2_num_steps"] = r.get("rc2_num_steps")
        wins[fn]["single_shot_set_ite"] = r.get("single_shot_set_ite")
    relabel_by = {r["full_name"]: r for r in relabel.get("rows", [])}
    forensic_by = {r["full_name"]: r for r in forensics.get("results", [])}

    ledger = []
    for fn, w in wins.items():
        rl = relabel_by.get(fn, {})
        fo = forensic_by.get(fn, {})
        single_shot = w.get("single_shot_set_ite")
        if single_shot:
            cat, decision, reason = ("credited_SET_ITE_SIMP", "credit",
                                     "single-shot simp [Set.ite] closes it; literal RC1 and "
                                     "all baselines fail (minimal relabel TRUE_SET_ITE_SIMP_WIN)")
        elif fo:
            cs = fo.get("credit_status")
            if cs == "sx3_sequence_candidate":
                cat, decision, reason = ("SX3_sequence_candidate", "defer",
                                         "simp [Set.ite] <;> aesop/simp_all closes it (depth-2); "
                                         "bare baselines + single-shot simp[Set.ite] do not -> SX3")
            elif cs == "credited":
                cat, decision, reason = ("credited_SET_ITE_SIMP", "credit",
                                         "forensic single-shot simp [Set.ite] closes it")
            else:
                cat, decision, reason = ("search_perturbation", "exclude", fo.get("reason", ""))
        else:
            cat, decision, reason = ("search_perturbation", "exclude",
                                     "multi-step win, no single-shot simp[Set.ite] evidence")
        ledger.append({
            "full_name": fn, "category": cat,
            "rc1_status": "failed", "rc2_status": "finished",
            "minimal_relabel": rl.get("attribution"),
            "direct_probe_result": (fo.get("direct_probes") if fo else None),
            "rc2_winning_tactic": w.get("rc2_winning_tactic"),
            "rc2_num_steps": w.get("rc2_num_steps"),
            "credit_decision": decision, "reason": reason,
        })
    ledger.sort(key=lambda r: (r["credit_decision"] != "credit", r["full_name"]))

    credited = sorted({r["full_name"] for r in ledger if r["credit_decision"] == "credit"})
    deferred = sorted({r["full_name"] for r in ledger if r["credit_decision"] == "defer"})
    excluded = sorted({r["full_name"] for r in ledger if r["credit_decision"] == "exclude"})
    cathist = {}
    for r in ledger:
        cathist[r["category"]] = cathist.get(r["category"], 0) + 1

    out = {"credited_delta": len(credited), "credited": credited,
           "deferred_sx3": deferred, "excluded": excluded,
           "category_histogram": cathist,
           "policy": "Official RC2 credited delta counts ONLY credited_SET_ITE_SIMP "
                     "(single-shot, literal-RC1-confirmed, minimal-relabel TRUE). "
                     "SX3_sequence_candidate -> deferred to SX3. search_perturbation -> "
                     "excluded.",
           "ledger": ledger}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), ensure_ascii=False, indent=2)

    L = ["# RC2 Credited-Delta Ledger", ""]
    L.append(f"- **credited delta = {len(credited)}**: {credited}")
    L.append(f"- deferred (SX3 sequence candidates): {deferred}")
    L.append(f"- excluded (search perturbation): {excluded}")
    L.append(f"- category histogram: `{cathist}`")
    L.append(f"- policy: {out['policy']}")
    L.append("")
    L.append("| theorem | category | minimal_relabel | win tactic (steps) | decision | reason |")
    L.append("|---|---|---|---|---|---|")
    for r in ledger:
        L.append(f"| `{r['full_name']}` | {r['category']} | {r['minimal_relabel']} | "
                 f"`{r['rc2_winning_tactic']}` ({r['rc2_num_steps']}) | "
                 f"**{r['credit_decision']}** | {r['reason'][:60]} |")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[rc2h:ledger] credited={len(credited)} deferred_sx3={len(deferred)} "
          f"excluded={len(excluded)} cats={cathist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
