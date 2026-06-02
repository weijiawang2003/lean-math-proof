#!/usr/bin/env python3
"""FLI2 Part 3 — generate small, gated lemma-deployment actions per (theorem, retrieved lemma).

Action templates: SIMPLE_SIMP / SIMP_AESOP / CONSTRUCTOR_SIMP / EXT_SIMP / INTRO_SIMP_AESOP /
EXACT_LEMMA / GCONGR_CLOSER / OMEGA_CLOSER. Banned: simp_all, bare aesop as credited deployment,
depth-3 chains, B20 search, unknown-namespace firing, unknown lemmas, long induction. Each action
is gated by namespace + constant overlap + pattern compatibility. Max 8 actions/theorem (hard 12).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOK = re.compile(r"[A-Za-z][A-Za-z0-9]+")
# pattern → list of (template, risk) using a lemma L
PATTERN_TEMPLATES = {
    "MEMBERSHIP_BRIDGE": [("SIMP_AESOP", "medium"), ("SIMPLE_SIMP", "low"),
                          ("CONSTRUCTOR_SIMP", "medium")],
    "IFF_SPLIT": [("CONSTRUCTOR_SIMP", "medium"), ("SIMP_AESOP", "medium"), ("SIMPLE_SIMP", "low")],
    "SINGLETON_CHARACTERIZATION": [("SIMP_AESOP", "medium"), ("SIMPLE_SIMP", "low"),
                                   ("CONSTRUCTOR_SIMP", "medium")],
    "SUBSET_BRIDGE": [("SIMPLE_SIMP", "low"), ("INTRO_SIMP_AESOP", "medium"),
                      ("EXACT_LEMMA", "low")],
    "MAP_FILTER_BIND_BRIDGE": [("SIMP_AESOP", "medium"), ("SIMPLE_SIMP", "low")],
    "DISJOINT_BRIDGE": [("SIMP_AESOP", "medium"), ("SIMPLE_SIMP", "low")],
    "EXTENSIONALITY_NEEDED": [("EXT_SIMP", "medium"), ("SIMPLE_SIMP", "low")],
    "INDUCTION_GENERALIZATION": [("SIMPLE_SIMP", "low"), ("SIMP_AESOP", "medium")],
}
# L-free closers gated by pattern/namespace
def _closers(item):
    out = []
    goal = (item.get("residual_goal") or item.get("statement") or "")
    pat = item.get("primary_pattern")
    if pat == "SUBSET_BRIDGE" or any(s in goal for s in ("⊆", "card", "≤")):
        out.append(("GCONGR_CLOSER", "gcongr", "low"))
    if (item.get("namespace") or "").split(".")[0] == "Nat" or "ℕ" in goal:
        out.append(("OMEGA_CLOSER", "omega", "low"))
    return out


def _tactic(template, L):
    return {
        "SIMPLE_SIMP": f"simp [{L}]",
        "SIMP_AESOP": f"simp [{L}] <;> aesop",
        "CONSTRUCTOR_SIMP": f"constructor <;> intro h <;> simp [{L}] at *",
        "EXT_SIMP": f"ext x <;> simp [{L}]",
        "INTRO_SIMP_AESOP": f"intro x hx <;> simp [{L}] at * <;> aesop",
        "EXACT_LEMMA": f"exact {L}",
    }[template]


def _lemma_ns(L):
    return L.split(".")[0] if "." in (L or "") else ""


def _core_tokens(L):
    # tokens of the lemma's short name (drop namespace)
    short = (L or "").split(".")[-1]
    return {t.lower() for t in _TOK.findall(short) if len(t) > 2}


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    ap.add_argument("--max-actions-per-theorem", type=int, default=8)
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(_p(args.pool)) if l.strip()]
    HARD = 12
    actions = []
    n = 0
    for item in pool:
        ns_root = (item.get("namespace") or "").split(".")[0]
        pat = item.get("primary_pattern")
        goal = (item.get("residual_goal") or item.get("statement") or "")
        goal_l = goal.lower()
        templates = PATTERN_TEMPLATES.get(pat, [("SIMPLE_SIMP", "low"), ("SIMP_AESOP", "medium")])
        lemmas = [L for L in (item.get("candidate_existing_lemmas") or []) if L][:3]
        per = []
        for L in lemmas:
            lns = _lemma_ns(L)
            ns_compat = (lns == ns_root) or (lns in ("", ns_root)) or (lns and ns_root.startswith(lns))
            if not ns_compat:
                continue
            core = _core_tokens(L)
            overlap = bool(core & {t.lower() for t in _TOK.findall(goal)}) or any(
                c in goal_l for c in core)
            for (template, risk) in templates:
                if template == "EXACT_LEMMA":
                    tac_variants = [f"exact {L}", f"exact {L} h", f"simpa using {L}"]
                else:
                    tac_variants = [_tactic(template, L)]
                for tac in tac_variants:
                    per.append({
                        "lemma": L, "template": template, "tactic": tac, "risk": risk,
                        "ns_compat": ns_compat, "overlap": overlap,
                        "gate_reason": f"ns {lns or 'root'}~{ns_root}; "
                                       f"{'const-overlap' if overlap else 'no-overlap'}; pattern {pat}",
                    })
        # L-free closers
        for (template, tac, risk) in _closers(item):
            per.append({"lemma": None, "template": template, "tactic": tac, "risk": risk,
                        "ns_compat": True, "overlap": True,
                        "gate_reason": f"L-free closer for {pat}/{ns_root}"})
        # rank: prefer overlap, low risk, simpler templates; cap
        order = {"SIMPLE_SIMP": 0, "EXACT_LEMMA": 1, "GCONGR_CLOSER": 1, "OMEGA_CLOSER": 1,
                 "SIMP_AESOP": 2, "EXT_SIMP": 2, "CONSTRUCTOR_SIMP": 3, "INTRO_SIMP_AESOP": 3}
        per.sort(key=lambda a: (not a["overlap"], {"low": 0, "medium": 1, "high": 2}[a["risk"]],
                                order.get(a["template"], 9), a["tactic"]))
        cap = min(args.max_actions_per_theorem, HARD)
        for a in per[:cap]:
            n += 1
            actions.append({
                "action_id": f"FLI2-A{n:04d}", "case_id": item["case_id"],
                "theorem": item["theorem"], "namespace": item["namespace"],
                "file_path": item.get("file_path"), "lemma": a["lemma"],
                "template": a["template"], "tactic": a["tactic"],
                "gate_reason": a["gate_reason"], "risk": a["risk"],
                "priority": item["priority"], "source": item["source"],
                "expected_pattern": pat,
            })

    actions.sort(key=lambda a: ({"high": 0, "medium": 1, "low": 2}[a["priority"]],
                                a["namespace"] or "", a["case_id"], a["action_id"]))
    with open(_p(args.out_jsonl), "w") as f:
        for a in actions:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    thms = {a["theorem"] for a in actions}
    summary = {"generated_by": "scripts/fli2_generate_deployment_actions.py",
               "num_actions": len(actions), "num_theorems": len(thms),
               "by_template": dict(Counter(a["template"] for a in actions).most_common()),
               "by_risk": dict(Counter(a["risk"] for a in actions)),
               "by_priority": dict(Counter(a["priority"] for a in actions)),
               "by_namespace": dict(Counter(a["namespace"] for a in actions).most_common()),
               "avg_actions_per_theorem": round(len(actions) / max(1, len(thms)), 2)}
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI2 deployment action summary", "",
          f"- actions: {summary['num_actions']} over {summary['num_theorems']} theorems "
          f"(avg {summary['avg_actions_per_theorem']}/thm)",
          f"- by template: {summary['by_template']}",
          f"- by risk: {summary['by_risk']} | by priority: {summary['by_priority']}",
          f"- by namespace: {summary['by_namespace']}", ""]
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli2-actions] actions={len(actions)} theorems={len(thms)} "
          f"templates={summary['by_template']}")


if __name__ == "__main__":
    main()
