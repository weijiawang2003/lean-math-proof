#!/usr/bin/env python3
"""FLI1 Part 5 — synthesize small, local candidate intermediate lemmas from residual goals.

For each captured residual goal we reconstruct a standalone Lean statement: the hypothesis block
becomes binders and the `⊢` goal becomes the conclusion (so the candidate is "given this context,
the residual goal holds" = exactly the intermediate lemma the search got stuck on). Type names come
from the real goal, so the statement reuses valid Mathlib vocabulary. Inaccessible names (`inst✝`,
daggers) are sanitized; `Type u_k` → `Type*`. Each candidate is tied to its downstream seed and
import module. Candidates are deliberately small (prefer the implication/iff the search needs).
"""
from __future__ import annotations

import argparse
import json
import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = "project/evolve/experiments/fli1/cases/fli1_live_rerun_plan.json"
_DAGGER = re.compile(r"✝\d*")


def _p(*a):
    return os.path.join(_REPO, *a)


def _parse_first_goal(pp):
    """Return (hyp_lines, goal_str) for the first goal in a pp dump."""
    if not pp or "⊢" not in pp:
        return [], (pp or "").strip()
    # the first goal segment ends at the next 'case ' block
    seg = pp.split("\ncase ")[0]
    # drop a leading 'case xxx' marker on the very first line
    seg = re.sub(r"^case \w+\n", "", seg)
    head, _, goal = seg.partition("⊢")
    hyp_lines = [ln.rstrip() for ln in head.splitlines() if ln.strip()]
    return hyp_lines, goal.strip()


def _sanitize(s):
    return _DAGGER.sub("", s)


def _binders(hyp_lines):
    """Convert pp hyp lines into Lean binder strings. Type vars → {names : Type*}; instance
    (inst✝) → [type]; rest → (names : type). Inaccessible names sanitized to aᵢ."""
    binders = []
    anon = 0
    for ln in hyp_lines:
        if ":" not in ln:
            continue
        names_part, _, typ = ln.partition(":")
        names = names_part.split()
        typ = _sanitize(typ).strip()
        typ = re.sub(r"\bType u_\d+\b|\bType\b", "Type*", typ)
        typ = re.sub(r"\bSort u_\d+\b", "Sort*", typ)
        clean_names = []
        for nm in names:
            nm = _sanitize(nm)
            if not nm or nm == "inst":
                anon += 1
                nm = None  # instance / anonymous
            clean_names.append(nm)
        is_inst = any(n is None for n in clean_names) or names_part.strip().startswith("inst")
        if "Type" in typ or "Sort" in typ:
            nm = " ".join(n for n in clean_names if n) or f"t{anon}"
            binders.append(("type", f"{{{nm} : {typ}}}"))
        elif is_inst:
            binders.append(("inst", f"[{typ}]"))
        else:
            nm = " ".join((n if n else f"a{anon}") for n in clean_names)
            binders.append(("hyp", f"({nm} : {typ})"))
    return binders


def _confidence_risk(pattern, goal):
    bridge = pattern in ("MEMBERSHIP_BRIDGE", "SUBSET_BRIDGE", "DISJOINT_BRIDGE",
                         "MAP_FILTER_BIND_BRIDGE", "IFF_SPLIT")
    short = len(goal) <= 160
    if bridge and short:
        return "high", "low"
    if bridge:
        return "medium", "medium"
    if pattern in ("INDUCTION_GENERALIZATION", "EXTENSIONALITY_NEEDED"):
        return "medium", "medium"
    return "low", "medium"


EXPECTED_TACTIC = {
    "MEMBERSHIP_BRIDGE": "simp", "SUBSET_BRIDGE": "intro <;> simp",
    "DISJOINT_BRIDGE": "simp [disjoint_left]", "MAP_FILTER_BIND_BRIDGE": "simp",
    "IFF_SPLIT": "constructor <;> simp", "SINGLETON_CHARACTERIZATION": "simp",
    "EXTENSIONALITY_NEEDED": "ext x <;> simp", "INDUCTION_GENERALIZATION": "induction <;> simp",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--residual-goals", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    plan = {s["seed_id"]: s for s in json.load(open(_p(PLAN)))["seeds"]}
    resid = {r["seed_id"]: r for r in (json.loads(l) for l in open(_p(args.residual_goals)) if l.strip())}
    clusters = json.load(open(_p(args.clusters)))["clusters"]
    cl_by_seed = {sid: c["cluster_id"] for c in clusters for sid in c["seed_ids"]}

    from collections import Counter
    candidates = []
    n = 0
    for sid, r in resid.items():
        if r["status"] != "captured" or not r.get("residual_goals"):
            continue
        seed = plan.get(sid, {})
        pat = seed.get("primary_pattern", "UNKNOWN")
        ns = (r["namespace"] or "").split(".")[0]
        module = seed.get("import_module")
        pp = r["residual_goals"][0]
        hyp_lines, goal = _parse_first_goal(pp)
        goal = _sanitize(goal)
        goal = re.sub(r"\bu_\d+\b", "", goal)
        goal = re.sub(r"\s+", " ", goal).strip()
        if not goal:
            continue
        binders = _binders(hyp_lines)
        binder_str = " ".join(b for _, b in binders)
        n += 1
        name = f"fli1_{ns.lower()}_{sid.replace('-', '_').lower()}_aux"
        conf, risk = _confidence_risk(pat, goal)
        candidates.append({
            "candidate_id": f"FLI1-L{n:02d}",
            "cluster_id": cl_by_seed.get(sid),
            "source_seed_ids": [sid],
            "namespace": ns, "pattern": pat,
            "lemma_name_suggestion": name,
            "lemma_statement_natural_language":
                f"Under the residual context of `{r['theorem']}` (after "
                f"`{' ; '.join(r.get('last_successful_tactic_prefix') or [])}`), the goal `{goal[:120]}` holds.",
            "lemma_statement_lean": f"lemma {name} {binder_str} : {goal} := by sorry".strip(),
            "lemma_binders": binder_str, "lemma_goal": goal,
            "residual_have_type": goal,  # for at-position `have` inlining in rescue
            "required_imports": [module] if module else [],
            "open_namespaces": [ns],
            "expected_tactic": EXPECTED_TACTIC.get(pat, "simp"),
            "confidence": conf, "risk": risk,
            "downstream_targets": [r["theorem"]],
            "why_it_might_help": (f"closing this residual goal is exactly the step the {pat} "
                                  f"search stalled on; as a reusable {ns} bridge it would let a "
                                  f"gated `simp [..]` discharge `{r['theorem']}`."),
            "prefix_to_reach": r.get("last_successful_tactic_prefix", []),
        })

    # keep candidates tied to downstream targets (drop empty), prefer bridge patterns
    candidates = [c for c in candidates if c["downstream_targets"] and c["lemma_goal"]]
    with open(_p(args.out_jsonl), "w") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    summary = {"generated_by": "scripts/fli1_synthesize_candidate_lemmas.py",
               "num_candidates": len(candidates),
               "by_pattern": dict(Counter(c["pattern"] for c in candidates).most_common()),
               "by_namespace": dict(Counter(c["namespace"] for c in candidates).most_common()),
               "by_confidence": dict(Counter(c["confidence"] for c in candidates)),
               "high_conf_low_risk": sum(1 for c in candidates
                                         if c["confidence"] == "high" and c["risk"] == "low")}
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI1 candidate lemma summary", "",
          f"- candidates: {summary['num_candidates']} | high-conf/low-risk: "
          f"{summary['high_conf_low_risk']}",
          f"- by pattern: {summary['by_pattern']}",
          f"- by namespace: {summary['by_namespace']} | confidence: {summary['by_confidence']}", "",
          "| id | seed | ns | pattern | conf | lemma goal |", "|---|---|---|---|---|---|"]
    for c in candidates:
        md.append(f"| {c['candidate_id']} | {c['source_seed_ids'][0]} | {c['namespace']} | "
                  f"{c['pattern']} | {c['confidence']} | `{c['lemma_goal'][:80]}` |")
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli1-synth] candidates={len(candidates)} by_pattern={summary['by_pattern']} "
          f"high_conf_low_risk={summary['high_conf_low_risk']}")


if __name__ == "__main__":
    main()
