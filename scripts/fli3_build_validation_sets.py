#!/usr/bin/env python3
"""FLI3 Part 3 — build focused validation sets.

rescue_replay (6) / family_holdout (same ns+constant+pattern) / offgate_negative (gate must not
fire) / canonical_floor + regression_guard (RC2-solvable guards; preservation by additive design).
Small and meaningful (target 40-120 theorem/action items), not a broad benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD_TOK = "card"
DEF_TOKS = ("filterMap", "map", "preimage", "subtype")
TRIGGER_TOKS = ("card", "filterMap", "map", "preimage", "subtype", "bidirectionalRec", "mem")
CATALOG = "project/discovered_theorems.json"


def _p(*a):
    return os.path.join(_REPO, *a)


def _rows(path):
    return [json.loads(l) for l in open(_p(path)) if l.strip()] if os.path.exists(_p(path)) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--fli0-enriched", required=True)
    ap.add_argument("--fli2-pool", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    cand = json.load(open(_p(args.candidates)))
    rescues = cand["rescue_candidates"]
    rescue_thms = {c["theorem"] for c in rescues}
    enriched = {e["theorem"]: e for e in _rows(args.fli0_enriched)}
    pool = {p["theorem"]: p for p in _rows(args.fli2_pool)}
    allcases = {**pool, **{t: {"theorem": t, "namespace": e.get("namespace"),
                               "statement": e.get("statement"), "file_path": e.get("file_path"),
                               "primary_pattern": e.get("primary_pattern" if False else None)}
                          for t, e in enriched.items()}}
    # prefer pool entry (richer) where both exist
    for t, p in pool.items():
        allcases[t] = p

    items = []

    def add(setname, theorem, family, lemma, expected_gate, why, src="FLI2/FLI0"):
        c = allcases.get(theorem, {})
        e = enriched.get(theorem, {})
        items.append({
            "set": setname, "theorem": theorem,
            "namespace": c.get("namespace") or e.get("namespace"),
            "statement": c.get("statement") or e.get("statement"),
            "file_path": c.get("file_path") or e.get("file_path"),
            "candidate_family": family, "lemma": lemma, "expected_gate": expected_gate,
            "rc2_result": e.get("rc2_result", "unknown"), "source": src,
            "source_stage": e.get("source_stage"), "why": why,
        })

    # 1. rescue_replay
    for c in rescues:
        add("rescue_replay", c["theorem"], c["candidate_family"], c["lemma"], True,
            "FLI2 robust rescue (theorem,action) replay", "FLI2_TRUE_RESCUE")

    # 2. family_holdout — same ns + constant family, not a rescue
    def stmt(t):
        return (allcases.get(t, {}).get("statement") or enriched.get(t, {}).get("statement") or "")
    card_hold, def_hold, list_hold = [], [], []
    for t, e in enriched.items():
        if t in rescue_thms or not e.get("clean_failure"):
            continue
        ns = (e.get("namespace") or "").split(".")[0]
        s = e.get("statement") or ""
        rl = [x.get("lemma") for x in (e.get("top_retrieved_lemmas_detailed") or []) if x.get("lemma")]
        if ns == "Finset" and CARD_TOK in t.lower() and CARD_TOK in s.lower() and len(card_hold) < 8:
            L = next((x for x in rl if x and "card" in x.lower() and x.startswith("Finset.")), "Finset.card_le_one")
            card_hold.append((t, L))
        elif ns == "Finset" and any(d in t for d in DEF_TOKS) and len(def_hold) < 8:
            d = next(d for d in DEF_TOKS if d in t)
            def_hold.append((t, f"Finset.{d}"))
        elif ns == "List" and "bidirectionalRec" in t and len(list_hold) < 4:
            list_hold.append((t, "List.bidirectionalRec"))
    for t, L in card_hold:
        add("family_holdout", t, "FINSET_CARD_BRIDGE", L, True, "Finset card-family holdout")
    for t, L in def_hold:
        add("family_holdout", t, "FINSET_MEM_DEF_UNFOLD", L, True, "Finset def-unfold-family holdout")
    for t, L in list_hold:
        add("family_holdout", t, "LIST_DEF_UNFOLD", L, True, "List bidirectionalRec holdout")

    # 3. offgate_negative — Finset/List failures lacking ALL trigger constants
    neg = 0
    for t, e in enriched.items():
        if neg >= 15 or t in rescue_thms:
            continue
        ns = (e.get("namespace") or "").split(".")[0]
        s = (e.get("statement") or "").lower()
        if ns in ("Finset", "List") and e.get("clean_failure") and \
                not any(tok.lower() in s or tok.lower() in t.lower() for tok in TRIGGER_TOKS):
            fam = "FINSET_MEM_DEF_UNFOLD" if ns == "Finset" else "LIST_DEF_UNFOLD"
            add("offgate_negative", t, fam, None, False,
                f"{ns} theorem lacking card/map/filterMap/preimage/subtype/mem/bidirectionalRec")
            neg += 1

    # 4 & 5. canonical_floor + regression_guard — low-difficulty catalog theorems (RC2-solvable guards)
    cat = json.load(open(_p(CATALOG))) if os.path.exists(_p(CATALOG)) else {}
    cat_t = sorted([t for t in cat.get("theorems", [])
                    if t.get("has_tactic_proof") and (t.get("difficulty_score") or 9) <= 2
                    and t["full_name"].split(".")[0] in ("Finset", "List", "Nat")
                    and t["full_name"] not in rescue_thms],
                   key=lambda x: (x.get("difficulty_score", 9), x["full_name"]))
    for i, t in enumerate(cat_t[:16]):
        setname = "canonical_floor" if i % 2 == 0 else "regression_guard"
        fam = ("FINSET_MEM_DEF_UNFOLD" if t["full_name"].startswith("Finset")
               else ("LIST_DEF_UNFOLD" if t["full_name"].startswith("List") else "FINSET_CARD_BRIDGE"))
        items.append({"set": setname, "theorem": t["full_name"],
                      "namespace": t["full_name"].split(".")[0],
                      "statement": None, "file_path": t.get("file_path"),
                      "candidate_family": fam, "lemma": None, "expected_gate": False,
                      "rc2_result": "solved_assumed", "source": "catalog_low_difficulty",
                      "why": "RC2-solvable guard; preservation by additive design (gate must not fire)"})

    items.sort(key=lambda r: (r["set"], r["namespace"] or "", r["theorem"]))
    out = {"generated_by": "scripts/fli3_build_validation_sets.py",
           "num_items": len(items), "items": items}
    with open(_p(args.out_json), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    summary = {"generated_by": "scripts/fli3_build_validation_sets.py", "num_items": len(items),
               "by_set": dict(Counter(r["set"] for r in items)),
               "by_family": dict(Counter(r["candidate_family"] for r in items).most_common()),
               "by_namespace": dict(Counter(r["namespace"] for r in items).most_common()),
               "expected_gate_true": sum(1 for r in items if r["expected_gate"]),
               "expected_gate_false": sum(1 for r in items if not r["expected_gate"])}
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI3 validation set summary", "",
          f"- total items: {summary['num_items']}",
          f"- by set: {summary['by_set']}",
          f"- by family: {summary['by_family']}",
          f"- by namespace: {summary['by_namespace']}",
          f"- expected gate fire: {summary['expected_gate_true']} | no-fire: "
          f"{summary['expected_gate_false']}", ""]
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli3-sets] items={len(items)} by_set={summary['by_set']}")


if __name__ == "__main__":
    main()
