#!/usr/bin/env python3
"""FLI3 Part 4 — conservative gates for failure-derived deployment families.

Pure classifier `gate(item)` → {gate, candidate_family, lemma, action_templates, reason, risk}.
Families: FINSET_CARD_BRIDGE / FINSET_MEM_DEF_UNFOLD / LIST_DEF_UNFOLD. Fires only on namespace +
trigger-constant + matching lemma. Hard constraints: no broad namespace-only firing, no unknown
lemma, no root/Order namespace, no simp_all, no bare aesop credited, no depth-3 chains.
"""
from __future__ import annotations

import re

DEF_TOKS = ("filterMap", "map", "preimage", "subtype")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def _has(text, tok):
    return tok.lower() in (text or "").lower()


def gate(item):
    theorem = item.get("theorem", "")
    ns = (item.get("namespace") or "").split(".")[0]
    stmt = item.get("statement") or ""
    fam = item.get("candidate_family")
    given_lemma = item.get("lemma")
    text = theorem + " " + stmt
    no = {"theorem": theorem, "candidate_family": fam, "lemma": None, "gate": False,
          "reason": "", "risk": "low", "action_templates": []}

    # hard namespace constraints
    if ns not in ("Finset", "List"):
        no["reason"] = f"namespace {ns} not in {{Finset,List}}"
        return no

    if fam == "FINSET_CARD_BRIDGE":
        if ns != "Finset" or not _has(text, "card"):
            no["reason"] = "no Finset card trigger"
            return no
        L = given_lemma if (given_lemma and given_lemma.startswith("Finset.") and "card" in given_lemma) \
            else "Finset.card_le_one"
        # constant overlap: the lemma's core token appears in the goal
        if not _has(text, "card"):
            no["reason"] = "card constant not in goal"
            return no
        return {"theorem": theorem, "candidate_family": fam, "lemma": L, "gate": True,
                "reason": "Finset + card + Finset.card_* lemma with constant overlap",
                "risk": "low", "action_templates": [f"simp [{L}]", f"simp [{L}] <;> aesop"]}

    if fam == "FINSET_MEM_DEF_UNFOLD":
        if ns != "Finset":
            no["reason"] = "not Finset"
            return no
        present = [d for d in DEF_TOKS if d in theorem or _has(stmt, d)]
        if not present:
            no["reason"] = "no filterMap/map/preimage/subtype trigger"
            return no
        d = present[0]
        L = given_lemma if (given_lemma and given_lemma == f"Finset.{d}") else f"Finset.{d}"
        acts = [f"simp [{L}]", f"simp [{L}] <;> aesop"]
        if "=" in stmt and "↔" not in stmt:  # equality goal → allow ext
            acts.append(f"ext x <;> simp [{L}]")
        return {"theorem": theorem, "candidate_family": fam, "lemma": L, "gate": True,
                "reason": f"Finset + {d} + Finset.{d} definition unfold", "risk": "low",
                "action_templates": acts}

    if fam == "LIST_DEF_UNFOLD":
        if ns != "List" or "bidirectionalRec" not in text:
            no["reason"] = "no List bidirectionalRec trigger"
            return no
        L = "List.bidirectionalRec"
        return {"theorem": theorem, "candidate_family": fam, "lemma": L, "gate": True,
                "reason": "List + bidirectionalRec definition unfold", "risk": "medium",
                "action_templates": [f"simp [{L}]", f"simp [{L}] <;> aesop"]}

    no["reason"] = f"unknown family {fam}"
    return no


if __name__ == "__main__":
    import json
    import sys
    items = json.load(open(sys.argv[1]))["items"] if len(sys.argv) > 1 else []
    for it in items:
        print(gate(it))
