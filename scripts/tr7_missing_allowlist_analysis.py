#!/usr/bin/env python3
"""TR7 Part 6 — missing allowlist analysis.

For every TR6 fresh win, examine its winning lemma: whether it is in the RC4D component
manifest / RC4R wrapper, which known family it belongs to (disjoint_left / disjoint_right /
subset_pair / forall_mem / def_unfold / other), how often it appears across the TR6 ranked
program plan, and whether it has namespace-parametric analogues. Recommend (analysis only, no
additions): ADD_TO_STATIC_ALLOWLIST / KEEP_DYNAMIC_ONLY / NEED_MORE_EVIDENCE / REJECT_SOURCE_SPECIFIC.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RC4_ALLOWLIST = {
    "Monotone", "MonotoneOn", "Antitone", "AntitoneOn", "StrictMono", "StrictMonoOn",
    "StrictAnti", "StrictAntiOn", "Finset.disjUnion", "Set.disjoint_left",
    "Multiset.disjoint_left", "Multiset.disjoint_right", "Set.subset_pair_iff_eq",
    "List.forall_iff_forall_mem"}


def _p(*a):
    return os.path.join(_REPO, *a)


def _family(lemma):
    if not lemma:
        return "none_tauto"
    l = lemma.lower()
    if "disjoint_left" in l:
        return "disjoint_left"
    if "disjoint_right" in l:
        return "disjoint_right"
    if "subset_pair" in l:
        return "subset_pair"
    if "forall" in l:
        return "forall_mem"
    if any(d.lower() in l for d in ("monotone", "antitone", "mapsto", "disjunion")):
        return "def_unfold"
    if "subset" in l:
        return "subset"
    if "add_eq_union" in l:
        return "add_eq_union"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tr6-plan", required=True)
    ap.add_argument("--tr6-attribution", required=True)
    ap.add_argument("--rc4-manifest", required=True)
    ap.add_argument("--rc4-wrapper", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    attr = json.load(open(_p(args.tr6_attribution)))
    wins = [r for r in attr["records"] if r["classification"] == "FRESH_TRUE_DELTA"]
    wrapper = json.load(open(_p(args.rc4_wrapper)))
    wrapper_lemmas = set()
    for t in wrapper.get("priority_templates", {}).get("any", []):
        import re
        m = re.search(r"simp \[([^\]]+)\]", t)
        if m:
            for L in m.group(1).split(","):
                wrapper_lemmas.add(L.strip())
    manifest = json.load(open(_p(args.rc4_manifest)))
    manifest_lemmas = set(RC4_ALLOWLIST)

    # lemma frequency across the full ranked plan (multiplicity / parametric signal)
    plan = json.load(open(_p(args.tr6_plan)))
    lemma_freq = Counter()
    for th in plan.get("theorems", []):
        for prog in th.get("programs", []) or th.get("ranked_programs", []) or []:
            for L in (prog.get("used_lemmas") or []):
                lemma_freq[L] += 1
    # win-lemma frequency
    win_lemma_freq = Counter()
    for w in wins:
        L = (w.get("winning_program") or {}).get("used_lemmas") or [None]
        win_lemma_freq[L[0]] += 1

    records = []
    for w in wins:
        wp = w.get("winning_program") or {}
        lemma = (wp.get("used_lemmas") or [None])[0]
        fam = _family(lemma)
        in_manifest = bool(lemma and (lemma in manifest_lemmas or lemma.split(".")[-1] in manifest_lemmas))
        in_wrapper = bool(lemma and (lemma in wrapper_lemmas or lemma.split(".")[-1] in wrapper_lemmas))
        win_count = win_lemma_freq[lemma]
        plan_count = lemma_freq.get(lemma, 0)
        # namespace-parametric analogue: does the same short-name appear in >1 namespace among wins?
        short = lemma.split(".")[-1] if lemma else None
        parametric = short is not None and sum(
            1 for w2 in wins
            if (((w2.get("winning_program") or {}).get("used_lemmas") or [None])[0] or "").split(".")[-1] == short
        ) > 1

        if in_wrapper or in_manifest:
            rec = "ALREADY_IN_ALLOWLIST"
        elif fam in ("none_tauto",) or wp.get("family") in ("d1_tauto", "d1_exact", "d1_rw", "d2_rw_aesop"):
            rec = "KEEP_DYNAMIC_ONLY"        # won via theorem-specific rw/exact/tauto
        elif fam in ("disjoint_left", "disjoint_right", "forall_mem", "subset_pair", "def_unfold") and (parametric or win_count > 1):
            rec = "ADD_TO_STATIC_ALLOWLIST"  # clean family lemma, recurs / parametric
        elif fam in ("subset", "add_eq_union", "def_unfold") and win_count == 1:
            rec = "NEED_MORE_EVIDENCE"       # single clean-lemma occurrence
        else:
            rec = "NEED_MORE_EVIDENCE"
        records.append({
            "full_name": w["full_name"], "namespace": w["namespace"], "winning_lemma": lemma,
            "winning_family": wp.get("family"), "lemma_family": fam,
            "in_rc4_manifest": in_manifest, "in_rc4_wrapper": in_wrapper,
            "win_occurrences": win_count, "plan_occurrences": plan_count,
            "namespace_parametric": parametric, "recommendation": rec,
        })

    hist = Counter(r["recommendation"] for r in records)
    add = [r for r in records if r["recommendation"] == "ADD_TO_STATIC_ALLOWLIST"]
    dyn = [r for r in records if r["recommendation"] == "KEEP_DYNAMIC_ONLY"]
    more = [r for r in records if r["recommendation"] == "NEED_MORE_EVIDENCE"]
    already = [r for r in records if r["recommendation"] == "ALREADY_IN_ALLOWLIST"]
    out = {
        "generated_by": "scripts/tr7_missing_allowlist_analysis.py",
        "num_wins": len(wins), "recommendation_histogram": dict(hist),
        "already_in_allowlist": len(already),
        "add_to_static_allowlist": [r["winning_lemma"] for r in add],
        "keep_dynamic_only": [{"thm": r["full_name"], "family": r["winning_family"]} for r in dyn],
        "need_more_evidence": [r["winning_lemma"] for r in more],
        "missing_lemmas_by_family": dict(Counter(r["lemma_family"] for r in records if not r["in_rc4_wrapper"])),
        "records": records,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR7 missing allowlist analysis", "",
          f"- TR6 fresh wins: {len(wins)} | recommendations: {dict(hist)}",
          f"- already in allowlist: {len(already)}",
          f"- ADD_TO_STATIC_ALLOWLIST: {len(add)} {out['add_to_static_allowlist']}",
          f"- KEEP_DYNAMIC_ONLY: {len(dyn)}",
          f"- NEED_MORE_EVIDENCE: {len(more)} {out['need_more_evidence']}",
          f"- missing lemmas by family: {out['missing_lemmas_by_family']}", "",
          "| theorem | lemma | family | in_wrapper | win_occ | parametric | recommendation |",
          "|---|---|---|---|---|---|---|"]
    for r in sorted(records, key=lambda x: x["recommendation"]):
        md.append(f"| `{r['full_name']}` | `{r['winning_lemma']}` | {r['lemma_family']} | "
                  f"{r['in_rc4_wrapper']} | {r['win_occurrences']} | {r['namespace_parametric']} | "
                  f"{r['recommendation']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr7-allowlist] {dict(hist)}")
    print(f"[tr7-allowlist] add={out['add_to_static_allowlist']} need_more={out['need_more_evidence']}")


if __name__ == "__main__":
    main()
