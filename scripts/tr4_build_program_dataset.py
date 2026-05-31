#!/usr/bin/env python3
"""TR4 Part 2 — build the program-level supervised dataset.

One row per (theorem, run program) from TR3 (primary) + SF5 + RC4A. label_success =
program closed the goal live; label_credit = success AND literal-RC2 failed AND
attributed TRUE_*_DELTA / TRUE_DEF_UNFOLD_SIMP_WIN (not baseline-dup / subsumed /
source-specific). No live Lean — consumes verified outcomes only.
"""
from __future__ import annotations

import argparse
import json
import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _load(path):
    fp = _p(path)
    return json.load(open(fp)) if os.path.exists(fp) else None


def _flags(text):
    t = (text or "")
    low = t.lower()
    return {
        "has_set": ("set" in low) or ("∈" in t) or ("∪" in t) or ("∩" in t) or ("⊆" in t),
        "has_finset": "finset" in low,
        "has_list": "list" in low,
        "has_nat": ("nat" in low) or ("ℕ" in t),
        "has_iff": ("↔" in t) or ("iff" in low),
        "has_subset": ("⊆" in t) or ("⊂" in t) or ("subset" in low),
        "has_disjoint": "disjoint" in low,
        "has_compl": ("compl" in low) or ("ᶜ" in t),
        "has_singleton": ("singleton" in low) or ("{" in t),
        "has_card": "card" in low,
        "has_tofinset": "tofinset" in low,
        "has_monotone": "monotone" in low,
    }


def _ns(fn):
    return fn.split(".")[0] if "." in fn else ""


def _retrieval_map(retr):
    """target -> {lemma: (rank, score, reason, source)} and target->goal_text."""
    m, goals = {}, {}
    if not retr:
        return m, goals
    for r in retr.get("results", []):
        tgt = r.get("target")
        goals[tgt] = r.get("goal_text")
        d = {}
        for i, t in enumerate(r.get("top_lemmas", r.get("retrieved", []))):
            d[t["lemma"]] = (i, t.get("score"), t.get("reason"), t.get("source"))
        m[tgt] = d
    return m, goals


def _program_row(eid, fn, fp, ns, cid, rc2_status, prog, rmap, goal_text,
                 attribution, credited_winner, source):
    lemmas = prog.get("lemmas") or ([prog["lemma"]] if prog.get("lemma") else [])
    outcome = prog.get("outcome", "proof_failed")
    solved = bool(prog.get("solved"))
    family = prog.get("family")
    depth = prog.get("depth", 1)
    tactic = prog.get("tactic", "")
    # retrieval rank/score = best (lowest rank / highest score) among used lemmas
    rank, score, reason, lsrc = None, None, None, None
    for L in lemmas:
        if L in rmap:
            rk, sc, rs, sr = rmap[L]
            if rank is None or rk < rank:
                rank, score, reason, lsrc = rk, sc, rs, sr
    uses_retrieved = bool(lemmas) and any(L in rmap for L in lemmas)
    lemma_ns_match = any(_ns(L) == ns for L in lemmas if "." in L)
    text_for_flags = (goal_text or "") + " " + fn
    fl = _flags(text_for_flags)
    tl = tactic.lower()
    label_success = 1 if solved else 0
    if solved and credited_winner and rc2_status == "failed":
        attr = attribution
    elif solved and attribution == "BASELINE_DUPLICATE":
        attr = "BASELINE_DUPLICATE"
    elif solved:
        attr = attribution or "NEEDS_REVIEW"
    else:
        attr = "FAILED"
    label_credit = 1 if (solved and credited_winner and rc2_status == "failed"
                         and attr in ("TRUE_RETRIEVAL_ONLY_DELTA", "TRUE_RETRIEVAL_DEPTH_DELTA",
                                      "TRUE_DEF_UNFOLD_SIMP_WIN", "TRUE_DEPTH_ONLY_DELTA")) else 0
    conf = "verified" if label_credit else ("strong" if solved else "strong")
    return {
        "example_id": eid, "full_name": fn, "file_path": fp, "namespace": ns,
        "cluster_id": cid, "rc2_status": rc2_status, "source": source,
        "goal_text": goal_text,
        "program_id": prog.get("program_id", f"{fn}::{tactic[:40]}"),
        "program_family": family, "program_depth": depth, "tactic": tactic,
        "used_lemmas": lemmas, "retrieval_rank": rank, "retrieval_score": score,
        "retrieval_reason": reason, "lemma_source": lsrc,
        "outcome": outcome, "attribution": attr,
        "label_success": label_success, "label_credit": label_credit,
        "label_confidence": conf,
        "features": {
            **{k: bool(v) for k, v in fl.items()},
            "lemma_namespace_matches": bool(lemma_ns_match),
            "program_uses_retrieved_lemma": bool(uses_retrieved),
            "is_def_unfold": family == "def_unfold_simp",
            "is_depth2_aesop": family == "d2_simp_aesop",
            "is_d1_simp_lemma": family == "d1_simp_lemma",
            "uses_simp": "simp" in tl, "uses_rw": "rw " in tl or tl.startswith("rw"),
            "uses_exact": "exact" in tl, "uses_simpa": "simpa" in tl,
            "uses_aesop": "aesop" in tl, "uses_ext": tl.startswith("ext"),
            "uses_constructor": "constructor" in tl, "uses_intro": "intro" in tl,
            "uses_omega": "omega" in tl, "uses_nlinarith": "nlinarith" in tl,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    rows = []
    eid = 0

    # ---- TR3 (primary) ----
    tr3_res = _load("project/evolve/experiments/tr3/out/tr3_depth_program_results.json")
    tr3_attr = {r["full_name"]: r for r in
                _load("project/evolve/experiments/tr3/out/tr3_attribution.json")["records"]}
    tr3_rmap, tr3_goals = _retrieval_map(
        _load("project/evolve/experiments/tr3/out/tr3_retrieval_results.json"))
    for r in tr3_res["results"]:
        fn = r["full_name"]
        a = tr3_attr.get(fn, {})
        rc2_status = "failed"  # all TR3 program theorems are confirmed failures
        win_tac = (a.get("winning_program") if a.get("credited") else None)
        attr_cls = a.get("classification")
        rmap = tr3_rmap.get(fn, {})
        goal = tr3_goals.get(fn)
        for prog in r.get("ran", []):
            if prog.get("skipped"):
                continue
            credited_winner = bool(a.get("credited") and prog.get("solved")
                                   and prog.get("tactic") == win_tac)
            rows.append(_program_row(f"tr4_{eid:05d}", fn, r.get("file_path"),
                                     r.get("namespace"), r.get("cluster_id"), rc2_status,
                                     prog, rmap, goal, attr_cls, credited_winner, "tr3"))
            eid += 1

    # ---- SF5 ----
    sf5_res = _load("project/evolve/experiments/sf5/out/sf5_retrieval_probe_results.json")
    sf5_attr = {r["full_name"]: r for r in
                (_load("project/evolve/experiments/sf5/out/sf5_retrieval_attribution.json") or {}).get("records", [])}
    sf5_rmap, sf5_goals = _retrieval_map(
        _load("project/evolve/experiments/sf5/out/sf5_retrieval_results.json"))
    if sf5_res:
        for r in sf5_res["results"]:
            fn = r["full_name"]
            a = sf5_attr.get(fn, {})
            rc2_status = "failed" if r.get("rc2_status") in ("CONFIRMED_RC2_FAILURE", None) else "solved"
            attr_cls = a.get("classification")
            rmap = sf5_rmap.get(fn, {})
            goal = sf5_goals.get(fn)
            for prog in r.get("ran", []):
                credited_winner = bool(prog.get("solved")
                                       and attr_cls in ("EXISTING_LEMMA_GAP", "RETRIEVAL_ROUTING_GAP"))
                # map SF5 attribution into TR4 credit space
                mapped = ("TRUE_RETRIEVAL_ONLY_DELTA" if attr_cls == "EXISTING_LEMMA_GAP"
                          else "TRUE_RETRIEVAL_DEPTH_DELTA" if attr_cls == "RETRIEVAL_ROUTING_GAP"
                          else attr_cls)
                rows.append(_program_row(f"tr4_{eid:05d}", fn, r.get("file_path"),
                                         r.get("namespace"), r.get("cluster_id"), rc2_status,
                                         prog, rmap, goal, mapped, credited_winner, "sf5"))
                eid += 1

    # ---- RC4A gated candidate probes ----
    rc4 = _load("project/evolve/experiments/rc4_candidates/def_unfold_simp/out/candidate_results.json")
    rc4_attr = {r["full_name"]: r for r in
                (_load("project/evolve/experiments/rc4_candidates/def_unfold_simp/out/minimal_attribution.json") or {}).get("records", [])}
    if rc4:
        for r in rc4["results"]:
            if not r.get("candidate_gate_fired"):
                continue
            fn = r["full_name"]
            solved = r.get("candidate_probe_outcome") == "success"
            a = rc4_attr.get(fn, {})
            credited = bool(a.get("credited"))
            prog = {"tactic": r.get("candidate_tactic"), "family": "def_unfold_simp",
                    "depth": 1, "lemmas": r.get("matched_defs", []),
                    "solved": solved, "outcome": "success" if solved else "proof_failed"}
            rows.append(_program_row(f"tr4_{eid:05d}", fn, r.get("file_path"),
                                     r.get("namespace"), None,
                                     "failed" if not r.get("rc2_finished") else "solved",
                                     prog, {}, None,
                                     "TRUE_DEF_UNFOLD_SIMP_WIN" if credited else None,
                                     credited, "rc4a"))
            eid += 1

    os.makedirs(os.path.dirname(_p(args.out_jsonl)), exist_ok=True)
    with open(_p(args.out_jsonl), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    n = len(rows)
    pos_s = sum(r["label_success"] for r in rows)
    pos_c = sum(r["label_credit"] for r in rows)
    by_source = Counter(r["source"] for r in rows)
    by_outcome = Counter(r["outcome"] for r in rows)
    by_attr = Counter(r["attribution"] for r in rows)
    pos_by_fam = Counter(r["program_family"] for r in rows if r["label_success"])
    summary = {
        "generated_by": "scripts/tr4_build_program_dataset.py",
        "num_examples": n, "num_positive_success": pos_s, "num_positive_credit": pos_c,
        "positive_rate_success": round(pos_s / max(1, n), 5),
        "by_source": dict(by_source), "outcome_histogram": dict(by_outcome),
        "attribution_histogram": dict(by_attr),
        "success_by_family": dict(pos_by_fam),
        "num_theorems": len({r["full_name"] for r in rows}),
        "num_namespaces": len({r["namespace"] for r in rows}),
    }
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR4 program dataset", "",
          f"- examples: **{n}** | success positives: **{pos_s}** | credit positives: **{pos_c}**",
          f"- positive rate (success): {summary['positive_rate_success']}",
          f"- by source: {dict(by_source)}",
          f"- theorems: {summary['num_theorems']} | namespaces: {summary['num_namespaces']}", "",
          f"- success by family: {dict(pos_by_fam)}", "",
          f"- outcome histogram: {dict(by_outcome)}", "",
          f"- attribution histogram: {dict(by_attr)}"]
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")
    print(f"[tr4-dataset] {n} examples, success={pos_s}, credit={pos_c}, by_source={dict(by_source)}")
    print(f"  success_by_family={dict(pos_by_fam)}")


if __name__ == "__main__":
    main()
