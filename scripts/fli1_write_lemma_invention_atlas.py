#!/usr/bin/env python3
"""FLI1 Part 10 — researcher-facing lemma-invention atlas (md + json).

Joins every FLI1 stage (residual capture → clusters → candidates → existing-check → typecheck →
proof → downstream rescue) into one narrative with hedged language and concrete best examples.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _rows(path):
    return [json.loads(l) for l in open(_p(path)) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residual-goals", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--checked", required=True)
    ap.add_argument("--typechecked", required=True)
    ap.add_argument("--proofs", required=True)
    ap.add_argument("--rescues", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    resid = _rows(args.residual_goals)
    clusters = json.load(open(_p(args.clusters)))["clusters"]
    proofs = {r["candidate_id"]: r for r in _rows(args.proofs)}
    rescues = {r["candidate_id"]: r for r in _rows(args.rescues)}
    checked = {r["candidate_id"]: r for r in _rows(args.checked)}

    cap = [r for r in resid if r["status"] == "captured"]
    rescue_recs = list(rescues.values())
    true_rescues = [r for r in rescue_recs if r["classification"] == "DOWNSTREAM_RESCUE"]
    partials = [r for r in rescue_recs if r["classification"] == "PARTIAL_PROGRESS"]
    proved = [r for r in proofs.values() if r.get("prove") == "PROVED"]
    retrieval_gaps = [r for r in checked.values() if r.get("retrieval_gap")]

    # best examples: rescues first, then partials, then proved
    best = []
    seen_best = set()
    for r in (true_rescues + partials + proved):
        cid = r["candidate_id"]
        if cid in seen_best:
            continue
        seen_best.add(cid)
        pr = proofs.get(cid, {})
        ck = checked.get(cid, {})
        rs = rescues.get(cid, {})
        sid = r.get("seed_id") or (r.get("source_seed_ids") or [None])[0]
        seed_resid = next((x for x in resid if x["seed_id"] == sid), {})
        theorem = r.get("theorem") or (r.get("downstream_targets") or [None])[0]
        r = rs if rs else r  # prefer the rescue record's fields for class/tactic
        best.append({
            "candidate_id": cid, "theorem": theorem, "namespace": r.get("namespace"),
            "residual_goal": (seed_resid.get("residual_goals") or [None])[0],
            "candidate_lemma_lean": pr.get("lemma_statement_lean"),
            "existing_check": ck.get("existing_check"),
            "closest_existing_lemma": ck.get("closest_existing_lemma"),
            "typecheck": pr.get("typecheck"), "proved": pr.get("prove"),
            "proof_tactic": pr.get("proof_tactic"),
            "rescue_class": r.get("classification", "PROVED_NO_RESCUE_RECORD"),
            "rescue_tactic": r.get("rescue_tactic"),
            "robust": r.get("robust"),
            "explanation": (f"At `{theorem}`'s position the restricted search failed; "
                            f"{'existing lemma `'+ck.get('closest_existing_lemma','?')+'`' if ck.get('retrieval_gap') else 'the synthesized intermediate lemma'} "
                            f"{'closes' if r['classification']=='DOWNSTREAM_RESCUE' else 'simplifies'} "
                            f"the goal via `{r.get('rescue_tactic')}`."),
        })

    atlas = {
        "generated_by": "scripts/fli1_write_lemma_invention_atlas.py",
        "residual_capture": {"captured": len(cap), "high_quality":
                             sum(1 for r in cap if r.get("capture_quality") == "high")},
        "clusters": len(clusters),
        "candidates": len(checked),
        "existing": dict(Counter(r.get("existing_check") for r in checked.values())),
        "retrieval_gaps": len(retrieval_gaps),
        "typecheck": dict(Counter(r.get("typecheck") for r in proofs.values())),
        "proved": len(proved),
        "rescue": dict(Counter(r["classification"] for r in rescue_recs)),
        "downstream_rescues": len(true_rescues),
        "best_examples": best[:12],
    }
    with open(_p(args.out_json), "w") as f:
        json.dump(atlas, f, ensure_ascii=False, indent=2)

    md = ["# FLI1 Lemma-Invention Atlas", "",
          "_From captured residual goals to candidate intermediate lemmas, proofs, and downstream "
          "rescue. Hedged language: candidates *suggest* missing lemmas; rescues are the real test._",
          "",
          "## 1. Overview",
          "FLI1 re-ran the 40 FLI0 seed failures live, captured residual goals, synthesized "
          "candidate intermediate lemmas, checked existence, typechecked, proved, and tested "
          "downstream rescue at each theorem's file position.", "",
          "## 2. Residual goal capture",
          f"- captured {atlas['residual_capture']['captured']}/40 "
          f"(high-quality {atlas['residual_capture']['high_quality']}); 0 solved directly.", "",
          "## 3. Goal clusters",
          f"- {atlas['clusters']} clusters by (namespace, pattern, relation, container-op).", "",
          "## 4. Candidate lemma families",
          f"- {atlas['candidates']} candidates synthesized (one per captured seed).", "",
          "## 5. Existing lemma / retrieval gaps",
          f"- existing-check: {atlas['existing']}",
          f"- **retrieval gaps (close lemma was retrieved but search didn't use it): "
          f"{atlas['retrieval_gaps']}**", "",
          "## 6. Typechecking results",
          f"- {atlas['typecheck']}", "",
          "## 7. Candidate lemma proof results",
          f"- PROVED: {atlas['proved']}", "",
          "## 8. Downstream rescue attempts",
          f"- {atlas['rescue']}",
          f"- **DOWNSTREAM_RESCUE: {atlas['downstream_rescues']}**", "",
          "## 9. Best examples", ""]
    for b in best[:12]:
        md += [f"### {b['candidate_id']} — `{b['theorem']}` ({b['rescue_class']})",
               f"- residual goal: `{(b['residual_goal'] or '')[:200].replace(chr(10),' / ')}`",
               f"- existing check: {b['existing_check']} (closest `{b['closest_existing_lemma']}`)",
               f"- typecheck: {b['typecheck']} | proved: {b['proved']} via `{b['proof_tactic']}`",
               f"- rescue: **{b['rescue_class']}** via `{b['rescue_tactic']}` (robust={b['robust']})",
               f"- {b['explanation']}", ""]
    md += ["## 10. Failure modes", "",
           "- 18/40 candidate statements fail to typecheck (universe/typeclass/type errors from "
           "reconstructing binders out of pretty-printed goal context).",
           "- Many residual goals fold the hypotheses into binders, so the standalone goal can look "
           "trivial; the content lives in the binders.",
           "- Rescue at-position is the honest bar: a candidate that proves under full `import "
           "Module` may still not rescue at the theorem's position.", "",
           "## 11. Recommendations for FLI2", "",
           "1. Fix candidate-statement synthesis (carry exact binder types from the goal's local "
           "context rather than pp reconstruction; preserve distinct universes/instances).",
           "2. Prioritize the retrieval-gap cluster — these bridges *already exist*; the win is a "
           "routing/deployment fix (gated `simp [L]`), exactly the RC4B/RC4C pattern.",
           "3. For PARTIAL_PROGRESS cases, capture the new residual after the candidate and iterate "
           "(multi-step invention).",
           "4. Promote nothing into production; keep FLI as an off-line discovery track."]
    with open(_p(args.out_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli1-atlas] captured={len(cap)} candidates={len(checked)} proved={len(proved)} "
          f"rescues={len(true_rescues)} partials={len(partials)} retrieval_gaps={len(retrieval_gaps)}")


if __name__ == "__main__":
    main()
