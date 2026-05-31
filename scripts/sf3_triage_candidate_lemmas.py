#!/usr/bin/env python3
"""SF3 Part 6 — candidate lemma triage for the Set deep-dive.

CONSERVATIVE by design. A failure becomes a candidate "missing lemma" only when:
  * its cluster rolled up to `missing_lemma` or `mixed`, AND
  * no probe (including source-inspired) closed it, AND
  * the official Mathlib proof is a rewrite-bridge over NAMED lemmas (i.e. the
    capability gap is a specific bridge, not generic automation).
If the official proof shows existing lemmas already discharge the goal (it is just
a tactic/search/routing gap), we DO NOT propose a lemma — we say so.

The Multiset singleton case is the cautionary precedent: a failing goal whose
official proof is count-extensionality over EXISTING lemmas is a tactic/search gap,
not a missing lemma.

Outputs:
  set_candidate_lemmas.json
  set_candidate_lemma_queue.jsonl
  set_candidate_lemma_triage.md
"""
from __future__ import annotations

import argparse
import json
import os
import re

CA = "project/evolve/experiments/sf2/out/set_cluster_deep_dive/cluster_analysis.json"
SRC = "project/evolve/experiments/sf2/out/set_cluster_deep_dive/source_context.json"
PROBES = "project/evolve/experiments/sf2/out/set_cluster_deep_dive/probe_results.json"
OUT_JSON = "project/evolve/experiments/sf3/out/set_candidate_lemmas.json"
OUT_QUEUE = "project/evolve/experiments/sf3/out/set_candidate_lemma_queue.jsonl"
OUT_MD = "project/evolve/experiments/sf3/out/set_candidate_lemma_triage.md"

# Lemmas referenced in an official proof signal the bridge ALREADY EXISTS in
# Mathlib -> not a missing lemma, just a routing/search gap.
RW_LEMMA_RE = re.compile(r"\b(?:rw|simp(?:\s+only)?|exact|apply)\b\s*\[?([A-Za-z][\w'.]*)")


def existing_lemmas_in_proof(proof):
    names = set()
    for m in re.finditer(r"[A-Za-z][\w']*(?:\.[A-Za-z][\w']*)+", proof or ""):
        names.add(m.group(0))
    # also bare snake_case lemma-ish tokens used in rw/exact
    for m in re.finditer(r"\b([a-z][a-z0-9_]{3,})\b", proof or ""):
        names.add(m.group(1))
    return sorted(names)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cluster-analysis", default=CA)
    p.add_argument("--source-context", default=SRC)
    p.add_argument("--probe-results", default=PROBES)
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-queue", default=OUT_QUEUE)
    p.add_argument("--out-md", default=OUT_MD)
    args = p.parse_args(argv)

    ca = json.load(open(args.cluster_analysis))
    src = {r["full_name"]: r for r in json.load(open(args.source_context))["cases"]}
    probe_by = {r["full_name"]: r for r in json.load(open(args.probe_results))["results"]}

    candidates, rejected = [], []
    for c in ca["clusters"]:
        gap = c["likely_gap_type"]
        for t in c["selected_theorems"]:
            pr = probe_by.get(t, {})
            sc = src.get(t, {})
            cls = (sc.get("classification") or {})
            proof = sc.get("official_proof", "") or ""
            solved = pr.get("solved_by_probe")
            base = {
                "candidate_id": f"setlemma::{t}",
                "cluster_id": c["cluster_id"],
                "source_failures": [t],
                "official_proof_style": cls.get("proof_style"),
                "probe_solved": solved,
                "probe_gap": pr.get("classification"),
            }
            # rejection logic
            if solved:
                rejected.append({**base, "verdict": "not_missing_lemma",
                                 "reason": f"probe solved it ({pr.get('classification')}): "
                                           f"`{pr.get('winning_probe')}`"})
                continue
            if gap not in ("missing_lemma", "mixed"):
                rejected.append({**base, "verdict": "not_missing_lemma",
                                 "reason": f"cluster gap is {gap}, not a missing-lemma cluster"})
                continue
            # official proof is rw-bridge over named existing lemmas -> the bridge
            # exists; this is a search/routing gap, not a new lemma.
            named = existing_lemmas_in_proof(proof)
            if cls.get("proof_style") in ("rw_bridge",) and named:
                rejected.append({**base, "verdict": "not_missing_lemma",
                                 "reason": "official proof is an rw-bridge over EXISTING "
                                           "named lemmas (search-depth gap, not missing): "
                                           + ", ".join(sorted(named)[:8]),
                                 "nearby_existing_lemmas": sorted(named)[:12]})
                continue
            # otherwise: genuinely unproven by probes and not an obvious existing bridge
            candidates.append({
                **base,
                "lemma_template_name": f"{t.split('.')[-1]}_bridge",
                "lean_statement_sketch": (sc.get("statement") or "").strip()[:240],
                "why_not_just_tactic": "no probe (incl. source-inspired) closed the goal; "
                                       "official proof not a simple existing-lemma rw-bridge",
                "nearby_existing_lemmas": [n["decl"] for n in sc.get("nearby_lemmas", [])][:10],
                "novelty_risk": "unknown",
                "expected_downstream_utility": "single",
                "first_proof_attempt": cls.get("likely_reusable_probe"),
                "priority": "medium",
            })

    out = {"num_candidates": len(candidates), "num_rejected": len(rejected),
           "policy": "conservative; a failure is a missing-lemma candidate only if "
                     "unproven by all probes AND official proof is not an existing-lemma "
                     "rw-bridge. Mirrors the Multiset-singleton negative result.",
           "candidates": candidates, "rejected": rejected}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), ensure_ascii=False, indent=2)
    with open(args.out_queue, "w") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    L = ["# SF3 Set Candidate-Lemma Triage", ""]
    L.append(f"- candidates: {len(candidates)} | rejected as not-missing-lemma: {len(rejected)}")
    L.append(f"- policy: {out['policy']}")
    L.append("")
    L.append("## Candidates (honest missing-lemma candidates)")
    if not candidates:
        L.append("- **None.** Every Set failure is explained by a tactic / routing / "
                 "search-depth gap with existing lemmas — no new bridge lemma is warranted.")
    for c in candidates:
        L.append(f"### `{c['candidate_id']}`")
        L.append(f"- cluster: {c['cluster_id']}")
        L.append(f"- sketch: `{c['lean_statement_sketch']}`")
        L.append(f"- why not just a tactic: {c['why_not_just_tactic']}")
        L.append(f"- novelty risk: {c['novelty_risk']} | utility: {c['expected_downstream_utility']}")
        L.append("")
    L.append("## Rejected (NOT missing lemmas)")
    L.append("")
    L.append("| theorem | verdict | reason |")
    L.append("|---|---|---|")
    for r in rejected:
        L.append(f"| `{r['source_failures'][0]}` | {r['verdict']} | {r['reason'][:90]} |")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[sf3:triage] candidates={len(candidates)} rejected={len(rejected)} "
          f"-> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
