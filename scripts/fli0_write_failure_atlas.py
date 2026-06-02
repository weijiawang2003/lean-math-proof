#!/usr/bin/env python3
"""FLI0 Part 7 — human-readable failure atlas (md) + machine atlas (json).

Synthesizes the enriched failures, pattern classification, and seed selection into a researcher-
facing atlas. Per major pattern: an intuitive explanation, an example theorem, why current search
failed, and the kind of intermediate lemma that might help. Uses hedged language throughout.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATTERN_INTUITION = {
    "MAP_FILTER_BIND_BRIDGE": "The goal talks about membership in (or equality of) an image / "
        "map / filter / bind / biUnion of a container. `simp`/`aesop` stall because they lack a "
        "membership-unfolding lemma for that specific transformer, so the elementwise structure "
        "never gets exposed.",
    "SUBSET_BRIDGE": "The goal is (or reduces to) a subset relation. Search fails when it cannot "
        "turn `s ⊆ t` into the pointwise `∀ x, x ∈ s → x ∈ t` form (or a pair/singleton subset "
        "characterization) that downstream automation can chew on.",
    "MEMBERSHIP_BRIDGE": "The goal is an iff whose left side is a membership statement about a "
        "derived structure. The missing piece appears to be a `x ∈ … ↔ <condition>` rewrite that "
        "bridges the membership to an elementwise predicate.",
    "IFF_SPLIT": "The goal is a biconditional. Single-shot tactics fail to make progress because "
        "the two directions need different reasoning; a `constructor` split plus a one-directional "
        "bridge lemma on each side is the likely shape.",
    "DISJOINT_BRIDGE": "The goal involves `Disjoint`. Automation stalls without the membership "
        "translation `Disjoint a b ↔ ∀ x, x ∈ a → x ∉ b` for this container — exactly the family "
        "the RC4B bridge captured for Set/Multiset; other containers may need their own.",
    "SINGLETON_CHARACTERIZATION": "The goal characterizes a singleton (or a `card ≤ 1` / "
        "subsingleton condition). The candidate is a `… = {x} ↔ …` (or `y ∈ {x} ↔ y = x`) lemma "
        "for the container.",
    "INDUCTION_GENERALIZATION": "A List/Multiset goal over a recursive constructor "
        "(cons/append/foldr…) where plain simp/aesop cannot fold through the recursion; a "
        "generalized induction helper with a stronger hypothesis seems needed.",
    "EXTENSIONALITY_NEEDED": "An equality of containers built by image/filter/union. The likely "
        "route is extensionality — reduce to elementwise membership (`ext x; simp [...]`) — which "
        "the current batteries do not attempt.",
    "ORDER_STRUCTURE_GAP": "Order/lattice goals where the tactic families have no order-specific "
        "structural route. Lower priority for lemma invention (often not a single missing lemma).",
    "NAT_ARITH_GAP": "Nat/Int arithmetic beyond the current omega/nlinarith reach. Usually not a "
        "reusable bridge lemma.",
    "UNKNOWN_NAME_OR_IMPORT": "Failure dominated by unavailable lemma names — an availability gap, "
        "not a genuine invention target.",
    "SIMP_LOOP_OR_RECURSION": "simp/aesop hit recursion or heartbeat limits — a simp-set / "
        "termination issue, not a missing math lemma.",
    "LOW_SIGNAL": "No obvious reusable missing lemma from statement + retrieval alone.",
    "NEEDS_REVIEW": "Statement or trace too thin to classify (often infra/setup error).",
}


def _p(*a):
    return os.path.join(_REPO, *a)


def _example(cases, pattern):
    cand = [c for c in cases if c["primary_pattern"] == pattern and c.get("clean_failure")
            and c.get("statement")]
    cand.sort(key=lambda c: (c.get("confidence") != "high", len(c["statement"]), c["theorem"]))
    return cand[0] if cand else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", required=True)
    ap.add_argument("--patterns", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    enriched = [json.loads(l) for l in open(_p(args.enriched)) if l.strip()]
    patterns = [json.loads(l) for l in open(_p(args.patterns)) if l.strip()]
    seeds = json.load(open(_p(args.seeds)))["seeds"]

    clean = [c for c in patterns if c.get("clean_failure")]
    primary_clean = Counter(c["primary_pattern"] for c in clean)
    ns_clean = Counter(c["namespace"] for c in clean)
    major = [p for p, _ in primary_clean.most_common() if p not in
             ("LOW_SIGNAL", "NEEDS_REVIEW")][:8]

    atlas = {
        "generated_by": "scripts/fli0_write_failure_atlas.py",
        "corpus": {"total_failures": len(patterns), "clean_failures": len(clean),
                   "seeds": len(seeds)},
        "by_namespace_clean": dict(ns_clean.most_common()),
        "by_primary_pattern_clean": dict(primary_clean.most_common()),
        "patterns": [], "seeds": seeds,
    }
    for pat in major:
        ex = _example(patterns, pat)
        atlas["patterns"].append({
            "pattern": pat, "clean_count": primary_clean[pat],
            "intuition": PATTERN_INTUITION.get(pat, ""),
            "example_theorem": ex["theorem"] if ex else None,
            "example_statement": ex["statement"] if ex else None,
            "candidate_lemma_shape_nl": ex["candidate_lemma_shape_nl"] if ex else None,
            "why_search_failed": (f"dynamic search ({ex['source_stage']}) result was "
                                  f"'{[c for c in enriched if c['theorem']==ex['theorem']][0]['dynamic_result']}'; "
                                  f"retrieval surfaced lemmas but none closed it") if ex else None,
        })
    with open(_p(args.out_json), "w") as f:
        json.dump(atlas, f, ensure_ascii=False, indent=2)

    md = ["# FLI0 Failure Atlas", "",
          "_A researcher-facing map of where the RC5 hybrid search fails and what intermediate "
          "lemmas might help. Language is deliberately hedged — these are candidate shapes, not "
          "verified requirements._", "",
          "## 1. Overview", "",
          "FLI0 mines the RC5V2 (complete) and RC5V3 (partial-raw) hybrid-search artifacts for "
          "theorems the full stack (RC2 → RC4 static → safe dynamic) could not prove, and groups "
          "them by the *kind* of gap each exhibits.", "",
          "## 2. Source stages used", "",
          "- **RC5V2** — complete, committed attribution (149 eligible, 8 solved).",
          "- **RC5V3** — `PARTIAL_ARTIFACTS_AVAILABLE` (raw B1/B3/B5 results only; analysis layer "
          "never produced). A B5 network outage produced ~112 infra-only records, separated out.",
          "- RC5V2 ∩ RC5V3 = 0 (disjoint fresh frontiers).", "",
          "## 3. Failure corpus size", "",
          f"- total failures extracted: **{len(patterns)}**",
          f"- clean failures (math, not infra/timeout/unknown-name): **{len(clean)}**",
          f"- seed cases selected for FLI1: **{len(seeds)}**", "",
          "## 4. Namespace distribution (clean failures)", "",
          "| namespace | clean failures |", "|---|---|"]
    for ns, n in ns_clean.most_common():
        md.append(f"| {ns} | {n} |")
    md += ["", "## 5. Main failure patterns (clean failures)", "",
           "| pattern | clean count |", "|---|---|"]
    for pat, n in primary_clean.most_common():
        md.append(f"| {pat} | {n} |")
    md += ["", "## 6. High-value seed cases", "",
           f"{len(seeds)} seeds selected (clean + fresh + readable + invention-friendly pattern). "
           "Full records in `cases/fli0_seed_cases.json`.", "",
           "| id | theorem | ns | pattern | conf |", "|---|---|---|---|---|"]
    for s in seeds[:20]:
        md.append(f"| {s['seed_id']} | `{s['theorem']}` | {s['namespace']} | "
                  f"{s['primary_pattern']} | {s['confidence']} |")
    md += [f"| … | _({len(seeds)-20} more)_ | | | |"] if len(seeds) > 20 else []
    md += ["", "## 7. Examples of residual goals", "",
           "> **Residual goal states are unavailable in all artifacts** (the dynamic logs record "
           "tactic *outcomes*, not post-tactic goals). FLI0 reasons from the theorem statement, "
           "feature vector, retrieved lemmas, and which tactic families failed. Capturing residual "
           "goals for the chosen seeds (via a short live re-run) is an explicit FLI1 step.", "",
           "Representative seed statements:"]
    for s in seeds[:5]:
        md.append(f"- `{s['theorem']}` — `{(s['statement'] or '')[:150]}`")
    md += ["", "## 8. Candidate missing-lemma shapes", ""]
    for entry in atlas["patterns"]:
        md += [f"### {entry['pattern']}  ({entry['clean_count']} clean)", "",
               entry["intuition"], ""]
        if entry["example_theorem"]:
            md += [f"- **example:** `{entry['example_theorem']}`",
                   f"  - statement: `{(entry['example_statement'] or '')[:170]}`",
                   f"  - why search failed: {entry['why_search_failed']}",
                   f"  - candidate lemma shape: {entry['candidate_lemma_shape_nl']}", ""]
    md += ["## 9. What FLI1 should try next", "",
           "1. Re-run the 40 seeds live to **capture residual goals** (the one missing ingredient).",
           "2. For the bridge patterns (membership / subset / disjoint / map-filter-bind), "
           "**synthesize the candidate `↔` lemma**, prove it (or retrieve it), add it as a gated "
           "`simp [L]` enabling action, and check whether the downstream theorem now closes — the "
           "RC4B/RC4C deployment pattern.",
           "3. Start with the highest-confidence, most-clustered families (Finset/List membership "
           "& subset bridges) where one invented lemma may rescue several theorems.",
           "4. Defer ORDER_STRUCTURE_GAP / NAT_ARITH_GAP (rarely a single missing lemma).", "",
           "## 10. Caveats", "",
           "- No residual goals → pattern labels are inferred from statement + retrieval, not from "
           "a stuck goal state. Conservative by design.",
           "- RC5V3 is partial; its B5 network outage means some V3 theorems only have B1 live data.",
           "- Several Finset seeds are `card_*` near-variants (one invented lemma may cover the "
           "cluster — or none generalize); treat the cluster as a single bet, not many.",
           "- Labels are multi-signal heuristics; a label says a failure *suggests* a lemma "
           "family, never that it *requires* one."]
    with open(_p(args.out_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli0-atlas] total={len(patterns)} clean={len(clean)} seeds={len(seeds)} "
          f"major_patterns={major}")


if __name__ == "__main__":
    main()
