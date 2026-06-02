#!/usr/bin/env python3
"""FLI0 Part 5 — classify each enriched failure into conservative pattern labels.

Multi-label, rule-based over the feature vector + statement tokens + retrieved-lemma names +
failure outcomes. Each case gets pattern labels (with per-label evidence), a confidence, an NL
explanation, a candidate-lemma-shape (NL), and a recommended FLI1 probe family. We deliberately
say "suggests / appears to need", never "requires". Statement-missing or infra/unknown-name cases
are routed to UNKNOWN_NAME_OR_IMPORT / NEEDS_REVIEW rather than a math pattern.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER_NS = {"Monotone", "MonotoneOn", "Antitone", "AntitoneOn", "IsLUB", "IsGLB",
            "IsLeast", "IsGreatest", "OrderIso", "Order"}
NAT_NS = {"Nat", "Int", ""}
STRUCT_NS = {"List", "Multiset", "Finset", "Set"}
# priority for choosing the PRIMARY label (most actionable for lemma invention first)
PRIORITY = ["DISJOINT_BRIDGE", "SINGLETON_CHARACTERIZATION", "MEMBERSHIP_BRIDGE",
            "SUBSET_BRIDGE", "MAP_FILTER_BIND_BRIDGE", "IFF_SPLIT", "EXTENSIONALITY_NEEDED",
            "INDUCTION_GENERALIZATION", "SIMP_LOOP_OR_RECURSION", "ORDER_STRUCTURE_GAP",
            "NAT_ARITH_GAP", "UNKNOWN_NAME_OR_IMPORT", "LOW_SIGNAL", "NEEDS_REVIEW"]

SHAPES = {
    "DISJOINT_BRIDGE": "a membership translation `Disjoint a b ↔ ∀ x, x ∈ a → x ∉ b` "
                       "specialized to this container, letting `simp`+`aesop` discharge it.",
    "SINGLETON_CHARACTERIZATION": "a `… = {x} ↔ …` (or `y ∈ {x} ↔ y = x`) characterization lemma "
                                  "for this container.",
    "MEMBERSHIP_BRIDGE": "a `x ∈ <transformed container> ↔ <elementwise condition>` rewrite lemma.",
    "SUBSET_BRIDGE": "a `s ⊆ t ↔ ∀ x, x ∈ s → x ∈ t` / pair-subset characterization for this type.",
    "MAP_FILTER_BIND_BRIDGE": "a membership lemma for the map/filter/bind/image, e.g. "
                              "`x ∈ f <$> s ↔ ∃ y ∈ s, f y = x`.",
    "IFF_SPLIT": "no new lemma per se — a `constructor`/two-direction split helper or the missing "
                 "one-directional bridge lemma each side needs.",
    "EXTENSIONALITY_NEEDED": "an extensionality bridge: reduce container equality to elementwise "
                             "membership (`ext x; simp [...]`).",
    "INDUCTION_GENERALIZATION": "a generalized induction helper (stronger IH over the recursive "
                                "structure) that the elementwise goal can fold through.",
    "SIMP_LOOP_OR_RECURSION": "not a missing lemma — a simp-set / termination fix (the search hit "
                              "recursion/heartbeat limits).",
    "ORDER_STRUCTURE_GAP": "an order/lattice structural lemma; current tactic batteries have no "
                           "order-specific route.",
    "NAT_ARITH_GAP": "arithmetic beyond the current search (omega/nlinarith gap), not an obvious "
                     "reusable bridge.",
    "UNKNOWN_NAME_OR_IMPORT": "availability gap — the needed lemma exists but was not imported / "
                              "named correctly, not a true invention target.",
    "LOW_SIGNAL": "no obvious reusable missing lemma from statement + retrieval alone.",
    "NEEDS_REVIEW": "ambiguous — statement or trace too thin to classify.",
}
PROBE = {
    "DISJOINT_BRIDGE": "generate_disjoint_membership_bridge",
    "SINGLETON_CHARACTERIZATION": "generate_singleton_iff",
    "MEMBERSHIP_BRIDGE": "generate_membership_bridge",
    "SUBSET_BRIDGE": "generate_subset_bridge",
    "MAP_FILTER_BIND_BRIDGE": "generate_map_filter_bind_membership",
    "IFF_SPLIT": "generate_iff_split_helper",
    "EXTENSIONALITY_NEEDED": "generate_ext_membership_bridge",
    "INDUCTION_GENERALIZATION": "generate_induction_helper",
    "SIMP_LOOP_OR_RECURSION": "tune_simp_set_or_termination",
    "ORDER_STRUCTURE_GAP": "design_order_structural_battery",
    "NAT_ARITH_GAP": "extend_arith_search",
    "UNKNOWN_NAME_OR_IMPORT": "fix_import_or_lemma_name",
    "LOW_SIGNAL": "defer",
    "NEEDS_REVIEW": "manual_review",
}


def _p(*a):
    return os.path.join(_REPO, *a)


def _lemma_blob(c):
    return " ".join((tl.get("lemma") or "") for tl in c.get("top_retrieved_lemmas_detailed", []))


# a genuine singleton literal on a side of `=` or `∈`: `{a}` with no binder colon, set-builder
# pipe, or comma inside the braces (those are implicit binders / set-builders / pairs, not
# singletons). Retrieved-lemma text and the noisy `has_singleton` feature are NOT used to trigger
# the singleton label — they over-fire on `{f : α → β}` binders.
_SINGLETON_LIT = re.compile(r"[=∈]\s*\{[^,|:}]+\}")


def classify(c):
    stmt = c.get("statement") or ""
    feats = set(c.get("feature_bucket", []))
    ns = c.get("namespace") or ""
    root_ns = ns.split(".")[0]
    name = c["theorem"]
    lname = name.lower()
    lemblob = _lemma_blob(c).lower()
    outs = c.get("num_failure_outcomes", {}) or {}
    labels = {}

    def add(label, ev):
        labels.setdefault(label, []).append(ev)

    # infra / availability first
    if c.get("dynamic_result") in ("infra_error",) or not stmt:
        add("NEEDS_REVIEW", "infra/setup error or missing statement")
    if c.get("dynamic_result") == "unknown_name" or (outs.get("unknown_name", 0)
                                                      and outs.get("unknown_name", 0) >= sum(outs.values())):
        add("UNKNOWN_NAME_OR_IMPORT", "failures dominated by unknown_name")
    if outs.get("max_recursion") or "recursion" in (c.get("last_error") or "").lower() \
            or "heartbeat" in (c.get("last_error") or "").lower():
        add("SIMP_LOOP_OR_RECURSION", "max_recursion / heartbeat in trace")

    # math patterns (only when we have a statement)
    if stmt:
        if "has_disjoint" in feats or "disjoint" in lname or "Disjoint" in stmt:
            add("DISJOINT_BRIDGE", "disjoint feature/name/stmt")
        # singleton: ONLY a literal `{a}` on a side of =/∈ or "singleton" in the theorem name.
        if "singleton" in lname or _SINGLETON_LIT.search(stmt):
            add("SINGLETON_CHARACTERIZATION",
                "singleton in name" if "singleton" in lname else "singleton literal in goal")
        if ("has_mem" in feats or "∈" in stmt) and ("↔" in stmt or "has_iff" in feats):
            add("MEMBERSHIP_BRIDGE", "membership + iff")
        if "has_subset" in feats or "⊆" in stmt or "subset" in lname:
            add("SUBSET_BRIDGE", "subset feature/name/stmt")
        if "has_map_filter" in feats or "has_tofinset" in feats or \
                any(t in c.get("involved_constants", []) for t in
                    ("image", "map", "filter", "bind", "biUnion", "iUnion", "preimage", "toFinset")):
            add("MAP_FILTER_BIND_BRIDGE", "map/filter/bind/image constant")
        if "has_iff" in feats or "↔" in stmt:
            add("IFF_SPLIT", "iff goal")
        if "has_eq" in feats and root_ns in STRUCT_NS and \
                any(t in c.get("involved_constants", []) for t in
                    ("image", "filter", "map", "biUnion", "iUnion", "sUnion", "preimage")):
            add("EXTENSIONALITY_NEEDED", "container equality over transformed structure")
        if root_ns in {"List", "Multiset"} and ("has_eq" in feats or "has_iff" in feats) and \
                any(t in lname for t in ("cons", "append", "foldr", "foldl", "join", "concat", "rec")):
            add("INDUCTION_GENERALIZATION", "recursive List/Multiset structure")
        if "has_order" in feats or root_ns in ORDER_NS or ns in ORDER_NS:
            add("ORDER_STRUCTURE_GAP", "order feature/namespace")
        if "has_nat_arith" in feats or root_ns in {"Nat", "Int"}:
            add("NAT_ARITH_GAP", "nat/int arithmetic")

    if not labels:
        add("LOW_SIGNAL", "no pattern signal from statement + retrieval")

    primary = next((p for p in PRIORITY if p in labels), "LOW_SIGNAL")
    nsig = len(labels)
    math_labels = [l for l in labels if l not in
                   ("UNKNOWN_NAME_OR_IMPORT", "SIMP_LOOP_OR_RECURSION", "LOW_SIGNAL", "NEEDS_REVIEW")]
    if primary in ("LOW_SIGNAL", "NEEDS_REVIEW", "UNKNOWN_NAME_OR_IMPORT"):
        conf = "low"
    elif len(labels[primary]) >= 1 and nsig >= 3 and primary in ("DISJOINT_BRIDGE",
                                                                  "SINGLETON_CHARACTERIZATION",
                                                                  "MEMBERSHIP_BRIDGE", "SUBSET_BRIDGE"):
        conf = "high"
    else:
        conf = "medium"
    expl = (f"Goal in `{ns}` shows signals {sorted(labels)}; primary pattern **{primary}**. "
            f"Retrieval surfaced lemmas ({', '.join(l['lemma'] for l in c.get('top_retrieved_lemmas_detailed', [])[:3] if l.get('lemma'))}) "
            f"but the dynamic search ({c.get('dynamic_result')}) did not close it.")
    return {
        "theorem": name, "namespace": ns, "source_stage": c.get("source_stage"),
        "freshness_status": c.get("freshness_status"),
        "pattern_labels": sorted(labels), "primary_pattern": primary,
        "label_evidence": {k: v for k, v in labels.items()},
        "has_math_pattern": bool(math_labels),
        "confidence": conf, "explanation": expl,
        "candidate_lemma_shape_nl": SHAPES[primary],
        "recommended_next_probe": PROBE[primary],
        "clean_failure": c.get("clean_failure"),
        "statement": stmt, "top_retrieved_lemmas": c.get("top_retrieved_lemmas", []),
        "similar_solved_theorem": c.get("similar_solved_theorem"),
        "failed_tactic_trace": c.get("failed_tactic_trace", []),
        "last_error": c.get("last_error"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(_p(args.enriched)) if l.strip()]
    out = [classify(c) for c in cases]
    out.sort(key=lambda r: (r["source_stage"], r["namespace"], r["theorem"]))
    with open(_p(args.out_jsonl), "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    clean = [r for r in out if r["clean_failure"]]
    primary_hist = Counter(r["primary_pattern"] for r in out)
    primary_clean = Counter(r["primary_pattern"] for r in clean)
    label_hist = Counter(l for r in out for l in r["pattern_labels"])
    high_signal_clean = [r for r in clean if r["has_math_pattern"]
                         and r["confidence"] in ("high", "medium")]
    summary = {
        "generated_by": "scripts/fli0_classify_failure_patterns.py",
        "num_cases": len(out), "num_clean": len(clean),
        "primary_pattern_histogram": dict(primary_hist.most_common()),
        "primary_pattern_histogram_clean": dict(primary_clean.most_common()),
        "label_histogram": dict(label_hist.most_common()),
        "confidence_histogram": dict(Counter(r["confidence"] for r in out)),
        "high_signal_clean_cases": len(high_signal_clean),
        "high_signal_patterns": dict(Counter(r["primary_pattern"] for r in high_signal_clean).most_common()),
    }
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI0 failure pattern summary", "",
          f"- cases: {summary['num_cases']} | clean: {summary['num_clean']} | "
          f"**high-signal clean: {summary['high_signal_clean_cases']}**",
          f"- confidence: {summary['confidence_histogram']}", "",
          "## Primary pattern (clean failures)", "", "| pattern | clean | all |", "|---|---|---|"]
    for pat in summary["primary_pattern_histogram"]:
        md.append(f"| {pat} | {primary_clean.get(pat, 0)} | {primary_hist[pat]} |")
    md += ["", "## All pattern labels (multi-label, all cases)", "", "| label | count |", "|---|---|"]
    for k, v in summary["label_histogram"].items():
        md.append(f"| {k} | {v} |")
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli0-classify] cases={len(out)} clean={len(clean)} "
          f"high_signal_clean={len(high_signal_clean)} primary_clean={dict(primary_clean.most_common(6))}")


if __name__ == "__main__":
    main()
