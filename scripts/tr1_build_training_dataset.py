#!/usr/bin/env python3
"""TR1 Part 2 — build the failure-to-action training dataset from VERIFIED artifacts.

Aggregates per-theorem records across RC2/SX/SF artifacts, assigns ONE verified
label per theorem by precedence, and emits tr1_examples.jsonl + a summary.

Verified-label discipline (see tr1_methodology.md): positive labels only from
actual production deltas / minimal-relabel-confirmed wins / accepted RC components.
SX3 proxy "wins" enter ONLY as the negative class SX3_PRODUCTION_SUBSUMED.
"""
from __future__ import annotations

import argparse
import json
import os
import re

ART = "project/evolve/experiments"

# ---- label precedence (lower index = more specific / wins ties) ----
LABEL_ORDER = [
    "SET_ITE_SIMP", "WX3_MULTISET_INDUCTION", "MX2_TOFINSET_AESOP",
    "SX3_PRODUCTION_SUBSUMED", "PROOF_SEARCH_DEPTH_GAP", "MISSING_BRIDGE_LEMMA_CANDIDATE",
    "BASELINE_DUPLICATE", "SOURCE_SPECIFIC_OR_REJECTED", "NO_CHEAP_ACTION",
]
LABEL_TYPE = {
    "SET_ITE_SIMP": "positive", "WX3_MULTISET_INDUCTION": "positive", "MX2_TOFINSET_AESOP": "positive",
    "SX3_PRODUCTION_SUBSUMED": "negative", "BASELINE_DUPLICATE": "negative",
    "SOURCE_SPECIFIC_OR_REJECTED": "negative",
    "NO_CHEAP_ACTION": "triage", "MISSING_BRIDGE_LEMMA_CANDIDATE": "triage",
    "PROOF_SEARCH_DEPTH_GAP": "triage",
}


def _load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def _ns(fn):
    return fn.split(".")[0] if "." in fn else ""


def _tokens(fn):
    base = fn.split(".")[-1]
    parts = re.split(r"[._]", fn)
    camel = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", base)
    toks = [t.lower() for t in (parts + camel) if t]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out


def _features(fn, goal, err, symptoms):
    low = fn.lower()
    g = (goal or "")
    return {
        "has_set": fn.startswith("Set.") or "set " in g.lower() or "Set " in g,
        "has_ite": "ite" in low or "dite" in low or "if " in g,
        "has_iff": "_iff" in low or "iff_" in low or "↔" in g,
        "has_subset": "subset" in low or "ssubset" in low or "⊆" in g or "⊂" in g,
        "has_multiset": "multiset" in low or fn.startswith("Multiset."),
        "has_tofinset": "tofinset" in low or "finite" in low,
        "has_card": "card" in low,
        "has_singleton": "singleton" in low or "pair" in low,
        "has_extensionality_shape": "ext" in low or "=" in g and fn.startswith("Set."),
        "has_induction_signal": "induction" in low or "rec" in low or "Multiset" in fn,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--out-summary-json", required=True)
    p.add_argument("--out-summary-md", required=True)
    args = p.parse_args(argv)

    # per-theorem accumulator
    rec = {}  # full_name -> dict

    def ensure(fn, **kw):
        r = rec.setdefault(fn, {"full_name": fn, "namespace": _ns(fn), "file_path": None,
                                "goal_text": None, "last_error": None, "trace_symptoms": [],
                                "rc2_status": "unknown", "source_artifacts": [],
                                "source_surface": None, "label_candidates": [],
                                "sx4_credit": None, "minimal_relabel_class": None,
                                "candidate_family": None})
        for k, v in kw.items():
            if k == "source_artifacts":
                for s in v:
                    if s not in r["source_artifacts"]:
                        r["source_artifacts"].append(s)
            elif k == "trace_symptoms":
                for s in v:
                    if s not in r["trace_symptoms"]:
                        r["trace_symptoms"].append(s)
            elif k == "label_candidates":
                r["label_candidates"].extend(v)
            elif v is not None and (r.get(k) in (None, "", "unknown") or k in ("sx4_credit", "minimal_relabel_class", "candidate_family")):
                r[k] = v
        return r

    def addlabel(fn, label, conf, **kw):
        ensure(fn, label_candidates=[{"label": label, "confidence": conf}], **kw)

    # ---- goal-text map from SF2 + SX3 results ----
    goalmap = {}
    sf2pr = _load(f"{ART}/sf2/out/set_cluster_deep_dive/probe_results.json")
    if sf2pr:
        for r in sf2pr.get("results", []):
            if r.get("full_name") and r.get("initial_goal"):
                goalmap[r["full_name"]] = r["initial_goal"]
    for f in ["sx3_set_cluster_results.json", "sx3_fresh_holdout_results.json", "sx3_deferred_results.json"]:
        d = _load(f"{ART}/sx3/out/{f}")
        if d:
            for r in d.get("results", []):
                if r.get("full_name") and r.get("initial_goal") and r["full_name"] not in goalmap:
                    goalmap[r["full_name"]] = r["initial_goal"]

    # ---- 1. SET_ITE_SIMP positives (rc2_delta_ledger credited) ----
    ledger = _load(f"{ART}/rc2_hardening/out/rc2_delta_ledger.json")
    if ledger:
        for row in ledger.get("ledger", []):
            if row.get("category") == "credited_SET_ITE_SIMP":
                addlabel(row["full_name"], "SET_ITE_SIMP", "verified",
                         source_artifacts=["rc2_delta_ledger"], rc2_status="solved",
                         minimal_relabel_class=row.get("minimal_relabel"),
                         candidate_family="SET_ITE_SIMP", sx4_credit=True)
        for fn in ledger.get("deferred_sx3", []):
            addlabel(fn, "SX3_PRODUCTION_SUBSUMED", "verified",
                     source_artifacts=["rc2_delta_ledger_deferred"], rc2_status="solved",
                     candidate_family="SX3_SET_ITE_AESOP", sx4_credit=False)

    # ---- 2. SX2 SET2 minimal relabel (confirm SET_ITE_SIMP + baseline dups + failures) ----
    sx2 = _load(f"{ART}/sx2/out/set2_minimal_relabel_results.json")
    if sx2:
        for row in sx2.get("rows", []):
            fn = row.get("full_name")
            if not fn:
                continue
            a = row.get("attribution")
            if a == "TRUE_SET2_WIN":
                addlabel(fn, "SET_ITE_SIMP", "verified", source_artifacts=["sx2_set2_relabel"],
                         minimal_relabel_class=a, candidate_family="SET_ITE_SIMP", sx4_credit=True)
            elif a == "BASELINE_DUPLICATE":
                addlabel(fn, "BASELINE_DUPLICATE", "verified", source_artifacts=["sx2_set2_relabel"],
                         minimal_relabel_class=a)
            elif a == "NEEDS_DEEPER_SEQUENCE":
                addlabel(fn, "NO_CHEAP_ACTION", "strong", source_artifacts=["sx2_set2_relabel"],
                         minimal_relabel_class=a)

    # ---- 3. SX4 reattribution (SX3 over-credit -> subsumed; baseline; single-shot) ----
    sx4 = _load(f"{ART}/sx4/out/sx3_set_ite_aesop_reattribution.json")
    if sx4:
        for r in sx4.get("records", []):
            fn = r["theorem"]
            cls = r.get("classification")
            if r.get("proxy_runner_credited"):
                addlabel(fn, "SX3_PRODUCTION_SUBSUMED", "verified", source_artifacts=["sx4_reattribution"],
                         candidate_family="SX3_SET_ITE_AESOP", sx4_credit=False,
                         minimal_relabel_class=cls)
            elif cls == "PRODUCTION_SUBSUMED":
                # RC2 solves via baseline path -> BASELINE_DUPLICATE unless it's a single-shot set.ite
                path = r.get("production_trace_analysis", {}).get("baseline_winning_path") or []
                if path == ["simp [Set.ite]"]:
                    addlabel(fn, "SET_ITE_SIMP", "verified", source_artifacts=["sx4_reattribution"],
                             rc2_status="solved", candidate_family="SET_ITE_SIMP", sx4_credit=True)
                else:
                    addlabel(fn, "BASELINE_DUPLICATE", "strong", source_artifacts=["sx4_reattribution"],
                             rc2_status="solved")

    # ---- 4. SF4 confirmation (rc2_status + NOW_SOLVED winning tactics) ----
    conf = _load(f"{ART}/sf4/out/rc2_failure_confirmation.json")
    if conf:
        for r in conf.get("results", []):
            fn = r["full_name"]
            cls = r.get("classification")
            if cls == "NOW_SOLVED_BY_RC2":
                wt = (r.get("winning_tactic") or "")
                if wt == "simp [Set.ite]":
                    addlabel(fn, "SET_ITE_SIMP", "verified", source_artifacts=["sf4_confirmation"],
                             file_path=r.get("file_path"), rc2_status="solved",
                             candidate_family="SET_ITE_SIMP", sx4_credit=True)
                elif "induction" in wt and "Multiset" in wt:
                    addlabel(fn, "WX3_MULTISET_INDUCTION", "verified", source_artifacts=["sf4_confirmation"],
                             file_path=r.get("file_path"), rc2_status="solved",
                             candidate_family="WX3_MULTISET_INDUCTION", sx4_credit=True)
                else:
                    addlabel(fn, "BASELINE_DUPLICATE", "verified", source_artifacts=["sf4_confirmation"],
                             file_path=r.get("file_path"), rc2_status="solved")
            elif cls == "CONFIRMED_RC2_FAILURE":
                ensure(fn, file_path=r.get("file_path"), rc2_status="failed",
                       source_artifacts=["sf4_confirmation"])

    # ---- 5. SF4 clusters (cluster id, goal shape, symptoms) ----
    clusters = _load(f"{ART}/sf4/out/rc2_failure_clusters.json")
    cluster_of = {}
    if clusters:
        for c in clusters.get("clusters", []):
            for fn in c.get("members", []):
                cluster_of[fn] = c
                ensure(fn, source_surface=c["cluster_id"], trace_symptoms=c.get("symptoms", []))

    # ---- 6. SF4 probe results (controls/trace symptoms) ----
    pr = _load(f"{ART}/sf4/out/sf4_probe_results.json")
    if pr:
        for r in pr.get("results", []):
            fn = r["full_name"]
            syms = []
            for c in r.get("controls", []) + r.get("probes_tried", []):
                oc = c.get("outcome")
                if oc in ("parse_error", "max_recursion", "timeout"):
                    syms.append(oc)
            ensure(fn, trace_symptoms=syms, source_artifacts=["sf4_probe_results"])

    # ---- 7. SF4 SX4 attribution (FAILED_CANDIDATE / BASELINE_DUPLICATE) ----
    sf4sx4 = _load(f"{ART}/sf4/out/sf4_sx4_attribution.json")
    if sf4sx4:
        for r in sf4sx4.get("records", []):
            fn = r["full_name"]
            cls = r.get("classification")
            if cls == "BASELINE_DUPLICATE":
                addlabel(fn, "PROOF_SEARCH_DEPTH_GAP", "verified", source_artifacts=["sf4_sx4_attribution"],
                         candidate_family="baseline_aesop")
            elif cls == "FAILED_CANDIDATE":
                addlabel(fn, "NO_CHEAP_ACTION", "verified", source_artifacts=["sf4_sx4_attribution"])

    # ---- 8. SF4 missing-lemma triage (bridge / depth-gap) ----
    triage = _load(f"{ART}/sf4/out/sf4_missing_lemma_candidates.json")
    if triage:
        for t in triage.get("triage", []):
            cat = t.get("category")
            for fn in t.get("members", []):
                if cat == "POSSIBLE_MISSING_BRIDGE_LEMMA":
                    addlabel(fn, "MISSING_BRIDGE_LEMMA_CANDIDATE", "strong",
                             source_artifacts=["sf4_missing_lemma_triage"])
                elif cat == "PROOF_SEARCH_DEPTH_GAP":
                    addlabel(fn, "PROOF_SEARCH_DEPTH_GAP", "verified",
                             source_artifacts=["sf4_missing_lemma_triage"])

    # ---- 9. SF2 deep-dive (source-specific / rejected only-solves) ----
    sf2ca = _load(f"{ART}/sf2/out/set_cluster_deep_dive/cluster_analysis.json")
    if sf2ca and isinstance(sf2ca, dict):
        for fn in sf2ca.get("source_specific_only", []) or []:
            addlabel(fn, "SOURCE_SPECIFIC_OR_REJECTED", "strong", source_artifacts=["sf2_cluster_analysis"])

    # ---- resolve final label by precedence; build examples ----
    examples = []
    for fn, r in rec.items():
        cands = r["label_candidates"]
        if not cands:
            # untracked failure with no label -> NO_CHEAP_ACTION if confirmed failure else skip
            if r["rc2_status"] == "failed":
                cands = [{"label": "NO_CHEAP_ACTION", "confidence": "strong"}]
            else:
                continue
        # pick by precedence, keeping strongest confidence seen for that label
        def rank(c):
            return LABEL_ORDER.index(c["label"]) if c["label"] in LABEL_ORDER else 999
        best = min(cands, key=rank)
        conf_rank = {"verified": 0, "strong": 1, "weak": 2}
        same = [c for c in cands if c["label"] == best["label"]]
        conf = min((c["confidence"] for c in same), key=lambda x: conf_rank.get(x, 9))
        goal = r.get("goal_text") or goalmap.get(fn)
        feats = _features(fn, goal, r.get("last_error"), r.get("trace_symptoms"))
        examples.append({
            "example_id": f"tr1_{len(examples):03d}",
            "full_name": fn, "file_path": r.get("file_path"), "namespace": r["namespace"],
            "theorem_name_tokens": _tokens(fn),
            "goal_text": goal, "last_error": r.get("last_error"),
            "trace_symptoms": r.get("trace_symptoms", []),
            "source_artifact": ",".join(r["source_artifacts"]),
            "source_surface": r.get("source_surface"),
            "rc2_status": r.get("rc2_status", "unknown"),
            "candidate_family": r.get("candidate_family"),
            "label": best["label"], "label_type": LABEL_TYPE.get(best["label"], "negative"),
            "label_confidence": conf,
            "sx4_credit": r.get("sx4_credit"),
            "minimal_relabel_class": r.get("minimal_relabel_class"),
            "features": feats,
        })
    examples.sort(key=lambda e: (e["label"], e["full_name"]))

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")

    label_dist, type_dist, conf_dist = {}, {}, {}
    for e in examples:
        label_dist[e["label"]] = label_dist.get(e["label"], 0) + 1
        type_dist[e["label_type"]] = type_dist.get(e["label_type"], 0) + 1
        conf_dist[e["label_confidence"]] = conf_dist.get(e["label_confidence"], 0) + 1
    # ensure all labels appear in map even if 0
    for lab in LABEL_ORDER:
        label_dist.setdefault(lab, 0)
    low_support = sorted([lab for lab, n in label_dist.items() if 0 < n < 3])
    zero_support = sorted([lab for lab, n in label_dist.items() if n == 0])

    summary = {
        "num_examples": len(examples),
        "label_distribution": dict(sorted(label_dist.items(), key=lambda kv: -kv[1])),
        "label_type_distribution": type_dist,
        "confidence_distribution": conf_dist,
        "low_support_labels": low_support,
        "zero_support_labels": zero_support,
        "num_with_goal_text": sum(1 for e in examples if e["goal_text"]),
        "namespaces": sorted({e["namespace"] for e in examples}),
        "source_artifacts_used": sorted({s for e in examples for s in e["source_artifact"].split(",") if s}),
        "verified_label_policy": "positives only from production deltas / minimal-relabel-confirmed / accepted RC components; SX3 proxy wins enter only as SX3_PRODUCTION_SUBSUMED (negative).",
    }
    json.dump(summary, open(args.out_summary_json, "w"), indent=2)

    L = ["# TR1 training dataset summary", "",
         f"- examples: **{len(examples)}** (with goal text: {summary['num_with_goal_text']})",
         f"- label types: {type_dist}",
         f"- confidence: {conf_dist}",
         f"- low-support labels (<3): {low_support}",
         f"- zero-support labels: {zero_support}", "",
         "## Label distribution", "", "| label | type | count |", "|---|---|---|"]
    for lab, n in sorted(label_dist.items(), key=lambda kv: -kv[1]):
        L.append(f"| `{lab}` | {LABEL_TYPE.get(lab,'?')} | {n} |")
    L += ["", f"Sources: {summary['source_artifacts_used']}",
          "", f"> {summary['verified_label_policy']}"]
    open(args.out_summary_md, "w").write("\n".join(L))
    print(f"[tr1-data] examples={len(examples)} labels={dict(sorted(label_dist.items(),key=lambda kv:-kv[1]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
