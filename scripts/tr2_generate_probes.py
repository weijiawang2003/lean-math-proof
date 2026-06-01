#!/usr/bin/env python3
"""TR2 Part 5 — generate a probe plan from the router's predictions.

For every confirmed RC2 failure (and, for control purposes, RC2-solved cases) the
top router prediction selects a *probe family* per the TR2 mapping. The plan is
conservative: MISSING_BRIDGE is routed to retrieval (exact?) + an SF5 flag rather
than blind tactic spam; DEPTH_GAP gets a bounded depth-2/3 battery; NO_CHEAP_ACTION
and BASELINE_DUPLICATE get the minimal control set; SET_ITE is a sanity negative
(RC2 already owns it -> expect PRODUCTION_SUBSUMED). Pure planning; no Lean.

Probe records mirror the SF4 gate schema so the live runner can reuse SF4's verified
control/probe outcomes and run only the uncovered increment.
"""
from __future__ import annotations

import argparse
import json
import os

CONTROLS = ["simp", "simp_all", "aesop", "classical <;> aesop"]

# bounded depth-2/3 generic battery for PROOF_SEARCH_DEPTH_GAP (no unbounded search)
DEPTH_BATTERY = ["aesop", "simp_all", "simp_all <;> aesop", "constructor <;> aesop",
                 "intro h <;> aesop", "ext x <;> aesop", "ext x <;> simp_all",
                 "simp_all <;> omega", "rintro ⟨_, _⟩ <;> aesop"]

# prediction -> (probe family, [tactics], retrieval?, expected_class, max_budget)
def _plan_for(label, ns):
    if label == "MISSING_BRIDGE_LEMMA_CANDIDATE":
        return ("retrieval", ["exact?"], True, "MISSING_BRIDGE_LEMMA_CANDIDATE", 8)
    if label == "PROOF_SEARCH_DEPTH_GAP":
        return ("depth_gap_bounded", DEPTH_BATTERY, False, "PROOF_SEARCH_DEPTH_GAP_or_TRUE_DELTA", 15)
    if label == "NO_CHEAP_ACTION":
        return ("minimal_controls", ["simp", "simp_all", "aesop"], False, "NO_CHEAP_ACTION_confirmed", 8)
    if label == "BASELINE_DUPLICATE":
        return ("controls", CONTROLS, False, "BASELINE_DUPLICATE", 8)
    if label == "SET_ITE_SIMP":
        return ("set_ite_sanity", ["simp [Set.ite]"], False, "PRODUCTION_SUBSUMED", 4)
    if label == "WX3_MULTISET_INDUCTION":
        return ("multiset_induction", ["induction s using Multiset.induction_on <;> simp_all"],
                False, "WX3_or_TRUE_DELTA", 8)
    if label == "MX2_TOFINSET_AESOP":
        return ("tofinset_aesop", ["aesop", "simp [Set.Finite.toFinset] <;> aesop"],
                False, "MX2_or_TRUE_DELTA", 8)
    return ("controls", CONTROLS, False, "review", 8)


def _gate(label, ns, fn):
    """Narrow gate per family (namespace + name-feature), off-gate discipline."""
    if label == "WX3_MULTISET_INDUCTION":
        return {"namespaces": ["Multiset"], "name_features": [], "max_emissions_per_theorem": 1}
    if label == "MX2_TOFINSET_AESOP":
        return {"namespaces": ["Multiset", "Finset", "Set"], "name_features": ["toFinset", "Finite"],
                "max_emissions_per_theorem": 1}
    if label == "SET_ITE_SIMP":
        return {"namespaces": ["Set"], "name_features": ["ite", "dite", ".if"], "max_emissions_per_theorem": 1}
    # retrieval / controls / depth battery are generic (no narrow gate)
    return {"namespaces": [], "name_features": [], "max_emissions_per_theorem": 1}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--confirmation", required=True)
    p.add_argument("--pool", default="project/evolve/experiments/tr2/cases/tr2_candidate_pool.jsonl",
                   help="candidate pool — supplies the router's predicted label per theorem")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--include-solved-controls", action="store_true", default=True,
                   help="plan control probes on RC2-solved cases to verify BASELINE_DUPLICATE with a clear control proof")
    args = p.parse_args(argv)

    conf = json.load(open(args.confirmation))
    results = conf.get("results", [])
    pred_of = {}
    if os.path.exists(args.pool):
        for l in open(args.pool):
            if l.strip():
                r = json.loads(l)
                pred_of[r["full_name"]] = r.get("tr1_predicted_label")
    for r in results:  # enrich confirmation rows with the router prediction
        r.setdefault("tr1_predicted_label", pred_of.get(r["full_name"]))

    theorems = []
    for r in results:
        fn = r["full_name"]
        cls = r["classification"]
        if cls not in ("CONFIRMED_RC2_FAILURE", "RC2_SOLVED"):
            continue
        ns = r.get("namespace") or (fn.split(".")[0] if "." in fn else "")
        # choose plan from top prediction; solved cases default to controls
        # (we must look up the router prediction from the pool, carried on confirmation? -> derive from cluster/name)
        pred = r.get("tr1_predicted_label")
        if cls == "RC2_SOLVED":
            family, tactics, retrieval, expected, budget = ("controls", CONTROLS, False, "BASELINE_DUPLICATE", 8)
            pred = pred or "BASELINE_DUPLICATE"
        else:
            pred = pred or "NO_CHEAP_ACTION"
            family, tactics, retrieval, expected, budget = _plan_for(pred, ns)
        gate = _gate(pred, ns, fn)
        probes = []
        for t in tactics[:budget]:
            probes.append({"family": family, "tactic_or_sequence": t, "is_sequence": "<;>" in t,
                           "retrieval": retrieval and t == "exact?", "gate": gate,
                           "risk": "low" if t in CONTROLS or t == "exact?" else "medium",
                           "from_prediction": pred, "source": "router_prediction",
                           "source_specific": False, "promotion_allowed": False})
        # always attach controls for SX4 attribution context (reused from SF4 where available)
        ctrl_set = {pp["tactic_or_sequence"] for pp in probes}
        controls = [c for c in CONTROLS if c not in ctrl_set]
        theorems.append({
            "full_name": fn, "file_path": r.get("file_path"), "namespace": ns,
            "rc2_classification": cls, "predicted_label": pred,
            "probe_family": family, "expected_outcome": expected,
            "for_sf5_retrieval": bool(retrieval),
            "probe_budget": budget, "num_probes": len(probes),
            "controls_for_attribution": controls,
            "probes": probes,
        })

    # router prediction needs to be on the confirmation; if absent we re-attach from pool
    import collections
    fam_hist = collections.Counter(t["probe_family"] for t in theorems)
    sf5 = [t["full_name"] for t in theorems if t["for_sf5_retrieval"]]
    out = {"confirmation_input": args.confirmation, "num_theorems": len(theorems),
           "probe_family_histogram": dict(fam_hist),
           "for_sf5_retrieval": sf5, "num_for_sf5_retrieval": len(sf5),
           "mapping": {
               "MISSING_BRIDGE_LEMMA_CANDIDATE": "retrieval (exact?) + SF5 flag — no blind spam",
               "PROOF_SEARCH_DEPTH_GAP": "bounded depth-2/3 battery (<=15)",
               "NO_CHEAP_ACTION": "minimal controls; negative if all fail",
               "BASELINE_DUPLICATE": "controls (simp/simp_all/aesop/classical;aesop)",
               "SET_ITE_SIMP": "sanity negative — RC2 owns it -> PRODUCTION_SUBSUMED",
               "WX3_MULTISET_INDUCTION": "Multiset.induction_on <;> simp_all (gated)",
               "MX2_TOFINSET_AESOP": "narrow toFinset/Finite aesop (gated)",
           },
           "budget_policy": "max 8 probes/theorem; 15 for depth-gap; controls reused from SF4 where present",
           "theorems": theorems}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    L = ["# TR2 probe plan", "",
         f"- theorems planned: **{len(theorems)}**",
         f"- probe-family histogram: {dict(fam_hist)}",
         f"- routed to SF5 retrieval (no tactic spam): **{len(sf5)}**", "",
         "| theorem | rc2 | predicted | family | #probes | sf5? |", "|---|---|---|---|---|---|"]
    for t in theorems:
        L.append(f"| `{t['full_name']}` | {t['rc2_classification']} | {t['predicted_label']} | "
                 f"{t['probe_family']} | {t['num_probes']} | {'Y' if t['for_sf5_retrieval'] else '—'} |")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[tr2-probes] theorems={len(theorems)} families={dict(fam_hist)} sf5={len(sf5)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
