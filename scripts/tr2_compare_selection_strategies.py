#!/usr/bin/env python3
"""TR2 Part 8 — compare model-selected vs rule-selected vs random-baseline selection.

The headline metric is ACTIVE_LEARNING_GAIN = useful verified labels per LeanDojo
probe spent, broken down by selection strategy, plus true-delta yield, label
diversity, and the under-represented / non-Set coverage each strategy achieved.

Because the pool is fully pre-labelled and tiny (see methodology), the decision
space includes INCONCLUSIVE_TOO_SMALL; the comparison is reported honestly with
both raw and probe-normalised figures and a guard against over-reading noise.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

STRATS = ("model", "rule", "random")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--batch-manifest", required=True)
    p.add_argument("--outcomes", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--pool", default="project/evolve/experiments/tr2/cases/tr2_candidate_pool.jsonl")
    args = p.parse_args(argv)

    manifest = json.load(open(args.batch_manifest))
    att = json.load(open(args.outcomes))
    rec_by = {r["full_name"]: r for r in att["records"]}
    pool = {r["full_name"]: r for r in (json.loads(l) for l in open(args.pool) if l.strip())} \
        if os.path.exists(args.pool) else {}

    per_strat = {}
    for s in STRATS:
        members = manifest["batches"][s]["members"]
        recs = [rec_by[m] for m in members if m in rec_by]
        cls_hist = collections.Counter(r["classification"] for r in recs)
        # live probes spent = sum of num_live across this batch's members (reused tactics cost 0 fresh budget)
        live_probes = sum((r.get("num_live") or 0) for r in recs)
        total_probes = sum((r.get("num_live") or 0) + (r.get("num_reused") or 0) for r in recs)
        useful = [r for r in recs if r["useful_label"]]
        true_delta = [r for r in recs if r["classification"] == "TRUE_DELTA"]
        missing = [r for r in recs if r["classification"] == "MISSING_BRIDGE_LEMMA_CANDIDATE"]
        depth = [r for r in recs if r["classification"] == "PROOF_SEARCH_DEPTH_GAP"]
        nocheap = [r for r in recs if r["classification"] == "NO_CHEAP_ACTION"]
        basedup = [r for r in recs if r["classification"] == "BASELINE_DUPLICATE"]
        flakes = [r for r in recs if r["classification"] == "OPEN_FLAKE"]
        confirmed = [r for r in recs if r["rc2_finished"] is False]
        ns = collections.Counter(pool.get(m, {}).get("namespace") for m in members)
        clusters = collections.Counter(pool.get(m, {}).get("sf4_cluster") for m in members if pool.get(m, {}).get("sf4_cluster"))
        non_set = sum(1 for m in members if pool.get(m, {}).get("namespace") not in ("Set", None))
        underrep = sum(1 for r in recs if r["classification"] in
                       ("PROOF_SEARCH_DEPTH_GAP",) or (rec_by[r["full_name"]].get("predicted_label") in
                        ("WX3_MULTISET_INDUCTION", "MX2_TOFINSET_AESOP", "PROOF_SEARCH_DEPTH_GAP")))
        per_strat[s] = {
            "selected": len(members), "scored": len(recs),
            "confirmed_rc2_failures": len(confirmed),
            "useful_labels": len(useful),
            "true_delta": len(true_delta),
            "missing_lemma_candidates": len(missing),
            "depth_gap_cases": len(depth),
            "no_cheap_action_confirmations": len(nocheap),
            "baseline_duplicates": len(basedup),
            "open_flakes": len(flakes),
            "live_probes": live_probes, "total_probes": total_probes,
            "useful_per_live_probe": round(len(useful) / live_probes, 4) if live_probes else None,
            "useful_per_total_probe": round(len(useful) / total_probes, 4) if total_probes else None,
            "true_delta_per_live_probe": round(len(true_delta) / live_probes, 4) if live_probes else 0.0,
            "namespace_diversity": len([k for k in ns if k]),
            "namespace_dist": {k: v for k, v in ns.items() if k},
            "cluster_diversity": len(clusters),
            "non_set_cases": non_set,
            "underrepresented_cases": underrep,
            "classification_histogram": dict(cls_hist),
        }

    # ---- decision logic ----
    m, r, rb = per_strat["model"], per_strat["rule"], per_strat["random"]
    total_true_delta = sum(per_strat[s]["true_delta"] for s in STRATS)
    # useful-per-live-probe ranking (None -> treat as 0)
    upp = {s: (per_strat[s]["useful_per_live_probe"] or 0.0) for s in STRATS}
    model_best_upp = upp["model"] >= max(upp["rule"], upp["random"])
    # diversity / coverage advantage
    model_div_adv = (m["namespace_diversity"] >= r["namespace_diversity"] and
                     m["namespace_diversity"] >= rb["namespace_diversity"] and
                     m["non_set_cases"] >= max(r["non_set_cases"], rb["non_set_cases"]))
    model_useful_adv = m["useful_labels"] >= max(r["useful_labels"], rb["useful_labels"])

    pool_exhausted = all(pool.get(mn, {}).get("in_tr1_training") for mn in
                         set().union(*[manifest["batches"][s]["members"] for s in STRATS])) if pool else False

    # tiny + fully pre-labelled + no fresh true deltas anywhere -> cannot separate strategies on yield
    if total_true_delta == 0 and pool_exhausted:
        if model_div_adv and m["useful_labels"] >= r["useful_labels"]:
            decision = "INCONCLUSIVE_TOO_SMALL"
            rationale = ("No strategy can yield a fresh TRUE_DELTA on an exhausted, fully pre-labelled pool; "
                         "model selection shows a diversity/coverage edge (more namespaces & non-Set cases) "
                         "but the sample is too small to call it a win on useful-labels-per-probe.")
        else:
            decision = "INCONCLUSIVE_TOO_SMALL"
            rationale = "Exhausted pre-labelled pool; strategies indistinguishable on label yield."
    elif total_true_delta == 0 and not model_useful_adv and upp["random"] >= upp["model"]:
        decision = "RANDOM_AS_GOOD"
        rationale = "Random matches model on useful-labels-per-probe and yields no fewer useful labels."
    elif model_best_upp and model_useful_adv and (model_div_adv or m["true_delta"] > 0):
        decision = "MODEL_SELECTION_USEFUL"
        rationale = "Model selection leads on useful-labels-per-probe AND diversity/true-delta yield."
    elif r["useful_per_live_probe"] and (r["useful_per_live_probe"] or 0) >= (upp["model"]) and \
            r["useful_labels"] >= m["useful_labels"]:
        decision = "RULE_BASELINE_SUFFICIENT"
        rationale = "Handcrafted rule baseline matches/exceeds the model with no extra benefit from learning."
    elif upp["model"] < min(upp["rule"], upp["random"]) and m["useful_labels"] <= min(r["useful_labels"], rb["useful_labels"]):
        decision = "MODEL_BAD_SELECTION"
        rationale = "Model selection is worse than both baselines on yield and per-probe usefulness."
    else:
        decision = "INCONCLUSIVE_TOO_SMALL"
        rationale = "Mixed signals on a small pool; no robust separation."

    out = {"batch_manifest": args.batch_manifest, "outcomes_input": args.outcomes,
           "per_strategy": per_strat,
           "totals": {"true_delta_all_strategies": total_true_delta,
                      "pool_exhausted_all_in_tr1": pool_exhausted},
           "diversity_edge_model": model_div_adv,
           "useful_per_live_probe": upp,
           "decision": decision, "rationale": rationale,
           "caveat": ("Pool is ~40 fully pre-labelled theorems; figures are descriptive. Random overlaps "
                      "model/rule on scarce failures by design (matched failure ratio), so per-probe "
                      "comparisons control for difficulty but cannot reach significance at this n.")}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    cols = ["selected", "confirmed_rc2_failures", "useful_labels", "true_delta",
            "missing_lemma_candidates", "depth_gap_cases", "no_cheap_action_confirmations",
            "baseline_duplicates", "open_flakes", "live_probes", "useful_per_live_probe",
            "namespace_diversity", "non_set_cases", "underrepresented_cases"]
    L = ["# TR2 selection-strategy comparison", "",
         f"## Decision: `{decision}`", "", f"{rationale}", "",
         f"> {out['caveat']}", "",
         "| metric | model | rule | random |", "|---|---|---|---|"]
    for c in cols:
        L.append(f"| {c} | {m.get(c)} | {r.get(c)} | {rb.get(c)} |")
    L += ["", "## Per-strategy classification histograms", ""]
    for s in STRATS:
        L.append(f"- **{s}**: {per_strat[s]['classification_histogram']}")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[tr2-compare] decision={decision} true_delta_all={total_true_delta} "
          f"useful(model/rule/random)={m['useful_labels']}/{r['useful_labels']}/{rb['useful_labels']} "
          f"upp={upp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
