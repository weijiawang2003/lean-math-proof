#!/usr/bin/env python3
"""SX3 Part 9 — sequence-family generalization analysis.

Aggregates per-family evidence across all SX3 result files + the minimal
attribution, and computes a heuristic generalization score:

  +2 per fresh true win, +1 per deferred true win,
  -3 per off-gate emission, -2 per baseline duplicate, -2 per source-specific,
  -1 per parse-fragility (family produced only parse/recursion failures on a case),
  +2 if the family has multi-cluster support (>=2 distinct case surfaces with a true win).

The score is a heuristic; raw evidence is reported alongside.
"""
from __future__ import annotations
import argparse
import json

DEFERRED4 = {"Set.ite_inter", "Set.ite_inter_self", "Set.ite_compl",
             "Set.ite_inter_compl_self"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--attribution", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--registry", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args()

    registry = json.load(open(args.registry))
    attr = json.load(open(args.attribution))
    fam_keys = list(registry["families"].keys())

    # gather per-family raw counts directly from result files (authoritative)
    fam = {k: {"cases_tried": 0, "gate_emissions": 0, "true_wins": set(),
               "fresh_wins": set(), "deferred_wins": set(), "duplicates": 0,
               "off_gate": 0, "parse_errors": 0, "timeouts": 0,
               "surfaces": set(), "win_surfaces": set()} for k in fam_keys}

    for path in args.results:
        d = json.load(open(path))
        surface = path.split("/")[-1].replace("sx3_", "").replace("_results.json", "")
        for r in d.get("results", []):
            ns = r.get("namespace", "")
            for g in r.get("gate_decisions", []):
                k = g["family"]
                if k not in fam:
                    continue
                fam[k]["cases_tried"] += 1
                if g["emitted"]:
                    fam[k]["gate_emissions"] += 1
                    if ns != "Set":
                        fam[k]["off_gate"] += 1
            for s in r.get("gated_sequences_tried", []):
                k = s.get("family")
                if k not in fam:
                    continue
                fam[k]["surfaces"].add(surface)
                if s.get("outcome") == "parse_error":
                    fam[k]["parse_errors"] += 1
                if s.get("outcome") in ("timeout_inner",):
                    fam[k]["timeouts"] += 1

    # true wins from attribution (per_theorem with winning_family)
    for rec in attr.get("per_theorem", []):
        if rec["attribution"] == "TRUE_DEPTH2_SEQUENCE_WIN" and rec.get("winning_family"):
            k = rec["winning_family"]
            n = rec["full_name"]
            fam[k]["true_wins"].add(n)
            fam[k]["win_surfaces"].add(rec.get("surface", ""))
            if n in DEFERRED4:
                fam[k]["deferred_wins"].add(n)
            else:
                fam[k]["fresh_wins"].add(n)
        if rec["attribution"] in ("SINGLE_STEP_DUPLICATE", "BASELINE_DUPLICATE") \
                and rec.get("winning_family"):
            fam[rec["winning_family"]]["duplicates"] += 1

    out_families = []
    for k in fam_keys:
        v = fam[k]
        n_fresh = len(v["fresh_wins"])
        n_def = len(v["deferred_wins"])
        multi = len(v["win_surfaces"]) >= 2
        score = (2 * n_fresh + 1 * n_def - 3 * v["off_gate"]
                 - 2 * v["duplicates"] + (2 if multi else 0)
                 - (1 if v["parse_errors"] > 0 and len(v["true_wins"]) == 0 else 0))
        # promotion recommendation
        if v["off_gate"] > 0:
            rec = "REJECT_OFF_GATE"
        elif n_fresh >= 1 and n_def >= 1:
            rec = "RC3_CANDIDATE" if k == "SX3_SET_ITE_AESOP" else "KEEP_EXPERIMENTAL"
        elif n_def >= 1 and n_fresh == 0:
            rec = "TRAINING_DATA_ONLY"
        elif len(v["true_wins"]) == 0:
            rec = "REJECT_NO_DELTA"
        else:
            rec = "KEEP_EXPERIMENTAL"
        out_families.append({
            "family": k,
            "sequence": registry["families"][k]["sequence"],
            "cases_tried": v["cases_tried"],
            "gate_emissions": v["gate_emissions"],
            "true_depth2_wins": sorted(v["true_wins"]),
            "fresh_wins": sorted(v["fresh_wins"]),
            "known_deferred_wins": sorted(v["deferred_wins"]),
            "duplicates": v["duplicates"],
            "off_gate": v["off_gate"],
            "parse_errors": v["parse_errors"],
            "timeouts": v["timeouts"],
            "surfaces_run": sorted(v["surfaces"]),
            "win_surfaces": sorted(s for s in v["win_surfaces"] if s),
            "multi_cluster_support": multi,
            "generalization_score": score,
            "promotion_recommendation": rec,
        })

    out = {"inputs": args.results, "families": out_families,
           "best_family": max(out_families, key=lambda f: f["generalization_score"])["family"],
           "note": "Generalization score is a heuristic; see raw evidence per family. "
                   "Only SX3_SET_ITE_AESOP is an RC3 candidate; other families are exploratory."}
    json.dump(out, open(args.out_json, "w"), ensure_ascii=False, indent=2)

    L = ["# SX3 Sequence-Family Generalization Analysis", ""]
    L.append(f"- best family (by heuristic score): **{out['best_family']}**")
    L.append("")
    L.append("| family | seq | fresh | deferred | dup | off-gate | parse-err | multi | score | recommendation |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for f in sorted(out_families, key=lambda x: -x["generalization_score"]):
        L.append(f"| {f['family']} | `{f['sequence'][:24]}` | {len(f['fresh_wins'])} | "
                 f"{len(f['known_deferred_wins'])} | {f['duplicates']} | {f['off_gate']} | "
                 f"{f['parse_errors']} | {f['multi_cluster_support']} | "
                 f"{f['generalization_score']} | **{f['promotion_recommendation']}** |")
    L.append("")
    for f in out_families:
        if f["true_depth2_wins"]:
            L.append(f"- **{f['family']}** true wins: "
                     f"{', '.join('`'+w+'`' for w in f['true_depth2_wins'])} "
                     f"(fresh={len(f['fresh_wins'])}, deferred={len(f['known_deferred_wins'])})")
    L.append("")
    L.append("> " + out["note"])
    open(args.out_md, "w").write("\n".join(L))
    print(f"[sx3:family] best={out['best_family']}")
    for f in out_families:
        if f["true_depth2_wins"] or f["off_gate"]:
            print(f"  {f['family']}: fresh={len(f['fresh_wins'])} def={len(f['known_deferred_wins'])} "
                  f"offgate={f['off_gate']} score={f['generalization_score']} -> {f['promotion_recommendation']}")


if __name__ == "__main__":
    main()
