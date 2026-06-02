#!/usr/bin/env python3
"""FLI1 Part 2 — build a controlled, deterministic live-rerun plan for the 40 FLI0 seeds.

Each seed gets a small set of probe sequences (each probe = a short tactic chain applied from the
initial state) designed to make ONE step of progress and stop, so Part 3 can capture a meaningful
residual goal. Pattern-specific openers; banned: simp_all / depth-3 chains / broad aesop loops /
B20 exhaustion. file_path + import module are joined from the FLI0 enriched cases.
"""
from __future__ import annotations

import argparse
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENRICHED = "project/evolve/experiments/fli0/cases/fli0_failed_cases_enriched.jsonl"

HIGH = {"MEMBERSHIP_BRIDGE", "SUBSET_BRIDGE", "DISJOINT_BRIDGE", "MAP_FILTER_BIND_BRIDGE",
        "IFF_SPLIT"}
MED = {"SINGLETON_CHARACTERIZATION", "EXTENSIONALITY_NEEDED"}
LOW = {"INDUCTION_GENERALIZATION"}


def _p(*a):
    return os.path.join(_REPO, *a)


def _module(file_path):
    if not file_path:
        return None
    fp = file_path[:-5] if file_path.endswith(".lean") else file_path
    return fp.replace("/", ".")


def _probes(seed):
    """Return list of probe sequences (each a list of tactic strings). Capture runs each from the
    initial state and keeps the best non-finishing progressing one. Kept small & safe."""
    pat = seed["primary_pattern"]
    retr = [l for l in (seed.get("top_retrieved_lemmas") or []) if l][:2]
    probes = [["simp"]]                       # does plain simp finish / progress?
    # original failed dynamic tactic, if any (one, deduped, no simp_all)
    for ft in (seed.get("failed_tactics") or []):
        if ft and "simp_all" not in ft and ft not in ("simp",):
            probes.append([ft])
            break
    if pat in ("IFF_SPLIT", "MEMBERSHIP_BRIDGE", "SINGLETON_CHARACTERIZATION"):
        probes.append(["constructor"])
        probes.append(["constructor", "intro h"])
    if pat == "SUBSET_BRIDGE":
        probes.append(["intro h"])            # works when goal is `… → …`
        probes.append(["intro x hx"])
    if pat == "MAP_FILTER_BIND_BRIDGE":
        for L in retr:
            probes.append([f"simp [{L}]"])
        probes.append(["simp only [Finset.mem_biUnion]"] if seed["namespace"].startswith("Finset")
                      else ["simp only [List.mem_map]"])
    if pat == "DISJOINT_BRIDGE":
        ns = seed["namespace"].split(".")[0]
        probes.append([f"simp [{ns}.disjoint_left]"])
        probes.append(["rw [Finset.disjoint_left]"] if ns == "Finset" else ["intro h"])
    if pat == "EXTENSIONALITY_NEEDED":
        probes.append(["ext x"])
        probes.append(["ext x", "simp"])
    if pat == "INDUCTION_GENERALIZATION":
        probes.append(["induction h"])        # shallow, one only; capture bounds it
    # dedup while preserving order
    seen, out = set(), []
    for pr in probes:
        key = tuple(pr)
        if key not in seen:
            seen.add(key)
            out.append(pr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    seeds = json.load(open(_p(args.seeds)))["seeds"]
    enr = {json.loads(l)["theorem"]: json.loads(l)
           for l in open(_p(ENRICHED)) if l.strip()}

    plan = []
    for s in seeds:
        fp = enr.get(s["theorem"], {}).get("file_path")
        pat = s["primary_pattern"]
        prio = "high" if pat in HIGH else ("medium" if pat in MED else "low")
        plan.append({
            "seed_id": s["seed_id"], "theorem": s["theorem"], "namespace": s["namespace"],
            "statement": s["statement"], "source_stage": s["source_stage"],
            "file_path": fp, "import_module": _module(fp),
            "failure_patterns": s["failure_pattern"], "primary_pattern": pat,
            "original_failed_tactics": s.get("failed_tactics", [])[:6],
            "top_retrieved_lemmas": [l for l in (s.get("top_retrieved_lemmas") or []) if l][:5],
            "rerun_probes": _probes(s),
            "capture_strategy": f"{prio}-priority {pat}: plain simp + original failed tactic + "
                                f"pattern openers; keep best non-finishing residual",
            "timeout_seconds": 30, "priority": prio,
        })
    prio_rank = {"high": 0, "medium": 1, "low": 2}
    plan.sort(key=lambda r: (prio_rank[r["priority"]], r["namespace"], r["seed_id"]))

    missing_fp = [r["seed_id"] for r in plan if not r["file_path"]]
    out = {"generated_by": "scripts/fli1_build_live_rerun_plan.py",
           "num_seeds": len(plan), "missing_file_path": missing_fp,
           "by_priority": {k: sum(1 for r in plan if r["priority"] == k)
                           for k in ("high", "medium", "low")},
           "seeds": plan}
    with open(_p(args.out_json), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    md = ["# FLI1 live rerun plan", "",
          f"- seeds: {len(plan)} | by priority: {out['by_priority']}",
          f"- seeds missing file_path: {len(missing_fp)} {missing_fp}", "",
          "| seed | theorem | ns | pattern | prio | #probes |", "|---|---|---|---|---|---|"]
    for r in plan:
        md.append(f"| {r['seed_id']} | `{r['theorem']}` | {r['namespace']} | "
                  f"{r['primary_pattern']} | {r['priority']} | {len(r['rerun_probes'])} |")
    with open(_p(args.out_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli1-plan] seeds={len(plan)} by_priority={out['by_priority']} "
          f"missing_fp={len(missing_fp)}")


if __name__ == "__main__":
    main()
