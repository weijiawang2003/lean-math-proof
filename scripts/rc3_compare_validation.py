#!/usr/bin/env python3
"""RC3 validation comparison + decision.

Joins the five validation artifacts (literal RC2, literal RC3, minimal relabel,
preservation/off-gate, determinism audit) into a single comparison JSON+MD and
emits one of the canonical decisions.

Decisions:
  RC3_RELEASE_CANDIDATE_CONFIRMED
  RC3_CANDIDATE_CONFIRMED_WITH_ENV_FLAKE
  KEEP_SX3_EXPERIMENTAL
  NEEDS_SEQUENCE_WRAPPER_SUPPORT
  REJECT_NO_LITERAL_DELTA
  REJECT_OFF_GATE
  REJECT_REGRESSION
  REJECT_NONDETERMINISM
"""
from __future__ import annotations

import argparse
import json
import os


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rc2", required=True)
    p.add_argument("--rc3", required=True)
    p.add_argument("--minimal", required=True)
    p.add_argument("--preservation", required=True)
    p.add_argument("--determinism", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args(argv)

    rc2 = json.load(open(args.rc2))
    rc3 = json.load(open(args.rc3))
    mn = json.load(open(args.minimal))
    pr = json.load(open(args.preservation))
    dt = json.load(open(args.determinism))

    b2 = {r["full_name"]: r for r in rc2["per_theorem"]}
    b3 = {r["full_name"]: r for r in rc3["per_theorem"]}
    names = sorted(set(b2) | set(b3))
    rc2_solved = sorted(fn for fn in names if b2.get(fn, {}).get("finished"))
    rc3_solved = sorted(fn for fn in names if b3.get(fn, {}).get("finished"))
    new_wins = sorted(fn for fn in names if b3.get(fn, {}).get("finished") and not b2.get(fn, {}).get("finished"))
    regressions = sorted(fn for fn in names if b2.get(fn, {}).get("finished") and not b3.get(fn, {}).get("finished"))

    true_wins = mn.get("true_sx3_set_ite_aesop_wins", [])
    # fresh = true wins that are NOT in the deferred reproduction set
    deferred = {"Set.ite_inter", "Set.ite_inter_self", "Set.ite_compl", "Set.ite_inter_compl_self"}
    fresh_true = sorted(set(true_wins) - deferred)
    reproduced_deferred = sorted(set(true_wins) & deferred)

    off_gate = pr.get("summary", {}).get("total_off_gate_emissions", 0)
    floors_pass = pr.get("summary", {}).get("all_floors_pass", False)
    pr_regression = pr.get("summary", {}).get("any_regression_vs_rc2_doc", False)
    det_status = dt.get("classification")

    credited_delta = len(true_wins)
    raw_delta = len(rc3_solved) - len(rc2_solved)

    # decision logic
    if regressions or pr_regression:
        decision = "REJECT_REGRESSION"
    elif off_gate > 0:
        decision = "REJECT_OFF_GATE"
    elif det_status == "nondeterministic":
        decision = "REJECT_NONDETERMINISM"
    elif credited_delta <= 0 or not fresh_true:
        decision = "REJECT_NO_LITERAL_DELTA"
    elif not floors_pass:
        decision = "KEEP_SX3_EXPERIMENTAL"
    elif det_status == "deterministic":
        decision = "RC3_RELEASE_CANDIDATE_CONFIRMED"
    elif det_status == "deterministic_except_environment_open_flake":
        decision = "RC3_CANDIDATE_CONFIRMED_WITH_ENV_FLAKE"
    else:
        decision = "KEEP_SX3_EXPERIMENTAL"

    criteria = {
        "positive_credited_delta": credited_delta > 0,
        "at_least_one_fresh_true_win": bool(fresh_true),
        "zero_regressions": not regressions and not pr_regression,
        "zero_off_gate": off_gate == 0,
        "canonical_floors_pass": floors_pass,
        "minimal_attribution_confirms": credited_delta == len(true_wins) and credited_delta > 0,
        "deterministic_or_env_flake_only": det_status in ("deterministic", "deterministic_except_environment_open_flake"),
    }

    out = {
        "validation_surface": {
            "total_theorems": len(names),
            "rc2_solved": len(rc2_solved), "rc3_solved": len(rc3_solved),
            "raw_delta": raw_delta,
            "rc2_solved_names": rc2_solved, "rc3_solved_names": rc3_solved,
        },
        "new_wins_over_rc2": new_wins,
        "regressions_vs_rc2": regressions,
        "minimal_attribution": {
            "true_sx3_set_ite_aesop_wins": sorted(true_wins),
            "credited_delta": credited_delta,
            "reproduced_deferred": reproduced_deferred,
            "fresh_true_wins": fresh_true,
            "histogram": mn.get("classification_histogram", {}),
        },
        "preservation_offgate": {
            "all_floors_pass": floors_pass,
            "total_off_gate_emissions": off_gate,
            "any_regression_vs_rc2_doc": pr_regression,
            "floors": {k: {"rc3_solved": v.get("rc3_solved"), "total": v.get("total"),
                           "floor_min": v.get("floor_min"), "pass": v.get("floor_pass")}
                       for k, v in pr.get("floors", {}).items() if isinstance(v, dict)},
        },
        "determinism": {
            "status": det_status,
            "run1_hash": dt.get("run1_hash"), "run2_hash": dt.get("run2_hash"),
            "hash_match": dt.get("hash_match"), "num_open_flakes": dt.get("num_open_flakes"),
        },
        "criteria": criteria,
        "credited_delta": credited_delta,
        "raw_delta": raw_delta,
        "fresh_true_wins": fresh_true,
        "decision": decision,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    L = ["# RC3 validation comparison", "",
         f"## DECISION: `{decision}`", "",
         "## Headline", "",
         f"- validation surface: **{len(names)}** theorems",
         f"- literal RC2 solved: **{len(rc2_solved)}**",
         f"- literal RC3 solved: **{len(rc3_solved)}**",
         f"- raw delta: **{raw_delta:+d}**",
         f"- **credited SX3 delta (minimal-attribution TRUE wins): {credited_delta}**",
         f"  - reproduced deferred: {reproduced_deferred}",
         f"  - fresh true wins: {fresh_true}",
         f"- regressions vs RC2: **{len(regressions)}** {regressions or ''}",
         f"- off-gate emissions: **{off_gate}**",
         f"- canonical floors pass: **{floors_pass}**",
         f"- determinism: **{det_status}** (hashes {dt.get('run1_hash')} vs {dt.get('run2_hash')}, "
         f"match={dt.get('hash_match')}, open_flakes={dt.get('num_open_flakes')})", "",
         "## Criteria for RC3_RELEASE_CANDIDATE_CONFIRMED", ""]
    for k, v in criteria.items():
        L.append(f"- {'✅' if v else '❌'} {k}")
    L += ["", "## New wins over literal RC2", ""]
    for fn in new_wins:
        cls = next((r["classification"] for r in mn.get("per_win", []) if r["full_name"] == fn), "?")
        L.append(f"- `{fn}` — {cls}")
    L += ["", "## Canonical floors", "",
          "| floor | RC3 solved | total | min | pass |", "|---|---|---|---|---|"]
    for k, v in out["preservation_offgate"]["floors"].items():
        L.append(f"| {k} | {v['rc3_solved']} | {v['total']} | {v['floor_min']} | {'✅' if v['pass'] else '❌'} |")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[compare] DECISION={decision} credited={credited_delta} fresh={fresh_true} "
          f"regr={len(regressions)} offgate={off_gate} det={det_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
