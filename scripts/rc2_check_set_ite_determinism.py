#!/usr/bin/env python3
"""RC2 Part 7 — determinism check for the SET_ITE_SIMP candidate.

Re-runs the candidate evaluation on the known-win + fresh-holdout sets and compares
the per-theorem outcome signature (full_name -> candidate_finished, set_ite_finished,
gate_fired, new_win) against the existing candidate_results.json (run 1). Reports a
stable hash per run and any per-theorem diffs.

A deterministic candidate should produce identical signatures.

Outputs:
  determinism_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sig(results, names=None):
    """Stable per-theorem signature dict (subset of fields that must be stable)."""
    sig = {}
    for r in results:
        fn = r["full_name"]
        if names is not None and fn not in names:
            continue
        sig[fn] = {"candidate_finished": bool(r.get("candidate_finished")),
                   "set_ite_gate_fired": bool(r.get("set_ite_gate_fired")),
                   "set_ite_finished": bool(r.get("set_ite_finished")),
                   "new_win": bool(r.get("new_win_over_literal_rc1"))}
    return sig


def _hash(sig):
    return hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:16]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-results",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/out/candidate_results.json")
    p.add_argument("--rerun", default="true")
    p.add_argument("--sets", default="set_ite_known_wins,set_ite_fresh_holdout")
    p.add_argument("--out",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/out/determinism_check.json")
    args = p.parse_args(argv)

    run1 = json.load(open(args.candidate_results))["results"]
    # restrict to theorems in the named sets
    names = {r["full_name"] for r in run1
             if any(s in (r.get("in_sets") or []) for s in args.sets.split(","))}
    sig1 = _sig(run1, names)
    h1 = _hash(sig1)

    rerun_path = "/tmp/rc2_cand_rerun.json"
    diffs = []
    if str(args.rerun).lower() in ("1", "true", "yes"):
        cmd = [sys.executable, os.path.join("scripts", "rc2_run_set_ite_candidate.py"),
               "--sets", args.sets, "--out-json", rerun_path,
               "--out-md", "/tmp/rc2_cand_rerun.md"]
        print(f"[rc2:det] re-running candidate on {args.sets} ...", flush=True)
        subprocess.run([sys.executable] + cmd[1:], cwd=_REPO,
                       capture_output=True, text=True)
        run2 = json.load(open(rerun_path))["results"]
        sig2 = _sig(run2, names)
    else:
        sig2 = sig1
    h2 = _hash(sig2)

    for fn in sorted(set(sig1) | set(sig2)):
        if sig1.get(fn) != sig2.get(fn):
            diffs.append({"full_name": fn, "run1": sig1.get(fn), "run2": sig2.get(fn)})

    out = {"sets": args.sets.split(","), "theorems_checked": len(names),
           "run1_hash": h1, "run2_hash": h2,
           "deterministic": (h1 == h2 and not diffs),
           "per_theorem_diffs": diffs,
           "note": "Candidate = deterministic literal-RC1 lookup + single fixed tactic "
                   "`simp [Set.ite]`; expected identical signatures across runs."}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[rc2:det] run1={h1} run2={h2} deterministic={out['deterministic']} "
          f"diffs={len(diffs)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
