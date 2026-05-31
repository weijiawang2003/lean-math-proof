#!/usr/bin/env python3
"""RC2 Part 7 — determinism check for the RC2 candidate (full-wrapper).

Re-runs the RC2 candidate wrapper on a small surface set (known_wins, fresh_holdout,
one canonical smoke = demo_v1) and compares the per-theorem finished signature to a
fresh second run. Reports stable hashes + per-theorem diffs.

Outputs:
  rc2_determinism_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sig(results, names):
    sig = {}
    for s in results.get("per_surface", []):
        if s["name"] not in names:
            continue
        for t in s.get("theorems", []):
            sig[t["full_name"]] = bool(t.get("finished"))
    return sig


def _hash(sig):
    return hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:16]


def _run(manifest, wrapper, only, out):
    cmd = [sys.executable, os.path.join("scripts", "rc2_run_benchmark.py"),
           "--manifest", manifest, "--policy", "rc2_candidate",
           "--strategy-config", wrapper, "--only", only, "--out", out]
    subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True)
    return json.load(open(out)) if os.path.exists(out) else {"per_surface": []}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--candidate-wrapper", required=True)
    p.add_argument("--existing-rc2",
                   default="project/evolve/experiments/rc2/out/rc2_candidate_results.json")
    p.add_argument("--sets", default="set_ite_known_wins,set_ite_fresh_holdout,demo_v1")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    names = set(args.sets.split(","))
    only = ",".join(names)

    # run 1: reuse the existing RC2 candidate results if they cover the sets, else run
    existing = json.load(open(args.existing_rc2)) if os.path.exists(args.existing_rc2) else {"per_surface": []}
    have = {s["name"] for s in existing.get("per_surface", []) if not s.get("skipped")}
    if names.issubset(have):
        run1 = existing
        run1_src = "existing rc2_candidate_results.json"
    else:
        run1 = _run(args.manifest, args.candidate_wrapper, only,
                    "/tmp/rc2_det_run1.json")
        run1_src = "fresh run1"
    sig1 = _sig(run1, names)
    h1 = _hash(sig1)

    run2 = _run(args.manifest, args.candidate_wrapper, only, "/tmp/rc2_det_run2.json")
    sig2 = _sig(run2, names)
    h2 = _hash(sig2)

    diffs = [{"full_name": k, "run1": sig1.get(k), "run2": sig2.get(k)}
             for k in sorted(set(sig1) | set(sig2)) if sig1.get(k) != sig2.get(k)]
    out = {"sets": sorted(names), "theorems_checked": len(set(sig1) | set(sig2)),
           "run1_source": run1_src, "run1_hash": h1, "run2_hash": h2,
           "deterministic": (h1 == h2 and not diffs), "per_theorem_diffs": diffs,
           "note": "RC2 full-wrapper eval; deterministic search expected to reproduce."}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[rc2:det] run1={h1} run2={h2} deterministic={out['deterministic']} "
          f"diffs={len(diffs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
