#!/usr/bin/env python3
"""RC3 determinism + LeanDojo open-flake audit.

Runs the RC3 candidate TWICE (sequentially — never concurrent Dojo drivers) on a
cheap, signal-bearing subset (sx3_known_deferred + sx3_fresh_win, plus
sx3_set_ite_holdout if --include-holdout) via scripts/rc3_run_literal_validation.py,
then compares the two runs:

  * run1 / run2 result hash (over sorted (full_name, finished, winning_tactic))
  * per-theorem diff of `finished` and `winning_tactic`
  * Dojo open failures (available==False with a skip_reason) per run
  * proof-result diffs

Classification:
  deterministic                            identical finished+winning_tactic, no open failures
  deterministic_except_environment_open_flake
                                           finished identical on every theorem that opened in
                                           both runs, but >=1 theorem failed to open in some run
  nondeterministic                         a theorem opened in both runs but its finished differs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNNER = os.path.join(_REPO, "scripts", "rc3_run_literal_validation.py")


def _hash(per):
    payload = json.dumps(sorted([(r["full_name"], bool(r["finished"]), r.get("winning_tactic"))
                                 for r in per]), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _run(manifest, sets, wrapper, route, out_json, tag):
    out_dir = os.path.join(os.path.dirname(out_json), f"determinism_{tag}_runs")
    cmd = ["/opt/anaconda3/bin/python3", _RUNNER,
           "--manifest", manifest, "--theorem-sets", *sets,
           "--policy", f"rc3_det_{tag}", "--strategy-config", wrapper,
           "--route-config", route, "--top-k", "8", "--max-steps", "8",
           "--out-dir", out_dir, "--out-json", out_json]
    print(f"[determinism] run {tag} ...", flush=True)
    subprocess.run(cmd, check=False)
    return json.load(open(out_json))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--candidate-wrapper", required=True)
    p.add_argument("--route-config", default="project/evolve/routing/ns24_router.json")
    p.add_argument("--include-holdout", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    man = json.load(open(args.manifest))
    sets_by = {t["name"]: t["file"] for t in man["theorem_sets"]}
    chosen = ["sx3_known_deferred", "sx3_fresh_win"]
    if args.include_holdout:
        chosen.append("sx3_set_ite_holdout")
    set_files = [sets_by[n] for n in chosen if n in sets_by]

    base = os.path.dirname(args.out)
    r1 = _run(args.manifest, set_files, args.candidate_wrapper, args.route_config,
              os.path.join(base, "rc3_determinism_run1.json"), "run1")
    r2 = _run(args.manifest, set_files, args.candidate_wrapper, args.route_config,
              os.path.join(base, "rc3_determinism_run2.json"), "run2")

    h1, h2 = _hash(r1["per_theorem"]), _hash(r2["per_theorem"])
    b1 = {r["full_name"]: r for r in r1["per_theorem"]}
    b2 = {r["full_name"]: r for r in r2["per_theorem"]}
    names = sorted(set(b1) | set(b2))

    diffs, open_flakes, proof_diffs = [], [], []
    for fn in names:
        a, b = b1.get(fn), b2.get(fn)
        a_open = a and a.get("available")
        b_open = b and b.get("available")
        if not a_open or not b_open:
            open_flakes.append({"full_name": fn,
                                "run1_available": bool(a_open), "run2_available": bool(b_open)})
            continue
        if bool(a["finished"]) != bool(b["finished"]):
            proof_diffs.append({"full_name": fn,
                                "run1_finished": a["finished"], "run2_finished": b["finished"]})
        if (bool(a["finished"]) != bool(b["finished"])) or (a.get("winning_tactic") != b.get("winning_tactic")):
            diffs.append({"full_name": fn,
                          "run1": {"finished": a["finished"], "win": a.get("winning_tactic")},
                          "run2": {"finished": b["finished"], "win": b.get("winning_tactic")}})

    if proof_diffs:
        status = "nondeterministic"
    elif open_flakes:
        status = "deterministic_except_environment_open_flake"
    else:
        status = "deterministic"

    out = {
        "subset": chosen,
        "run1_hash": h1, "run2_hash": h2, "hash_match": h1 == h2,
        "num_theorems": len(names),
        "open_flakes": open_flakes,
        "num_open_flakes": len(open_flakes),
        "proof_result_diffs": proof_diffs,
        "per_theorem_diffs": diffs,
        "classification": status,
        "interpretation": {
            "deterministic": "identical finished + winning_tactic across both runs; no open failures.",
            "deterministic_except_environment_open_flake": "every theorem that opened in BOTH runs has identical finished; >=1 open failure is an environment (LeanDojo open) flake, not a proof-result flake.",
            "nondeterministic": "a theorem opened in both runs but produced different finished.",
        }[status],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[determinism] {status} | hashes {h1} vs {h2} match={h1==h2} "
          f"open_flakes={len(open_flakes)} proof_diffs={len(proof_diffs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
