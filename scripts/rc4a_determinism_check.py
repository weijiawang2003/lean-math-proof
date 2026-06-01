#!/usr/bin/env python3
"""RC4A Part 9 — determinism check.

Re-runs the gated candidate probe twice over known_wins + same_cluster_holdout +
negative_controls and compares per-theorem (gate-fired, solved) outcomes. Negative
controls contribute 0 gate emissions (so 0 probes) — they verify the gate decision is
stable. Reports run1/run2 hashes, diffs, flakes, deterministic true/false.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4a_gate as G  # noqa: E402

_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
SETS = ("known_wins", "same_cluster_holdout", "negative_controls")


def _p(*a):
    return os.path.join(_REPO, *a)


def worker(args):
    case = json.loads(args.case_json)
    res = G.run_tactics_live(case["file_path"], case["full_name"], [case["tactic"]],
                             open_timeout=args.open_timeout, per_tactic=args.timeout_per_tactic)
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


def _one_run(targets, args):
    """Return {full_name: {gate_fired, solved, outcome}} for one pass."""
    out = {}
    for t in targets:
        fn, fp, tac = t["full_name"], t["file_path"], t["tactic"]
        if not tac:  # gate did not fire
            out[fn] = {"gate_fired": False, "solved": False, "outcome": "no_emission"}
            continue
        with __import__("tempfile").NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            wout = tf.name
        cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout),
               sys.executable, os.path.abspath(__file__), "--worker", "--worker-out", wout,
               "--case-json", json.dumps({"full_name": fn, "file_path": fp, "tactic": tac}),
               "--open-timeout", str(args.open_timeout),
               "--timeout-per-tactic", str(args.timeout_per_tactic)]
        subprocess.run(cmd, capture_output=True, text=True)
        try:
            wres = json.load(open(wout))
        finally:
            try:
                os.unlink(wout)
            except OSError:
                pass
        ran = wres.get("ran", [])
        r0 = ran[0] if ran else {}
        out[fn] = {"gate_fired": True, "solved": bool(r0.get("solved")),
                   "outcome": r0.get("outcome"), "setup_error": wres.get("setup_error")}
    return out


def _hash(run):
    norm = {fn: {"gate_fired": v["gate_fired"], "solved": v["solved"]}
            for fn, v in sorted(run.items())}
    return hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--case-json")
    ap.add_argument("--manifest")
    ap.add_argument("--policy")
    ap.add_argument("--out-json")
    ap.add_argument("--open-timeout", type=int, default=90)
    ap.add_argument("--timeout-per-tactic", type=int, default=12)
    ap.add_argument("--hard-timeout", type=int, default=300)
    args = ap.parse_args()
    if args.worker:
        sys.exit(worker(args))

    manifest = json.load(open(_p(args.manifest)))
    policy = G.load_policy(args.policy)
    targets, seen = [], set()
    for setname in SETS:
        rel = manifest["set_files"].get(setname)
        if not rel:
            continue
        for e in json.load(open(_p(rel))):
            fn = e["full_name"]
            if fn in seen:
                continue
            seen.add(fn)
            fires, tac, _ = G.gate_fires(policy, e.get("goal_text"), fn)
            targets.append({"full_name": fn, "file_path": e.get("file_path"),
                            "tactic": tac if fires else None})

    print(f"[rc4a-determinism] run 1/2 over {len(targets)} targets "
          f"({sum(1 for t in targets if t['tactic'])} gate-firing) ...", flush=True)
    run1 = _one_run(targets, args)
    print("[rc4a-determinism] run 2/2 ...", flush=True)
    run2 = _one_run(targets, args)

    diffs = []
    for fn in run1:
        a, b = run1[fn], run2[fn]
        if a["gate_fired"] != b["gate_fired"] or a["solved"] != b["solved"]:
            diffs.append({"full_name": fn, "run1": a, "run2": b})
    flakes = [fn for fn in run1 if run1[fn].get("setup_error") or run2[fn].get("setup_error")]
    h1, h2 = _hash(run1), _hash(run2)
    deterministic = (h1 == h2) and not diffs

    out = {"generated_by": "scripts/rc4a_determinism_check.py",
           "num_targets": len(targets),
           "num_gate_firing": sum(1 for t in targets if t["tactic"]),
           "run1_hash": h1, "run2_hash": h2, "diffs": diffs,
           "open_flakes": flakes, "deterministic": deterministic,
           "run1": run1, "run2": run2}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    print(f"[rc4a-determinism] hash {h1} vs {h2} | diffs={len(diffs)} | "
          f"deterministic={deterministic}")


if __name__ == "__main__":
    main()
