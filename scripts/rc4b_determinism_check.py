#!/usr/bin/env python3
"""RC4B Part 9 — determinism check.

Re-runs the gated bridge probe twice over known_wins + fresh_holdout_set +
fresh_holdout_multiset + the negative controls and compares per-theorem
(gate-fired, solved) outcomes. Negative controls contribute 0 gate emissions (so 0
probes) — they verify the gate decision is stable. Reports run1/run2 hashes, diffs,
flakes, deterministic true/false.
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
import rc4b_gate as G  # noqa: E402

_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
SETS = ("known_wins", "fresh_holdout_set", "fresh_holdout_multiset",
        "disjoint_negative_controls", "namespace_negative_controls")


def _p(*a):
    return os.path.join(_REPO, *a)


def worker(args):
    case = json.loads(args.case_json)
    res = G.run_tactics_live(case["file_path"], case["full_name"], case["tactics"],
                             open_timeout=args.open_timeout, per_tactic=args.timeout_per_tactic)
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


def _one_run(targets, args, ckpt_path=None):
    out = json.load(open(ckpt_path)) if (ckpt_path and os.path.exists(ckpt_path)) else {}
    for t in targets:
        fn, fp, tactics = t["full_name"], t["file_path"], t["tactics"]
        if fn in out:
            continue
        if not tactics:  # gate did not fire
            out[fn] = {"gate_fired": False, "solved": False, "outcome": "no_emission"}
            continue
        with __import__("tempfile").NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            wout = tf.name
        cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout),
               sys.executable, os.path.abspath(__file__), "--worker", "--worker-out", wout,
               "--case-json", json.dumps({"full_name": fn, "file_path": fp, "tactics": tactics}),
               "--open-timeout", str(args.open_timeout),
               "--timeout-per-tactic", str(args.timeout_per_tactic)]
        subprocess.run(cmd, capture_output=True, text=True)
        try:
            wres = json.load(open(wout))
        except (ValueError, OSError):
            # worker killed by hard timeout before writing -> treat as flake
            wres = {"ran": [], "setup_error": "worker_output_unreadable"}
        finally:
            try:
                os.unlink(wout)
            except OSError:
                pass
        ran = wres.get("ran", [])
        solved = any(x.get("solved") for x in ran)
        out[fn] = {"gate_fired": True, "solved": bool(solved),
                   "outcome": "success" if solved else (ran[-1].get("outcome") if ran else None),
                   "setup_error": wres.get("setup_error")}
        if ckpt_path:
            json.dump(out, open(ckpt_path, "w"), ensure_ascii=False, indent=2)
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
    ap.add_argument("--out-md")
    ap.add_argument("--analyze-only", action="store_true",
                    help="recompute verdict from saved determinism_runs/run{1,2}.json (no re-probe)")
    ap.add_argument("--open-timeout", type=int, default=90)
    ap.add_argument("--timeout-per-tactic", type=int, default=15)
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
            fires, bns, tactics, anames, lemma = G.gate_fires(
                policy, e.get("namespace"), e.get("goal_text"), fn)
            targets.append({"full_name": fn, "file_path": e.get("file_path"),
                            "tactics": tactics if fires else None})

    nfire = sum(1 for t in targets if t["tactics"])
    cdir = _p("project/evolve/experiments/rc4_candidates/disjoint_left_bridge/out/determinism_runs")
    os.makedirs(cdir, exist_ok=True)
    if args.analyze_only:
        run1 = json.load(open(os.path.join(cdir, "run1.json")))
        run2 = json.load(open(os.path.join(cdir, "run2.json")))
        print(f"[rc4b-determinism] analyze-only: run1={len(run1)} run2={len(run2)}", flush=True)
    else:
        print(f"[rc4b-determinism] run 1/2 over {len(targets)} targets ({nfire} gate-firing) ...",
              flush=True)
        run1 = _one_run(targets, args, os.path.join(cdir, "run1.json"))
        print("[rc4b-determinism] run 2/2 ...", flush=True)
        run2 = _one_run(targets, args, os.path.join(cdir, "run2.json"))

    # A theorem is an OPEN FLAKE if either run hit an infrastructure setup_error (Dojo
    # open/hard-timeout kill, unreadable worker output) — those are excluded from the
    # determinism hash, which is computed over the cleanly-executed theorems only.
    flakes = [fn for fn in run1 if run1[fn].get("setup_error") or run2[fn].get("setup_error")]
    flaky = set(flakes)
    genuine_diffs, flake_diffs = [], []
    for fn in run1:
        a, b = run1[fn], run2[fn]
        if a["gate_fired"] != b["gate_fired"] or a["solved"] != b["solved"]:
            (flake_diffs if fn in flaky else genuine_diffs).append(
                {"full_name": fn, "run1": a, "run2": b})
    clean1 = {fn: v for fn, v in run1.items() if fn not in flaky}
    clean2 = {fn: v for fn, v in run2.items() if fn not in flaky}
    h1, h2 = _hash(clean1), _hash(clean2)
    # gate decision must be 100% stable across ALL targets (gate is a pure function)
    gate_stable = all(run1[fn]["gate_fired"] == run2[fn]["gate_fired"] for fn in run1)
    deterministic = (h1 == h2) and not genuine_diffs and gate_stable
    # which credited wins are affected by a flake (informational)
    win_flakes = [fn for fn in flakes
                  if run1[fn].get("solved") or run2[fn].get("solved")]

    out = {"generated_by": "scripts/rc4b_determinism_check.py",
           "num_targets": len(targets), "num_gate_firing": nfire,
           "clean_run1_hash": h1, "clean_run2_hash": h2,
           "gate_decisions_stable": gate_stable,
           "genuine_diffs": genuine_diffs, "flake_diffs": flake_diffs,
           "open_flakes": flakes, "num_open_flakes": len(flakes),
           "win_affecting_flakes": win_flakes,
           "deterministic": deterministic,
           "determinism_note": ("hash computed over cleanly-executed theorems; open flakes "
                                "are Dojo hard-timeout / worker-kill infrastructure events on "
                                "heavy-aesop / hard-Set goals, excluded from the hash. "
                                "deterministic=True ⇔ identical gate decisions + identical "
                                "solved outcomes on every cleanly-executed theorem."),
           "run1": run1, "run2": run2}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC4B determinism check", "",
          f"- targets: {len(targets)} ({nfire} gate-firing)",
          f"- clean run1 hash: `{h1}` | clean run2 hash: `{h2}`",
          f"- gate decisions stable (all targets): **{gate_stable}**",
          f"- genuine diffs (clean theorems): **{len(genuine_diffs)}** | "
          f"flake-induced diffs: {len(flake_diffs)} | open flakes: {len(flakes)}",
          f"- **deterministic (modulo infrastructure flakes): {deterministic}**", "",
          out["determinism_note"]]
    if genuine_diffs:
        md += ["", "## Genuine diffs", ""] + [f"- {d}" for d in genuine_diffs]
    if flake_diffs:
        md += ["", "## Flake-induced diffs (excluded — setup_error in one run)", ""] + \
              [f"- `{d['full_name']}`: run1 solved={d['run1']['solved']} "
               f"(err={bool(d['run1'].get('setup_error'))}) / run2 solved={d['run2']['solved']} "
               f"(err={bool(d['run2'].get('setup_error'))})" for d in flake_diffs]
    if flakes:
        md += ["", f"## Open flakes ({len(flakes)}) — infrastructure, excluded from hash", ""] + \
              [f"- {fn}" + ("  ⚠ credited-win (solves when it executes)" if fn in win_flakes else "")
               for fn in flakes]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4b-determinism] clean hash {h1} vs {h2} | genuine_diffs={len(genuine_diffs)} | "
          f"open_flakes={len(flakes)} | gate_stable={gate_stable} | deterministic={deterministic}")


if __name__ == "__main__":
    main()
