#!/usr/bin/env python3
"""RC5S Part 6 — timeout-safe live runner (core engineering deliverable).

Runs each theorem's program batch in an ISOLATED subprocess wrapped by `run_with_timeout.py`,
which enforces a hard per-theorem wall-clock cap via PROCESS-GROUP kill (SIGTERM→SIGKILL),
guaranteeing that a stuck LeanDojo / aesop / simp_all cannot stall the whole run even when it
ignores the per-tactic SIGALRM. The inner probe reuses the validated TR5 worker. Every theorem
records started_at / ended_at / wall_seconds / killed_by_timeout / exit_code / outcomes.
Per-theorem checkpoint + deterministic resume. Reusable by later RC5H-v2 tasks.

Usage:
  driver:  imported via run_plan(...), or CLI below.
  worker:  --worker (delegates the Lean probing to tr5_run_ranked_live_search.worker).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tr5_run_ranked_live_search as R  # validated one-Dojo probe worker

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")


def _p(*a):
    return os.path.join(_REPO, *a)


def worker(args):
    """Delegate to the TR5 worker (opens one Dojo, runs the programs, writes outcomes)."""
    return R.worker(args)


def run_one_theorem(fn, file_path, programs, *, wall_cap, per_tactic, open_timeout, run_controls):
    """Spawn an isolated, process-group-killable subprocess for one theorem. Returns a record
    with timing + kill classification + the worker's tactic outcomes (or a bounded-timeout marker)."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        wout = tf.name
    cmd = [sys.executable, _TIMEOUT_HELPER, str(wall_cap), sys.executable,
           os.path.abspath(__file__), "--worker", "--worker-out", wout,
           "--case-json", json.dumps({"full_name": fn, "file_path": file_path}),
           "--programs-json", json.dumps(programs),
           "--run-controls", "true" if run_controls else "false",
           "--open-timeout", str(open_timeout), "--timeout-per-tactic", str(per_tactic)]
    started = time.time()
    rc = subprocess.run(cmd, capture_output=True, text=True)
    ended = time.time()
    killed = rc.returncode == 124  # run_with_timeout exit code on timeout
    wres = None
    try:
        wres = json.load(open(wout))
    except Exception:
        wres = None
    finally:
        try:
            os.unlink(wout)
        except OSError:
            pass

    ran = (wres or {}).get("ran", []) if wres else []
    controls = (wres or {}).get("controls", []) if wres else []
    win = next((r for r in ran if r.get("solved")), None)
    rec = {
        "full_name": fn, "started_at": round(started, 3), "ended_at": round(ended, 3),
        "wall_seconds": round(ended - started, 2), "killed_by_timeout": killed,
        "exit_code": rc.returncode,
        "live": bool((wres or {}).get("live")) if wres else False,
        "setup_error": (wres or {}).get("setup_error") if wres else ("worker_no_output_or_killed" if killed else "worker_no_output"),
        "programs_attempted": len(ran),
        "success": bool(win),
        "first_success_rank": (win.get("rank") if win else None),
        "winning_program": ({"rank": win["rank"], "tactic": win["tactic"], "family": win.get("family"),
                             "used_lemmas": win.get("lemmas", []), "ranker_score": win.get("ranker_score")}
                            if win else None),
        "controls": controls,
        "control_wins": [c["tactic"] for c in controls if c.get("solved")],
        "failures": [{"rank": r.get("rank"), "tactic": r["tactic"], "outcome": r.get("outcome")}
                     for r in ran if not r.get("solved")],
        "stderr_tail": (rc.stderr or "")[-300:] if killed else None,
    }
    return rec


def run_plan(theorems, *, budget, rank_lo, run_controls, only_unsolved, ckpt_path,
             wall_cap, per_tactic, open_timeout, label):
    """Driver loop with per-theorem checkpoint + deterministic resume. theorems carry
    programs_ranked (each with a `rank`)."""
    ckpt = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}
    results = []
    for t in sorted(theorems, key=lambda x: x["full_name"]):  # deterministic order
        fn = t["full_name"]
        if fn in ckpt:
            results.append(ckpt[fn]); continue
        if only_unsolved is not None and fn not in only_unsolved:
            continue
        progs = [p for p in t.get("programs_ranked", []) if rank_lo <= p.get("rank", 99) <= budget]
        if not progs:
            rec = {"full_name": fn, "wall_seconds": 0.0, "killed_by_timeout": False,
                   "exit_code": 0, "live": None, "programs_attempted": 0, "success": False,
                   "winning_program": None, "failures": [], "controls": [],
                   "setup_error": "no programs in window"}
            results.append(rec); ckpt[fn] = rec
            json.dump(ckpt, open(ckpt_path, "w"), ensure_ascii=False, indent=2)
            continue
        print(f"[rc5s-{label}] {fn}: ranks {rank_lo}-{budget} ({len(progs)} programs, cap {wall_cap}s) ...",
              flush=True)
        rec = run_one_theorem(fn, t.get("file_path"), progs, wall_cap=wall_cap,
                              per_tactic=per_tactic, open_timeout=open_timeout, run_controls=run_controls)
        rec["budget"] = budget
        results.append(rec)
        ckpt[fn] = rec
        json.dump(ckpt, open(ckpt_path, "w"), ensure_ascii=False, indent=2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--case-json")
    ap.add_argument("--programs-json")
    ap.add_argument("--run-controls", default="true")
    ap.add_argument("--open-timeout", type=int, default=90)
    ap.add_argument("--timeout-per-tactic", type=int, default=8)
    args = ap.parse_args()
    if args.worker:
        # tr5 worker reads these exact fields
        wargs = types.SimpleNamespace(worker_out=args.worker_out, case_json=args.case_json,
                                      programs_json=args.programs_json, run_controls=args.run_controls,
                                      open_timeout=args.open_timeout,
                                      timeout_per_tactic=args.timeout_per_tactic)
        sys.exit(worker(wargs))
    print("rc5s_timeout_safe_runner: library module; use --worker, or import run_plan/run_one_theorem.")


if __name__ == "__main__":
    main()
