#!/usr/bin/env python3
"""FLI1 Part 3 — live residual-goal capture via LeanDojo.

Driver loops the rerun plan; each seed runs in an isolated worker subprocess wrapped by
run_with_timeout (process-group hard kill). The worker opens one Dojo, records the initial goal,
runs each probe sequence from the initial state (chaining next_state), and keeps the best
non-finishing residual goal (via TransitionOutcome.next_state.pp). Per-seed JSON checkpoints in
--trace-dir enable resume. A theorem closed during rerun → solved_directly (recorded, NOT FLI1
success).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")


def _p(*a):
    return os.path.join(_REPO, *a)


# ----------------------------- worker -----------------------------
def worker(case, open_timeout, per_tactic):
    import signal
    res = {"seed_id": case["seed_id"], "theorem": case["theorem"],
           "namespace": case["namespace"], "status": "needs_review",
           "initial_goal": None, "residual_goals": [], "last_successful_tactic_prefix": [],
           "failed_tactic": None, "error_message": None, "runtime_sec": None,
           "heartbeat_or_timeout": None, "capture_quality": "missing", "probe_log": []}
    import time
    t0 = time.time()

    class _PT(Exception):
        pass

    def _alarm(sig, frm):
        raise _PT()

    try:
        sys.path.insert(0, _REPO)
        import env as _env
        from core_types import TheoremConfig as _TC
        from lean_dojo import Dojo as _Dojo
        repo = _env.make_repo()
        thm = _env.make_theorem(repo, _TC(file_path=case["file_path"], full_name=case["theorem"]))
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(open_timeout)
        try:
            cm = _Dojo(thm)
            dojo, state0 = cm.__enter__()
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        res["initial_goal"] = getattr(state0, "pp", None)
        res["capture_quality"] = "medium"  # we at least have the initial goal
        best = None  # (prefix, residual_pp, num_goals)
        solved = False
        dead = False
        for probe in case["rerun_probes"]:
            cur = state0
            prefix = []
            log = {"probe": probe, "outcome": None}
            for tac in probe:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(per_tactic)
                try:
                    out = _env.run_transition(dojo, thm, cur, tac)
                except _PT:
                    log["outcome"] = "timeout"
                    res["heartbeat_or_timeout"] = f"per_tactic {per_tactic}s on `{tac}`"
                    break
                except Exception as e:  # noqa: BLE001
                    log["outcome"] = f"exception:{type(e).__name__}"
                    res["failed_tactic"] = tac
                    res["error_message"] = str(e)[:200]
                    break
                finally:
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(0)
                if getattr(out, "session_dead", False):
                    dead = True
                    log["outcome"] = "session_dead"
                    break
                if getattr(out, "is_finished", False):
                    solved = True
                    prefix.append(tac)
                    log["outcome"] = "finished"
                    break
                rec = getattr(out, "record", None)
                err = getattr(rec, "error_message", None) if rec else None
                ns = getattr(out, "next_state", None)
                if ns is None or getattr(out, "is_error", False):
                    res["failed_tactic"] = tac
                    if err:
                        res["error_message"] = err[:200]
                    el = (err or "").lower()
                    log["outcome"] = ("unknown_name" if ("unknown identifier" in el
                                      or "unknown constant" in el) else "proof_failed")
                    break
                cur = ns
                prefix.append(tac)
                log["outcome"] = "progressed"
            res["probe_log"].append(log)
            if solved:
                res["last_successful_tactic_prefix"] = prefix
                break
            if dead:
                break
            # track best progressing, non-trivial probe (prefix beyond nothing, state changed)
            if prefix and log["outcome"] in ("progressed",) and cur is not state0:
                cand = (prefix, getattr(cur, "pp", None), getattr(cur, "num_goals", None))
                if best is None or len(prefix) > len(best[0]):
                    best = cand
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass

        if solved:
            res["status"] = "solved_directly"
            res["capture_quality"] = "high"
        elif dead:
            res["status"] = "infra_error"
            res["error_message"] = res["error_message"] or "session_dead"
        elif best is not None:
            res["status"] = "captured"
            res["residual_goals"] = [best[1]]
            res["last_successful_tactic_prefix"] = best[0]
            res["capture_quality"] = "high"
            res["residual_num_goals"] = best[2]
        elif res["initial_goal"]:
            # no opener progressed, but the initial goal is itself the residual to invent against
            uo = {l["outcome"] for l in res["probe_log"]}
            if uo == {"unknown_name"}:
                res["status"] = "unknown_name"
            elif "timeout" in uo:
                res["status"] = "timeout"
            else:
                res["status"] = "captured"
                res["residual_goals"] = [res["initial_goal"]]
                res["capture_quality"] = "medium"
        else:
            res["status"] = "no_goal"
    except _PT:
        res["status"] = "timeout"
        res["error_message"] = f"dojo open exceeded {open_timeout}s"
    except Exception as e:  # noqa: BLE001
        import traceback
        res["status"] = "infra_error"
        res["error_message"] = f"{type(e).__name__}: {str(e)[:160]}"
        res["traceback_tail"] = traceback.format_exc()[-300:]
    res["runtime_sec"] = round(time.time() - t0, 2)
    return res


# ----------------------------- driver -----------------------------
def driver(args):
    plan = json.load(open(_p(args.plan)))["seeds"]
    trace_dir = _p(args.trace_dir)
    os.makedirs(trace_dir, exist_ok=True)
    results = []
    for case in plan:
        ckpt = os.path.join(trace_dir, f"{case['seed_id']}.json")
        if os.path.exists(ckpt):
            results.append(json.load(open(ckpt)))
            print(f"[fli1-capture] {case['seed_id']} cached", flush=True)
            continue
        wout = ckpt + ".tmp"
        cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout),
               sys.executable, os.path.abspath(__file__), "--worker",
               "--worker-out", wout, "--case-json", json.dumps(case),
               "--open-timeout", str(args.open_timeout),
               "--timeout-per-tactic", str(args.timeout_per_tactic)]
        print(f"[fli1-capture] {case['seed_id']} {case['theorem']} ...", flush=True)
        rc = subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(wout):
            r = json.load(open(wout))
            os.replace(wout, ckpt)
        else:
            r = {"seed_id": case["seed_id"], "theorem": case["theorem"],
                 "namespace": case["namespace"],
                 "status": ("timeout" if rc.returncode == 124 else "infra_error"),
                 "initial_goal": None, "residual_goals": [], "capture_quality": "missing",
                 "error_message": (rc.stderr or "")[-200:], "runtime_sec": None}
            json.dump(r, open(ckpt, "w"))
        r["trace_path"] = os.path.relpath(ckpt, _REPO)
        results.append(r)
        print(f"  -> {r['status']} q={r.get('capture_quality')} {r.get('runtime_sec')}s", flush=True)

    with open(_p(args.out_jsonl), "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hist = Counter(r["status"] for r in results)
    captured = [r for r in results if r["status"] == "captured"]
    summary = {
        "generated_by": "scripts/fli1_capture_residual_goals.py",
        "num_seeds": len(results), "status_histogram": dict(hist),
        "captured": len(captured),
        "captured_high_quality": sum(1 for r in captured if r.get("capture_quality") == "high"),
        "solved_directly": hist.get("solved_directly", 0),
        "infra_error": hist.get("infra_error", 0), "timeout": hist.get("timeout", 0),
        "unknown_name": hist.get("unknown_name", 0),
        "captured_by_namespace": dict(Counter(r["namespace"] for r in captured).most_common()),
        "target_met_25": len(captured) >= 25,
    }
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI1 residual goal capture summary", "",
          f"- seeds: {summary['num_seeds']} | **captured: {summary['captured']}** "
          f"(high quality {summary['captured_high_quality']}) | target ≥25: "
          f"{summary['target_met_25']}",
          f"- status: {summary['status_histogram']}",
          f"- solved_directly (NOT FLI1 success): {summary['solved_directly']}",
          f"- captured by namespace: {summary['captured_by_namespace']}", "",
          "| seed | theorem | status | quality | prefix | #goals |",
          "|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['seed_id']} | `{r['theorem']}` | {r['status']} | "
                  f"{r.get('capture_quality')} | `{' ; '.join(r.get('last_successful_tactic_prefix') or [])}` | "
                  f"{r.get('residual_num_goals', '')} |")
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli1-capture] DONE captured={summary['captured']}/{summary['num_seeds']} "
          f"status={summary['status_histogram']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--case-json")
    ap.add_argument("--open-timeout", type=int, default=120)
    ap.add_argument("--timeout-per-tactic", type=int, default=30)
    ap.add_argument("--plan")
    ap.add_argument("--out-jsonl")
    ap.add_argument("--out-summary-json")
    ap.add_argument("--out-summary-md")
    ap.add_argument("--trace-dir")
    ap.add_argument("--hard-timeout", type=int, default=240)
    args = ap.parse_args()
    if args.worker:
        r = worker(json.loads(args.case_json), args.open_timeout, args.timeout_per_tactic)
        json.dump(r, open(args.worker_out, "w"), ensure_ascii=False)
        return
    driver(args)


if __name__ == "__main__":
    main()
