#!/usr/bin/env python3
"""SF5 Part 6 — run live retrieval probes.

Driver/worker model (mirrors scripts/tr2_run_live_probes.py): one worker
subprocess per theorem under an OS hard timeout (scripts/run_with_timeout.py); the
worker opens ONE Dojo (SIGALRM-bounded) and runs every probe from the *initial*
proof state under a per-tactic SIGALRM.

RC2-failure status is taken from the prior literal confirmation
(rc2_failure_confirmation.json, all targets are CONFIRMED_RC2_FAILURE); no win here
is final — SF5 attribution (Part 7) re-judges each win against literal RC2.
Production configs are untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import traceback

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")


def _p(*a):
    return os.path.join(_REPO, *a)


class _ProbeTimeout(Exception):
    pass


def _alarm(_s, _f):
    raise _ProbeTimeout()


def _classify_outcome(err, solved):
    if solved:
        return "success"
    e = (err or "").lower()
    if not e:
        return "proof_failed"
    if "unknown identifier" in e or "unknown constant" in e:
        return "unknown_name"
    if "maximum recursion" in e or "maxrecdepth" in e:
        return "max_recursion"
    # tactic-level shape failures (rewrite/simp etc.) are NOT parse errors
    if ("'rewrite' failed" in e or "'rw' failed" in e or "motive is not type correct" in e
            or "did not find instance" in e or "made no progress" in e
            or "equality or iff proof expected" in e):
        return "proof_failed"
    if "unexpected token" in e or "unexpected identifier" in e \
            or ("expected" in e and ("'{'" in e or "term" in e or "command" in e)):
        return "parse_error"
    return "proof_failed"


# ------------------------------- worker -----------------------------------
def worker(args):
    case = json.loads(args.case_json)
    probes = json.loads(args.probes_json)  # list of {tactic, family, lemma, ...}
    full_name = case["full_name"]
    res = {"full_name": full_name, "file_path": case.get("file_path"),
           "live": False, "ran": [], "setup_error": None}
    try:
        sys.path.insert(0, _REPO)
        import env as _env
        from core_types import TheoremConfig as _TC
        from lean_dojo import Dojo as _Dojo
        repo = _env.make_repo()
        thm = _env.make_theorem(repo, _TC(file_path=case["file_path"], full_name=full_name))
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(args.open_timeout)
        try:
            dojo_cm = _Dojo(thm)
            dojo, state0 = dojo_cm.__enter__()
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        try:
            res["live"] = True

            def run_one(tac):
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(args.timeout_per_tactic)
                try:
                    out = _env.run_transition(dojo, thm, state0, tac)
                    rec = getattr(out, "record", None)
                    fin = bool(getattr(out, "is_finished", False))
                    err = getattr(rec, "error_message", None) if rec else None
                    dead = bool(getattr(out, "session_dead", False))
                except _ProbeTimeout:
                    return {"solved": False, "outcome": "timeout", "dead": False}
                except Exception as e:
                    return {"solved": False, "outcome": "exception",
                            "error": f"{type(e).__name__}: {str(e)[:140]}", "dead": False}
                finally:
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(0)
                r = {"solved": bool(fin), "outcome": _classify_outcome(err, fin),
                     "dead": bool(dead)}
                if err and not fin:
                    r["error"] = err[:200]
                return r

            for pr in probes:
                r = run_one(pr["tactic"])
                r.update({"tactic": pr["tactic"], "family": pr["family"],
                          "lemma": pr.get("lemma"), "diagnostic": pr.get("diagnostic", False),
                          "parse_risk": pr.get("parse_risk")})
                res["ran"].append(r)
                if r["dead"]:
                    break
        finally:
            try:
                dojo_cm.__exit__(None, None, None)
            except Exception:
                pass
    except _ProbeTimeout:
        res["setup_error"] = f"dojo open exceeded {args.open_timeout}s"
    except Exception as e:
        res["setup_error"] = f"{type(e).__name__}: {str(e)[:200]}\n" + \
            traceback.format_exc()[-300:]
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


# ------------------------------- driver -----------------------------------
def _rc2_status_map(path):
    m = {}
    if path and os.path.exists(_p(path)):
        for r in json.load(open(_p(path))).get("results", []):
            m[r["full_name"]] = r.get("classification")
    return m


def driver(args):
    plan = json.load(open(_p(args.probe_plan)))
    rc2 = _rc2_status_map(args.rc2_confirmation)

    results = []
    win_hist = {"retrieval_win": 0, "no_win": 0, "needs_review": 0}
    n_live = 0
    for t in plan["theorems"]:
        fn = t["full_name"]
        fp = t.get("file_path")
        probes = t.get("probes", [])
        rec = {"full_name": fn, "file_path": fp, "namespace": t.get("namespace"),
               "cluster_id": t.get("cluster_id"),
               "rc2_status": rc2.get(fn, "unknown"),
               "retrieval_probe_count": len(probes),
               "live": False, "setup_error": None,
               "wins": [], "best_win": None, "failed_lemmas": [],
               "outcome_histogram": {}, "ran": []}
        if not fp:
            rec["setup_error"] = "no file_path"
            rec["classification_pre_attribution"] = "needs_review"
            results.append(rec)
            win_hist["needs_review"] += 1
            continue

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            wout = tf.name
        cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout),
               sys.executable, os.path.abspath(__file__), "--worker",
               "--worker-out", wout,
               "--case-json", json.dumps({"full_name": fn, "file_path": fp}),
               "--probes-json", json.dumps(probes),
               "--open-timeout", str(args.open_timeout),
               "--timeout-per-tactic", str(args.timeout_per_tactic)]
        print(f"[sf5-probe] {fn}: {len(probes)} probes ...", flush=True)
        rc = subprocess.run(cmd, capture_output=True, text=True).returncode
        try:
            wres = json.load(open(wout))
        except Exception:
            wres = {"live": False, "setup_error": f"worker no output (rc={rc})", "ran": []}
        finally:
            try:
                os.unlink(wout)
            except OSError:
                pass

        rec["live"] = bool(wres.get("live"))
        rec["setup_error"] = wres.get("setup_error")
        ran = wres.get("ran", [])
        rec["ran"] = ran
        if rec["live"]:
            n_live += 1
        hist = {}
        for r in ran:
            hist[r["outcome"]] = hist.get(r["outcome"], 0) + 1
            if r.get("solved"):
                rec["wins"].append({"tactic": r["tactic"], "family": r["family"],
                                    "lemma": r.get("lemma"),
                                    "diagnostic": r.get("diagnostic", False)})
            elif r.get("lemma"):
                rec["failed_lemmas"].append(r["lemma"])
        rec["outcome_histogram"] = hist
        rec["failed_lemmas"] = sorted(set(rec["failed_lemmas"]))
        if rec["wins"]:
            # prefer a non-diagnostic, named-lemma win as the headline
            named = [w for w in rec["wins"] if w.get("lemma")]
            rec["best_win"] = (named[0] if named else rec["wins"][0])
            rec["classification_pre_attribution"] = "retrieval_win"
            win_hist["retrieval_win"] += 1
        elif rec["setup_error"]:
            rec["classification_pre_attribution"] = "needs_review"
            win_hist["needs_review"] += 1
        else:
            rec["classification_pre_attribution"] = "no_win"
            win_hist["no_win"] += 1
        results.append(rec)

    out = {
        "generated_by": "scripts/sf5_run_retrieval_probes.py",
        "probe_plan_input": args.probe_plan,
        "rc2_confirmation_input": args.rc2_confirmation,
        "num_theorems": len(results),
        "num_live": n_live,
        "win_histogram": win_hist,
        "open_timeout": args.open_timeout,
        "timeout_per_tactic": args.timeout_per_tactic,
        "hard_timeout": args.hard_timeout,
        "results": results,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# SF5 retrieval probe results", "",
          f"- theorems: {len(results)} | live: {n_live}",
          f"- wins: {win_hist['retrieval_win']} | no_win: {win_hist['no_win']} | "
          f"needs_review: {win_hist['needs_review']}", ""]
    md.append("| target | live | probes | wins | best_win |")
    md.append("|---|---|---|---|---|")
    for r in results:
        bw = r["best_win"]["tactic"] if r["best_win"] else ""
        md.append(f"| {r['full_name']} | {r['live']} | {r['retrieval_probe_count']} | "
                  f"{len(r['wins'])} | `{bw}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")

    print(f"[sf5-probe] done: live={n_live}/{len(results)} win_hist={win_hist}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--case-json")
    ap.add_argument("--probes-json")
    ap.add_argument("--probe-plan")
    ap.add_argument("--rc2-confirmation",
                    default="project/evolve/experiments/sf4/out/rc2_failure_confirmation.json")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--open-timeout", type=int, default=90)
    ap.add_argument("--timeout-per-tactic", type=int, default=15)
    ap.add_argument("--hard-timeout", type=int, default=900)
    args = ap.parse_args()
    if args.worker:
        sys.exit(worker(args))
    driver(args)


if __name__ == "__main__":
    main()
