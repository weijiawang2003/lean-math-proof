#!/usr/bin/env python3
"""TR5 Part 5 — run the live B-budget ranker-guided search.

Driver/worker LeanDojo harness (serialized; one Dojo per theorem under an OS hard
timeout via run_with_timeout.py). For each confirmed RC2 failure the worker opens the
theorem, runs the 4 bare controls (simp / simp_all / aesop / classical <;> aesop), then
the TOP-B ranker-ordered programs in rank order, per-tactic SIGALRM-bounded, STOPPING
after the first solving program. Confirmed failures are processed first, then any
controls/known winners. Per-theorem checkpoint enables resume.

No win is final here — TR5 attribution (Part 7) re-judges every win against literal RC2.
Production configs untouched.
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
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
CONTROLS = ["simp", "simp_all", "aesop", "classical <;> aesop"]


def _p(*a):
    return os.path.join(_REPO, *a)


class _ProbeTimeout(Exception):
    pass


def _alarm(_s, _f):
    raise _ProbeTimeout()


def _classify(err, solved):
    if solved:
        return "success"
    e = (err or "").lower()
    if not e:
        return "proof_failed"
    if "unknown identifier" in e or "unknown constant" in e:
        return "unknown_name"
    if "maximum recursion" in e or "maxrecdepth" in e:
        return "max_recursion"
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
    programs = json.loads(args.programs_json)   # already sliced to the rank window
    run_controls = args.run_controls == "true"
    fn = case["full_name"]
    res = {"full_name": fn, "file_path": case.get("file_path"), "live": False,
           "controls": [], "ran": [], "setup_error": None}
    try:
        sys.path.insert(0, _REPO)
        import env as _env
        from core_types import TheoremConfig as _TC
        from lean_dojo import Dojo as _Dojo
        repo = _env.make_repo()
        thm = _env.make_theorem(repo, _TC(file_path=case["file_path"], full_name=fn))
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
                r = {"solved": bool(fin), "outcome": _classify(err, fin), "dead": bool(dead)}
                if err and not fin:
                    r["error"] = err[:200]
                return r

            if run_controls:
                for c in CONTROLS:
                    r = run_one(c)
                    r["tactic"] = c
                    res["controls"].append(r)
                    if r["dead"]:
                        raise RuntimeError("dojo died during controls")
            for pgm in programs:
                r = run_one(pgm["tactic"])
                r.update({"tactic": pgm["tactic"], "family": pgm.get("family"),
                          "depth": pgm.get("depth"), "rank": pgm.get("rank"),
                          "ranker_score": pgm.get("ranker_score"),
                          "lemmas": pgm.get("used_lemmas", []),
                          "candidate_family_tags": pgm.get("candidate_family_tags", [])})
                res["ran"].append(r)
                if r["solved"]:
                    break          # stop after first success
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
        res["setup_error"] = f"{type(e).__name__}: {str(e)[:200]}\n" + traceback.format_exc()[-300:]
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


# ------------------------------- driver -----------------------------------
def run_budget(theorems, budget, ckpt_path, args, rank_lo=1, run_controls=True,
               only_unsolved=None):
    """Shared driver loop. rank_lo..budget window per theorem; only_unsolved = set of
    full_names to run (others skipped)."""
    ckpt = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}
    results = []
    for t in theorems:
        fn = t["full_name"]
        if fn in ckpt:
            results.append(ckpt[fn]); continue
        if only_unsolved is not None and fn not in only_unsolved:
            continue
        progs = [p for p in t.get("programs_ranked", []) if rank_lo <= p["rank"] <= budget]
        if not progs:
            rec = {"full_name": fn, "namespace": t.get("namespace"), "budget": budget,
                   "rank_window": [rank_lo, budget], "live": None,
                   "programs_attempted": 0, "first_success_rank": None, "success": False,
                   "winning_program": None, "failures": [], "open_flake": False,
                   "timeout": False, "setup_error": "no programs in window"}
            results.append(rec); ckpt[fn] = rec
            json.dump(ckpt, open(ckpt_path, "w"), ensure_ascii=False, indent=2)
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            wout = tf.name
        cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout),
               sys.executable, os.path.abspath(__file__), "--worker",
               "--worker-out", wout,
               "--case-json", json.dumps({"full_name": fn, "file_path": t.get("file_path") or _file_for(fn)}),
               "--programs-json", json.dumps(progs),
               "--run-controls", "true" if run_controls else "false",
               "--open-timeout", str(args.open_timeout),
               "--timeout-per-tactic", str(args.timeout_per_tactic)]
        print(f"[tr5-live] B{budget} {fn}: ranks {rank_lo}-{budget} ({len(progs)} programs) ...", flush=True)
        rc = subprocess.run(cmd, capture_output=True, text=True).returncode
        try:
            wres = json.load(open(wout))
        except Exception:
            wres = {"live": False, "setup_error": f"worker no output (rc={rc})",
                    "controls": [], "ran": []}
        finally:
            try:
                os.unlink(wout)
            except OSError:
                pass
        ran = wres.get("ran", [])
        controls = wres.get("controls", [])
        win = next((r for r in ran if r.get("solved")), None)
        timed = any(r.get("outcome") == "timeout" for r in ran)
        rec = {
            "full_name": fn, "namespace": t.get("namespace"), "budget": budget,
            "rank_window": [rank_lo, budget],
            "live": bool(wres.get("live")), "setup_error": wres.get("setup_error"),
            "programs_attempted": len(ran),
            "first_success_rank": (win.get("rank") if win else None),
            "success": bool(win),
            "winning_program": ({"rank": win["rank"], "tactic": win["tactic"],
                                 "family": win.get("family"), "depth": win.get("depth"),
                                 "used_lemmas": win.get("lemmas", []),
                                 "ranker_score": win.get("ranker_score"),
                                 "candidate_family_tags": win.get("candidate_family_tags", [])}
                                if win else None),
            "controls": controls,
            "control_wins": [c["tactic"] for c in controls if c.get("solved")],
            "failures": [{"rank": r.get("rank"), "tactic": r["tactic"],
                          "outcome": r["outcome"], "family": r.get("family")}
                         for r in ran if not r.get("solved")],
            "open_flake": bool(wres.get("setup_error")) and "exceeded" in (wres.get("setup_error") or ""),
            "timeout": timed,
        }
        results.append(rec)
        ckpt[fn] = rec
        json.dump(ckpt, open(ckpt_path, "w"), ensure_ascii=False, indent=2)
    return results


_FILE_CACHE = {}


def _file_for(fn):
    return _FILE_CACHE.get(fn)


def driver(args):
    plan = json.load(open(_p(args.ranked_plan)))
    theorems = plan["theorems"]
    # confirmed failures first, then the rest (controls/known-winners), deterministic
    theorems.sort(key=lambda t: (0 if t.get("rc2_status") == "CONFIRMED_RC2_FAILURE" else 1,
                                 t["full_name"]))
    for t in theorems:
        if t.get("file_path"):
            _FILE_CACHE[t["full_name"]] = t["file_path"]
    # backfill file_path from the target pool if missing
    if args.target_pool and os.path.exists(_p(args.target_pool)):
        for l in open(_p(args.target_pool)):
            if l.strip():
                r = json.loads(l)
                _FILE_CACHE.setdefault(r["full_name"], r.get("file_path"))
        for t in theorems:
            t.setdefault("file_path", _FILE_CACHE.get(t["full_name"]))

    ckpt_path = _p(args.checkpoint)
    results = run_budget(theorems, args.budget, ckpt_path, args,
                         rank_lo=1, run_controls=True)

    n_live = sum(1 for r in results if r["live"])
    n_win = sum(1 for r in results if r["success"])
    n_setup = sum(1 for r in results if r["setup_error"])
    rank_hist = Counter(r["first_success_rank"] for r in results if r["success"])
    out = {
        "generated_by": "scripts/tr5_run_ranked_live_search.py",
        "budget": args.budget, "ranked_plan": args.ranked_plan,
        "num_theorems": len(results), "num_live": n_live,
        "num_success": n_win, "num_setup_error": n_setup,
        "first_success_rank_histogram": {str(k): v for k, v in sorted(rank_hist.items(), key=lambda x: (x[0] is None, x[0]))},
        "results": results,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR5 B%d live results" % args.budget, "",
          f"- theorems: {len(results)} | live: {n_live} | successes: **{n_win}** | "
          f"setup errors: {n_setup}",
          f"- first-success rank histogram: {dict(rank_hist)}", "",
          "| theorem | ns | attempted | success | first_rank | winning tactic |",
          "|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda x: (not x["success"], x["full_name"])):
        wt = r["winning_program"]["tactic"][:50] if r["winning_program"] else ""
        md.append(f"| `{r['full_name']}` | {r['namespace']} | {r['programs_attempted']} | "
                  f"{r['success']} | {r['first_success_rank']} | `{wt}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr5-live] B{args.budget} done: live={n_live}/{len(results)} successes={n_win} "
          f"setup_err={n_setup} rank_hist={dict(rank_hist)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--case-json")
    ap.add_argument("--programs-json")
    ap.add_argument("--run-controls", default="true")
    ap.add_argument("--ranked-plan")
    ap.add_argument("--target-pool",
                    default="project/evolve/experiments/tr5/cases/tr5_target_pool.jsonl")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--checkpoint",
                    default="project/evolve/experiments/tr5/out/b5_live_checkpoint.json")
    ap.add_argument("--open-timeout", type=int, default=90)
    ap.add_argument("--timeout-per-tactic", type=int, default=12)
    ap.add_argument("--hard-timeout", type=int, default=1200)
    args = ap.parse_args()
    if args.worker:
        sys.exit(worker(args))
    driver(args)


if __name__ == "__main__":
    main()
