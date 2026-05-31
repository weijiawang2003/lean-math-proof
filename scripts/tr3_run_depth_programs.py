#!/usr/bin/env python3
"""TR3 Part 7 — run live retrieval-aware depth programs.

Driver/worker LeanDojo harness (serialized; one Dojo per theorem under an OS hard
timeout via run_with_timeout.py). For each confirmed RC2 failure the worker runs the
4 bare controls (simp / simp_all / aesop / classical <;> aesop) then the gated
programs in plan order, per-tactic SIGALRM-bounded. Per-theorem incremental
checkpoint enables resume; --stop-after-win stops a theorem after its first solving
program but records the remaining programs as skipped.

No win is final here — TR3 SX4-style attribution (Part 8) re-judges every win against
literal RC2. Production configs untouched.
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
    programs = json.loads(args.programs_json)
    stop_after_win = args.stop_after_win == "true"
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

            # controls
            for c in CONTROLS:
                r = run_one(c)
                r["tactic"] = c
                res["controls"].append(r)
                if r["dead"]:
                    raise RuntimeError("dojo died during controls")
            # programs
            won = False
            for pgm in programs:
                if won and stop_after_win:
                    res["ran"].append({"tactic": pgm["tactic"], "family": pgm["family"],
                                       "depth": pgm["depth"], "lemmas": pgm.get("lemmas", []),
                                       "outcome": "skipped", "solved": False, "skipped": True})
                    continue
                r = run_one(pgm["tactic"])
                r.update({"tactic": pgm["tactic"], "family": pgm["family"],
                          "depth": pgm["depth"], "lemmas": pgm.get("lemmas", []),
                          "risk": pgm.get("risk")})
                res["ran"].append(r)
                if r["solved"]:
                    won = True
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
def driver(args):
    plan = json.load(open(_p(args.program_plan)))
    theorems = plan["theorems"]
    ckpt_path = _p(args.checkpoint)
    ckpt = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}

    results = []
    win_hist = {"candidate_win": 0, "no_win": 0, "baseline_duplicate": 0, "needs_review": 0}
    n_live = 0
    for t in theorems:
        fn = t["full_name"]
        fp = t.get("file_path")
        programs = t.get("programs", [])
        if fn in ckpt:
            rec = ckpt[fn]
            results.append(rec)
            if rec.get("live"):
                n_live += 1
            win_hist[rec.get("classification_pre_attribution", "needs_review")] = \
                win_hist.get(rec.get("classification_pre_attribution", "needs_review"), 0) + 1
            continue

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            wout = tf.name
        cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout),
               sys.executable, os.path.abspath(__file__), "--worker",
               "--worker-out", wout,
               "--case-json", json.dumps({"full_name": fn, "file_path": fp}),
               "--programs-json", json.dumps(programs),
               "--open-timeout", str(args.open_timeout),
               "--timeout-per-tactic", str(args.timeout_per_tactic),
               "--stop-after-win", args.stop_after_win]
        print(f"[tr3-depth] {fn}: {len(programs)} programs ...", flush=True)
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
        wins = [r for r in ran if r.get("solved")]
        control_wins = [c for c in controls if c.get("solved")]
        best = None
        if wins:
            # headline: prefer a retrieval/def-unfold win, else first
            pref = [w for w in wins if w.get("lemmas")] or \
                   [w for w in wins if w.get("family") == "def_unfold_simp"]
            best = (pref[0] if pref else wins[0])
        if wres.get("setup_error"):
            cls = "needs_review"
        elif control_wins:
            cls = "baseline_duplicate"
        elif wins:
            cls = "candidate_win"
        else:
            cls = "no_win"

        hist = {}
        for r in ran:
            hist[r["outcome"]] = hist.get(r["outcome"], 0) + 1

        rec = {
            "full_name": fn, "file_path": fp, "namespace": t.get("namespace"),
            "cluster_id": t.get("cluster_id"), "rc2_confirmed_failure": True,
            "live": bool(wres.get("live")), "setup_error": wres.get("setup_error"),
            "programs_tried": sum(1 for r in ran if not r.get("skipped")),
            "programs_total": len(programs),
            "controls": controls, "control_wins": [c["tactic"] for c in control_wins],
            "wins": [{"tactic": w["tactic"], "family": w["family"], "depth": w["depth"],
                      "lemmas": w.get("lemmas", [])} for w in wins],
            "best_win": ({"tactic": best["tactic"], "family": best["family"],
                          "depth": best["depth"], "lemmas": best.get("lemmas", [])}
                         if best else None),
            "outcome_histogram": hist, "ran": ran,
            "classification_pre_attribution": cls,
        }
        if rec["live"]:
            n_live += 1
        win_hist[cls] = win_hist.get(cls, 0) + 1
        results.append(rec)
        ckpt[fn] = rec
        json.dump(ckpt, open(ckpt_path, "w"), ensure_ascii=False, indent=2)

    out = {
        "generated_by": "scripts/tr3_run_depth_programs.py",
        "program_plan_input": args.program_plan,
        "stop_after_win": args.stop_after_win == "true",
        "num_theorems": len(results), "num_live": n_live,
        "classification_histogram": win_hist, "results": results,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# TR3 depth-program results", "",
          f"- theorems: {len(results)} | live: {n_live}",
          f"- pre-attribution: {win_hist}", "",
          "| target | live | progs | wins | control_wins | best |", "|---|---|---|---|---|---|"]
    for r in results:
        bw = r["best_win"]["tactic"] if r["best_win"] else ""
        md.append(f"| `{r['full_name']}` | {r['live']} | {r['programs_tried']}/"
                  f"{r['programs_total']} | {len(r['wins'])} | {len(r['control_wins'])} | "
                  f"`{bw}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr3-depth] done: live={n_live}/{len(results)} {win_hist}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--case-json")
    ap.add_argument("--programs-json")
    ap.add_argument("--program-plan")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--checkpoint",
                    default="project/evolve/experiments/tr3/out/depth_program_checkpoint.json")
    ap.add_argument("--stop-after-win", default="true")
    ap.add_argument("--open-timeout", type=int, default=90)
    ap.add_argument("--timeout-per-tactic", type=int, default=12)
    ap.add_argument("--hard-timeout", type=int, default=1200)
    args = ap.parse_args()
    if args.worker:
        sys.exit(worker(args))
    driver(args)


if __name__ == "__main__":
    main()
