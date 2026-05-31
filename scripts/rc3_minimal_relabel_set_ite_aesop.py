#!/usr/bin/env python3
"""RC3 minimal-sufficient attribution for SX3_SET_ITE_AESOP.

For every RC3 new win over LITERAL RC2 (RC3 finished & RC2 not finished), open a
fresh Dojo and run the control battery single-shot from state0:

    simp
    simp_all
    aesop
    classical <;> aesop
    simp [Set.ite]                 (RC2's credited single-shot mechanism)
    simp [Set.ite] <;> aesop       (the SX3 depth-2 sequence)

Classification (per RC3-new-win theorem):

    TRUE_SX3_SET_ITE_AESOP_WIN  RC2 failed; one-step controls all failed;
                                depth-2 sequence solved; gate valid (Set.ite, non-forbidden ns)
    SINGLE_STEP_DUPLICATE       single-shot 'simp [Set.ite]' solves -> belongs to RC2 SET_ITE_SIMP
    BASELINE_DUPLICATE          a bare baseline (simp/simp_all/aesop/classical) solves
    SOURCE_SPECIFIC             (structurally 0: generic battery, no theorem-specific rw)
    PARSER_ARTIFACT             depth-2 only failed via parse/recursion limits (prior runner syntax)
    OPEN_FLAKE                  Dojo could not be opened reliably
    NEEDS_REVIEW                anything else (e.g. RC3 won in-harness but no standalone control reproduces)

RC2_ALREADY_SOLVED rows are excluded up front (not new wins).

Driver/worker model mirrors scripts/sx3_run_depth2_sequences.py: the driver
spawns one worker subprocess per theorem under an OS-level hard timeout
(scripts/run_with_timeout.py); the worker opens ONE Dojo, bounds the open with
SIGALRM, runs the controls with a per-tactic SIGALRM, and writes a JSON. No
production config is read or modified.
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

ONE_STEP_CONTROLS = ["simp", "simp_all", "aesop", "classical <;> aesop", "simp [Set.ite]"]
SINGLE_STEP_SET_ITE = "simp [Set.ite]"
BASELINE_CONTROLS = {"simp", "simp_all", "aesop", "classical <;> aesop"}
DEPTH2_SEQUENCE = "simp [Set.ite] <;> aesop"
FORBID_NS = {"Nat", "Int", "Multiset", "List"}
PARSE_OUTCOMES = {"parse_error", "max_recursion", "timeout_inner", "exception", "no_goals"}


class _ProbeTimeout(Exception):
    pass


def _alarm(_s, _f):
    raise _ProbeTimeout()


def _classify_outcome(err, solved):
    if solved:
        return "solved"
    e = (err or "").lower()
    if not e:
        return "proof_failed"
    if ("expected end of input" in e or "expected '{' or tactic" in e
            or "unexpected token" in e or "unexpected identifier" in e
            or "expected term" in e):
        return "parse_error"
    if "maximum recursion depth" in e or "maxrecdepth" in e:
        return "max_recursion"
    if "no goals" in e:
        return "no_goals"
    return "proof_failed"


def _gate_valid(full_name):
    ns = full_name.split(".")[0] if "." in full_name else ""
    if ns in FORBID_NS:
        return False, f"forbidden_ns:{ns}"
    if "Set.ite" not in full_name:
        return False, "name_lacks_Set.ite"
    return True, "ok"


# ------------------------------- worker -----------------------------------
def worker(args):
    case = json.loads(args.case_json)
    res = {"full_name": case["full_name"], "file_path": case.get("file_path"),
           "live": False, "controls": [], "setup_error": None, "classification": None}
    try:
        sys.path.insert(0, _REPO)
        import env as _env
        from core_types import TheoremConfig as _TC
        from lean_dojo import Dojo as _Dojo
        repo = _env.make_repo()
        thm = _env.make_theorem(repo, _TC(file_path=case["file_path"],
                                          full_name=case["full_name"]))
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
                    return {"tactic": tac, "solved": False, "outcome": "timeout_inner", "dead": False}
                except Exception as e:
                    return {"tactic": tac, "solved": False, "outcome": "exception",
                            "error": f"{type(e).__name__}: {str(e)[:160]}", "dead": False}
                finally:
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(0)
                r = {"tactic": tac, "solved": bool(fin),
                     "outcome": _classify_outcome(err, fin), "dead": bool(dead)}
                if err and not fin:
                    r["error"] = err[:200]
                return r

            for tac in ONE_STEP_CONTROLS + [DEPTH2_SEQUENCE]:
                r = run_one(tac)
                res["controls"].append(r)
                if r["dead"]:
                    res.setdefault("notes", []).append(f"session_dead after {tac}")
                    break
        finally:
            try:
                dojo_cm.__exit__(None, None, None)
            except Exception:
                pass
    except _ProbeTimeout:
        res["setup_error"] = f"dojo open exceeded {args.open_timeout}s"
        res["classification"] = "OPEN_FLAKE"
    except Exception as e:
        res["setup_error"] = f"{type(e).__name__}: {str(e)[:200]}\n" + traceback.format_exc()[-300:]
        res["classification"] = "NEEDS_REVIEW"
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


def _classify(full_name, worker_res, rc3_winning_tactic):
    if worker_res.get("classification") in ("OPEN_FLAKE", "NEEDS_REVIEW") and not worker_res.get("live"):
        return worker_res["classification"], worker_res.get("setup_error")
    solved = {c["tactic"] for c in worker_res.get("controls", []) if c.get("solved")}
    outcomes = {c["tactic"]: c["outcome"] for c in worker_res.get("controls", [])}
    gate_ok, gate_reason = _gate_valid(full_name)
    # single-shot Set.ite belongs to RC2
    if SINGLE_STEP_SET_ITE in solved:
        return "SINGLE_STEP_DUPLICATE", f"single-shot '{SINGLE_STEP_SET_ITE}' solves (RC2 mechanism)"
    if solved & BASELINE_CONTROLS:
        return "BASELINE_DUPLICATE", f"bare baseline solves: {sorted(solved & BASELINE_CONTROLS)}"
    if DEPTH2_SEQUENCE in solved:
        if not gate_ok:
            return "NEEDS_REVIEW", f"depth-2 solved but gate invalid: {gate_reason}"
        return "TRUE_SX3_SET_ITE_AESOP_WIN", "depth-2 solved; all one-step controls failed; gate valid"
    # depth-2 did not solve standalone
    if outcomes.get(DEPTH2_SEQUENCE) in PARSE_OUTCOMES:
        return "PARSER_ARTIFACT", f"depth-2 standalone outcome={outcomes.get(DEPTH2_SEQUENCE)}"
    return "NEEDS_REVIEW", "RC3 won in-harness but no standalone control reproduced the close"


def driver(args):
    rc2 = json.load(open(args.rc2))
    rc3 = json.load(open(args.rc3))
    rc2_fin = {r["full_name"]: r for r in rc2["per_theorem"]}
    rc3_fin = {r["full_name"]: r for r in rc3["per_theorem"]}

    new_wins, excluded = [], []
    for fn, r3 in rc3_fin.items():
        r2 = rc2_fin.get(fn)
        if not r3.get("finished"):
            continue
        if r2 and r2.get("finished"):
            excluded.append({"full_name": fn, "reason": "RC2_ALREADY_SOLVED"})
            continue
        new_wins.append({"full_name": fn, "file_path": r3.get("file_path"),
                         "rc3_winning_tactic": r3.get("winning_tactic"),
                         "role": r3.get("role")})

    results = []
    for w in new_wins:
        if not w.get("file_path"):
            results.append({**w, "classification": "NEEDS_REVIEW",
                            "reason": "no file_path; not live-resolvable", "controls": []})
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            wout = tf.name
        cmd = ["/opt/anaconda3/bin/python3", _TIMEOUT_HELPER, str(args.hard_timeout),
               "/opt/anaconda3/bin/python3", os.path.abspath(__file__),
               "--worker", "--worker-out", wout,
               "--case-json", json.dumps({"full_name": w["full_name"], "file_path": w["file_path"]}),
               "--open-timeout", str(args.open_timeout),
               "--timeout-per-tactic", str(args.timeout_per_tactic)]
        print(f"[relabel] probing {w['full_name']} ...", flush=True)
        rc = subprocess.run(cmd, capture_output=True, text=True).returncode
        try:
            wres = json.load(open(wout))
        except Exception:
            wres = {"live": False, "classification": "OPEN_FLAKE",
                    "setup_error": f"worker produced no output (rc={rc}, likely OS timeout)"}
        finally:
            try: os.unlink(wout)
            except OSError: pass
        if rc == 124 and not wres.get("live"):
            wres["classification"] = "OPEN_FLAKE"
            wres["setup_error"] = f"OS hard timeout ({args.hard_timeout}s) before live"
        cls, reason = _classify(w["full_name"], wres, w["rc3_winning_tactic"])
        results.append({**w, "classification": cls, "reason": reason,
                        "live": wres.get("live"), "setup_error": wres.get("setup_error"),
                        "controls": wres.get("controls", [])})
        print(f"[relabel]   -> {cls}", flush=True)

    hist = {}
    for r in results:
        hist[r["classification"]] = hist.get(r["classification"], 0) + 1
    true_wins = [r["full_name"] for r in results if r["classification"] == "TRUE_SX3_SET_ITE_AESOP_WIN"]
    out = {
        "inputs": {"rc2": args.rc2, "rc3": args.rc3},
        "num_rc3_finished": sum(1 for r in rc3_fin.values() if r.get("finished")),
        "num_rc2_finished": sum(1 for r in rc2_fin.values() if r.get("finished")),
        "num_new_wins_over_rc2": len(new_wins),
        "num_rc2_already_solved_excluded": len(excluded),
        "classification_histogram": hist,
        "true_sx3_set_ite_aesop_wins": sorted(true_wins),
        "num_true_wins": len(true_wins),
        "excluded_rc2_already_solved": excluded,
        "per_win": results,
        "promotion_note": "Only TRUE_SX3_SET_ITE_AESOP_WIN counts toward the credited SX3 delta.",
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)
    _write_md(out, args.out_md)
    print(f"[relabel] wrote {args.out_json}: TRUE wins={len(true_wins)} {sorted(true_wins)}")
    return 0


def _write_md(out, path):
    L = ["# RC3 minimal-sufficient attribution — SX3_SET_ITE_AESOP", "",
         f"- RC2 finished (validation surface): **{out['num_rc2_finished']}**",
         f"- RC3 finished (validation surface): **{out['num_rc3_finished']}**",
         f"- RC3 new wins over literal RC2: **{out['num_new_wins_over_rc2']}**",
         f"- RC2-already-solved (excluded): {out['num_rc2_already_solved_excluded']}",
         f"- **TRUE_SX3_SET_ITE_AESOP_WIN: {out['num_true_wins']}** → {out['true_sx3_set_ite_aesop_wins']}",
         "", "## Classification histogram", ""]
    for k, v in sorted(out["classification_histogram"].items()):
        L.append(f"- `{k}`: {v}")
    L += ["", "## Per new-win", "",
          "| theorem | role | classification | reason |", "|---|---|---|---|"]
    for r in out["per_win"]:
        L.append(f"| `{r['full_name']}` | {r.get('role')} | **{r['classification']}** | {r.get('reason','')} |")
    L += ["", "## Control battery per win", ""]
    for r in out["per_win"]:
        if not r.get("controls"):
            continue
        L.append(f"### `{r['full_name']}` ({r['classification']})")
        L.append("| tactic | solved | outcome |")
        L.append("|---|---|---|")
        for c in r["controls"]:
            L.append(f"| `{c['tactic']}` | {'✅' if c.get('solved') else '—'} | {c.get('outcome')} |")
        L.append("")
    open(path, "w").write("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--worker-out")
    p.add_argument("--case-json")
    p.add_argument("--rc2")
    p.add_argument("--rc3")
    p.add_argument("--out-json")
    p.add_argument("--out-md")
    p.add_argument("--hard-timeout", type=int, default=200)
    p.add_argument("--open-timeout", type=int, default=90)
    p.add_argument("--timeout-per-tactic", type=int, default=60)
    args = p.parse_args(argv)
    if args.worker:
        return worker(args)
    return driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
