#!/usr/bin/env python3
"""SX2 Parts 3/4/5 — live LeanDojo evaluator for the SET2 candidate policy.

Mirrors the proven SF2 driver/worker harness: DRIVER spawns ONE worker subprocess
per theorem under an OS hard timeout (scripts/run_with_timeout.py); WORKER opens
ONE Dojo session, then:

  1. runs the BASELINE battery (simp, simp_all, aesop, classical <;> aesop) from the
     initial state. This is the RC1-proxy for these Set/Basic goals: RC1 = NS9 base
     + WX3 Multiset-induction + MX2 narrow Set.Finite-aesop; neither WX3 nor MX2
     applies to plain Set.* goals here, so RC1's reach on this surface == its
     NS9/baseline battery. (The 12 selected cases are already known RC1-failed.)
  2. evaluates the SET2 gate policy (force_enable=True; production default is OFF)
     over (full_name, goal_pp) and tries each EMITTED gated tactic in gate order,
     stopping at the first solve.

Records per theorem: rc1_solved (proxy), baseline_outcomes, set2_emitted, the
emitting gate/tactic, set2_solved, parse_error/proof_failed/timeout flags, off_gate.
NEVER modifies RC1 / NS24 / NS9.

Outputs (paths via --out-json/--out-md):
  set2_*_eval_results.json
  set2_*_eval_results.md
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sx2_set2_wrapper as setw  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")

BASELINES = ["simp", "simp_all", "aesop", "classical <;> aesop"]


class _ProbeTimeout(Exception):
    pass


def _alarm(_s, _f):
    raise _ProbeTimeout()


def classify_outcome(err, solved):
    if solved:
        return "solved"
    e = (err or "").lower()
    if not e:
        return "proof_failed"
    if ("expected end of input" in e or "expected '{' or tactic" in e
            or "unexpected token" in e or "unexpected identifier" in e):
        return "parse_error"
    if "maximum recursion depth" in e or "maxrecdepth" in e:
        return "max_recursion"
    if "applyexttheorem only applies" in e:
        return "ext_not_applicable"
    if "no goals" in e:
        return "no_goals"
    if "unknown" in e and ("identifier" in e or "constant" in e):
        return "unknown_ident"
    return "proof_failed"


def _load_cases(path):
    """Accept either {"selected":[...]} (SF2) or {"cases":[...]} (holdout)."""
    obj = json.load(open(path))
    if isinstance(obj, dict):
        for k in ("selected", "cases", "theorems"):
            if k in obj:
                return obj[k]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"unrecognized cases schema in {path}")


# ----------------------------- worker -------------------------------------
def worker(args):
    cases = _load_cases(args.cases)
    policy = setw.load_policy(args.gate_policy)
    case = cases[args.worker_theorem]
    res = {"full_name": case["full_name"], "file_path": case["file_path"],
           "cluster_id": case.get("cluster_id"),
           "primary_goal_shape": case.get("primary_goal_shape"),
           "live": False, "initial_goal": None,
           "rc1_solved": None, "rc1_proxy": "baseline_battery",
           "baseline_outcomes": [], "baseline_solved_by": None,
           "set2_emitted": False, "set2_gate": None, "set2_tactic": None,
           "set2_solved": False, "set2_emissions": [],
           "parse_error": False, "proof_failed": False, "timeout": False,
           "off_gate": False, "notes": [], "setup_error": None}
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
        with _Dojo(thm) as (dojo, state0):
            res["live"] = True
            goal = getattr(state0, "pp", None) or getattr(state0, "state", None)
            res["initial_goal"] = goal

            def apply(tac):
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(args.timeout_per_probe)
                try:
                    out = _env.run_transition(dojo, thm, state0, tac)
                finally:
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(0)
                rec = getattr(out, "record", None)
                return (bool(getattr(out, "is_finished", False)),
                        bool(getattr(out, "session_dead", False)),
                        getattr(rec, "error_message", None) if rec else None)

            # ---- 1. baseline (RC1-proxy) battery ----
            for b in BASELINES:
                try:
                    fin, dead, err = apply(b)
                except _ProbeTimeout:
                    res["baseline_outcomes"].append({"probe": b, "solved": False,
                                                      "outcome": "timeout_inner"})
                    continue
                except Exception as e:
                    res["baseline_outcomes"].append({"probe": b, "solved": False,
                                                      "outcome": "exception",
                                                      "error": str(e)[:120]})
                    continue
                oc = classify_outcome(err, fin)
                rec = {"probe": b, "solved": bool(fin), "outcome": oc}
                if err and not fin:
                    rec["error"] = err[:160]
                res["baseline_outcomes"].append(rec)
                if fin and res["baseline_solved_by"] is None:
                    res["baseline_solved_by"] = b
                if dead:
                    res["notes"].append("session_dead during baselines")
                    break
            res["rc1_solved"] = res["baseline_solved_by"] is not None

            # ---- 2. SET2 gated emission (force_enable; prod default OFF) ----
            log = []
            emissions = setw.eval_gates(policy, case["full_name"], goal,
                                        force_enable=True, log=log)
            res["set2_emissions"] = log
            res["set2_emitted"] = bool(emissions)
            res["off_gate"] = any(e.get("off_gate") for e in emissions)
            for e in emissions:
                try:
                    fin, dead, err = apply(e["tactic"])
                except _ProbeTimeout:
                    res["timeout"] = True
                    e["result"] = "timeout_inner"
                    continue
                except Exception as ex:
                    e["result"] = f"exception: {str(ex)[:120]}"
                    continue
                oc = classify_outcome(err, fin)
                e["result"] = oc
                if err and not fin:
                    e["error"] = err[:160]
                if oc == "parse_error":
                    res["parse_error"] = True
                if not fin:
                    res["proof_failed"] = True
                if fin:
                    res["set2_solved"] = True
                    res["set2_gate"] = e["gate_id"]
                    res["set2_tactic"] = e["tactic"]
                    res["proof_failed"] = False
                    break
                if dead:
                    res["notes"].append("session_dead during SET2 emission")
                    break
            if res["set2_emitted"] and not res["set2_solved"]:
                res["notes"].append("SET2 emitted but did not close the goal")
            if not res["set2_emitted"]:
                res["notes"].append("no SET2 gate fired")
    except Exception as e:
        res["setup_error"] = f"{type(e).__name__}: {str(e)[:200]}\n" + \
            traceback.format_exc()[-300:]
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


# ----------------------------- driver -------------------------------------
def _metrics(results):
    total = len(results)
    live = sum(1 for r in results if r.get("live"))
    rc1 = sum(1 for r in results if r.get("rc1_solved"))
    set2 = sum(1 for r in results if r.get("set2_solved"))
    emitted = sum(1 for r in results if r.get("set2_emitted"))
    emit_solved = sum(1 for r in results if r.get("set2_emitted") and r.get("set2_solved"))
    emit_failed = sum(1 for r in results if r.get("set2_emitted") and not r.get("set2_solved"))
    not_emitted = total - emitted
    new_wins = sum(1 for r in results if r.get("set2_solved") and not r.get("rc1_solved"))
    # SET2 is an ADDITIVE external candidate: it is tried ALONGSIDE RC1 (first solver
    # wins) and never removes/alters an RC1 tactic. It therefore cannot regress RC1 by
    # construction -> regressions = 0. We still report the diagnostic count of cases
    # where SET2 fired but failed on an already-RC1-solved theorem (harmless extra try).
    emitted_failed_on_rc1_solved = sum(
        1 for r in results
        if r.get("rc1_solved") and not r.get("set2_solved") and r.get("set2_emitted"))
    off_gate = sum(1 for r in results if r.get("off_gate"))
    return {"total": total, "live": live, "rc1_solved": rc1, "set2_solved": set2,
            "set2_new_wins_over_rc1": new_wins, "set2_regressions": 0,
            "regression_note": "0 by construction — SET2 is additive/off-by-default; "
                               "it never alters RC1 behavior.",
            "set2_emitted_but_failed_on_rc1_solved": emitted_failed_on_rc1_solved,
            "off_gate_emissions": off_gate,
            "gate_precision": {"emitted_and_solved": emit_solved,
                               "emitted_and_failed": emit_failed,
                               "not_emitted": not_emitted}}


def driver(args):
    cases = _load_cases(args.cases)
    results = []

    def checkpoint():
        agg = {"cases_file": args.cases, "gate_policy": args.gate_policy,
               "production_default_emits": False,
               "rc1_proxy_note": "rc1_solved == baseline battery closes goal; valid "
                                 "RC1 proxy for Set/Basic surfaces (WX3/MX2 do not "
                                 "apply). The 12 SF2 selected cases are known "
                                 "RC1-failed.",
               "metrics": _metrics(results), "results": results,
               "constraints": "No solve is promotion-confirmed; NS23 minimal relabel "
                              "required. RC1/NS24/NS9 untouched. SET2 off-by-default."}
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        json.dump(agg, open(args.out_json, "w"), ensure_ascii=False, indent=2)
        return agg

    for idx in range(len(cases)):
        hard = args.timeout_per_probe * (len(BASELINES) + 5) + 90
        wout = f"/tmp/sx2_set2_eval_t{idx}.json"
        if os.path.exists(wout):
            os.remove(wout)
        cmd = [sys.executable, _TIMEOUT_HELPER, str(hard), sys.executable,
               os.path.abspath(__file__), "--worker-theorem", str(idx),
               "--worker-out", wout, "--cases", args.cases,
               "--gate-policy", args.gate_policy,
               "--timeout-per-probe", str(args.timeout_per_probe)]
        print(f"[sx2:eval] ({idx+1}/{len(cases)}) {cases[idx]['full_name']} "
              f"hard={hard}s", flush=True)
        rc = subprocess.run(cmd, capture_output=True, text=True).returncode
        if os.path.exists(wout):
            try:
                rec = json.load(open(wout))
            except Exception as e:
                rec = {"full_name": cases[idx]["full_name"], "live": False,
                       "setup_error": f"unreadable worker out: {e}"}
        else:
            rec = {"full_name": cases[idx]["full_name"], "live": False,
                   "setup_error": f"no worker output (rc={rc}); OS-killed at {hard}s"}
        results.append(rec)
        print(f"            -> live={rec.get('live')} rc1={rec.get('rc1_solved')} "
              f"emitted={rec.get('set2_emitted')} gate={rec.get('set2_gate')} "
              f"set2_solved={rec.get('set2_solved')}", flush=True)
        checkpoint()

    final = checkpoint()
    write_md(final, args.out_md)
    m = final["metrics"]
    print(f"[sx2:eval] DONE total={m['total']} live={m['live']} rc1={m['rc1_solved']} "
          f"set2={m['set2_solved']} new_wins={m['set2_new_wins_over_rc1']} "
          f"off_gate={m['off_gate_emissions']}")
    return 0


def write_md(agg, path):
    m = agg["metrics"]
    L = ["# SX2 — SET2 Live Eval Results", ""]
    L.append(f"- cases: `{agg['cases_file']}`")
    L.append(f"- production default emits: **{agg['production_default_emits']}** "
             "(SET2 off-by-default; forced on for this experiment)")
    L.append(f"- total={m['total']} live={m['live']} | RC1-proxy solved="
             f"{m['rc1_solved']} | SET2 solved={m['set2_solved']} | "
             f"**SET2 new wins over RC1={m['set2_new_wins_over_rc1']}** | "
             f"regressions={m['set2_regressions']} | off-gate={m['off_gate_emissions']}")
    L.append(f"- gate precision: {m['gate_precision']}")
    L.append(f"- {agg['rc1_proxy_note']}")
    L.append("")
    L.append("| theorem | shape | rc1 | emitted | gate | set2_tactic | set2_solved | off_gate |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in agg["results"]:
        L.append(f"| `{r['full_name']}` | {r.get('primary_goal_shape','')} | "
                 f"{r.get('rc1_solved')} | {r.get('set2_emitted')} | "
                 f"{r.get('set2_gate')} | `{(r.get('set2_tactic') or '')}` | "
                 f"{r.get('set2_solved')} | {r.get('off_gate')} |")
    L.append("")
    for r in agg["results"]:
        L.append(f"## `{r['full_name']}`")
        if r.get("setup_error"):
            L.append(f"- setup_error: {r['setup_error'][:200]}")
        if r.get("initial_goal"):
            L.append(f"- goal: `{r['initial_goal'][:180]}`")
        L.append(f"- rc1(proxy)={r.get('rc1_solved')} (by `{r.get('baseline_solved_by')}`) "
                 f"| set2_solved={r.get('set2_solved')} via gate {r.get('set2_gate')}")
        gates = [f"{e['gate_id']}:{e.get('result','-')}" for e in r.get("set2_emissions", [])
                 if e.get("emitted")]
        L.append(f"- emitted gates → results: {gates}")
        if r.get("notes"):
            L.append(f"- notes: {r['notes']}")
        L.append("")
    L.append("> " + agg["constraints"])
    open(path, "w").write("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True)
    p.add_argument("--gate-policy",
                   default="project/evolve/experiments/sx2/set2_gate_policy.json")
    # not required: worker subprocesses don't pass these (only the driver writes).
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-md", default=None)
    p.add_argument("--timeout-per-probe", type=int, default=30)
    p.add_argument("--worker-theorem", type=int, default=None)
    p.add_argument("--worker-out", default=None)
    args = p.parse_args(argv)
    if args.worker_theorem is not None:
        return worker(args)
    return driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
