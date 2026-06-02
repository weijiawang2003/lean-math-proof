#!/usr/bin/env python3
"""FLI2 Part 5 — live deployment evaluation in LeanDojo (one Dojo per theorem, at position).

For each theorem: open the Dojo (real file position → target theorem & downstream out of scope),
capture the initial goal, run controls, then every candidate action from the initial state,
recording solved / residual_after / errors. Rescue candidates (solved + all controls fail,
non-vacuous) are re-run once for robustness. Process-group hard timeout + per-tactic SIGALRM +
per-theorem checkpoint/resume.
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


def _classify_err(err, finished):
    if finished:
        return "solved", None
    el = (err or "").lower()
    if not err:
        return "failed", None
    if "unknown identifier" in el or "unknown constant" in el:
        return "unknown_name", err[:160]
    if "type mismatch" in el or "function expected" in el or "failed to synthesize" in el:
        return "type_error", err[:160]
    return "failed", err[:160]


def worker(case, open_timeout, per_tactic):
    import signal
    out = {"theorem": case["theorem"], "file_path": case["file_path"],
           "initial_goal": None, "controls": [], "actions": [], "setup_error": None}

    class _PT(Exception):
        pass

    def _alarm(s, f):
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
            dojo, st0 = cm.__enter__()
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        out["initial_goal"] = getattr(st0, "pp", None)

        def run(tac):
            if hasattr(signal, "SIGALRM"):
                signal.alarm(per_tactic)
            try:
                o = _env.run_transition(dojo, thm, st0, tac)
            except _PT:
                return {"solved": False, "status": "timeout", "residual_after": None, "error": None,
                        "dead": False}
            except Exception as e:  # noqa: BLE001
                return {"solved": False, "status": "needs_review", "residual_after": None,
                        "error": f"{type(e).__name__}: {str(e)[:120]}", "dead": False}
            finally:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)
            if getattr(o, "session_dead", False):
                return {"solved": False, "status": "infra_error", "residual_after": None,
                        "error": "session_dead", "dead": True}
            fin = bool(getattr(o, "is_finished", False))
            rec = getattr(o, "record", None)
            err = getattr(rec, "error_message", None) if rec else None
            ns = getattr(o, "next_state", None)
            status, emsg = _classify_err(err, fin)
            return {"solved": fin, "status": status,
                    "residual_after": getattr(ns, "pp", None) if ns is not None else None,
                    "num_goals_after": getattr(ns, "num_goals", None) if ns is not None else None,
                    "error": emsg, "dead": False}

        dead = False
        for ctrl in case["controls"]:
            r = run(ctrl)
            out["controls"].append({"tactic": ctrl, **r})
            if r.get("dead"):
                dead = True
                break
        if not dead:
            for a in case["actions"]:
                r = run(a["tactic"])
                out["actions"].append({**a, **r})
                if r.get("dead"):
                    break
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass
    except _PT:
        out["setup_error"] = f"dojo open exceeded {open_timeout}s"
    except Exception as e:  # noqa: BLE001
        out["setup_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    return out


def _run_worker(case, hard, open_to, per_tac, ckpt):
    if os.path.exists(ckpt):
        return json.load(open(ckpt))
    wout = ckpt + ".tmp"
    cmd = [sys.executable, _TIMEOUT_HELPER, str(hard), sys.executable, os.path.abspath(__file__),
           "--worker", "--worker-out", wout, "--case-json", json.dumps(case),
           "--open-timeout", str(open_to), "--timeout-per-tactic", str(per_tac)]
    rc = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(wout):
        os.replace(wout, ckpt)
        return json.load(open(ckpt))
    r = {"theorem": case["theorem"], "setup_error": ("timeout" if rc.returncode == 124
         else (rc.stderr or "")[-160:]), "controls": [], "actions": []}
    json.dump(r, open(ckpt, "w"))
    return r


def driver(args):
    plan = json.load(open(_p(args.plan)))
    trace_dir = _p(args.trace_dir)
    os.makedirs(trace_dir, exist_ok=True)
    results = []
    for t in plan["theorems"]:
        thm = t["theorem"]
        ckpt = os.path.join(trace_dir, thm.replace("/", "_").replace(".", "_") + ".json")
        case = {"theorem": thm, "file_path": t["file_path"], "controls": t["controls"],
                "actions": t["actions"]}
        print(f"[fli2-eval] {thm} ({len(t['actions'])} actions) ...", flush=True)
        w = _run_worker(case, t.get("timeout_per_theorem", 240), args.open_timeout,
                        t.get("timeout_per_tactic", 20), ckpt)
        ctrl_solved = [c["tactic"] for c in w.get("controls", []) if c.get("solved")]
        for a in w.get("actions", []):
            is_resc = bool(a.get("solved")) and not ctrl_solved and a.get("lemma") != thm
            rec = {"action_id": a.get("action_id"), "case_id": a.get("case_id"), "theorem": thm,
                   "namespace": t["namespace"], "lemma": a.get("lemma"), "template": a.get("template"),
                   "tactic": a.get("tactic"), "status": a.get("status"), "solved": bool(a.get("solved")),
                   "control_status": {c["tactic"]: bool(c.get("solved")) for c in w.get("controls", [])},
                   "control_solved": ctrl_solved,
                   "is_rescue_candidate": is_resc,
                   "vacuous_self": a.get("lemma") == thm,
                   "residual_before": w.get("initial_goal"),
                   "residual_after": a.get("residual_after"),
                   "num_goals_after": a.get("num_goals_after"),
                   "error_message": a.get("error"), "setup_error": w.get("setup_error"),
                   "trace_path": os.path.relpath(ckpt, _REPO), "robust": None}
            results.append(rec)
        if w.get("setup_error") and not w.get("actions"):
            print(f"  setup_error: {w['setup_error'][:80]}", flush=True)

    # robustness pass: re-run each rescue candidate once (fresh worker, controls + that action)
    rob_dir = os.path.join(trace_dir, "robust")
    os.makedirs(rob_dir, exist_ok=True)
    for rec in results:
        if not rec["is_rescue_candidate"]:
            continue
        thm = rec["theorem"]
        fp = next((t["file_path"] for t in plan["theorems"] if t["theorem"] == thm), None)
        ck = os.path.join(rob_dir, rec["action_id"] + ".json")
        case = {"theorem": thm, "file_path": fp, "controls": plan["controls"],
                "actions": [{"action_id": rec["action_id"], "tactic": rec["tactic"],
                             "lemma": rec["lemma"], "template": rec["template"]}]}
        w = _run_worker(case, 240, args.open_timeout, 20, ck)
        cs = [c["tactic"] for c in w.get("controls", []) if c.get("solved")]
        again = any(a.get("solved") for a in w.get("actions", []))
        rec["robust"] = bool(again and not cs)
        print(f"[fli2-eval] robust {rec['action_id']} {thm}: {rec['robust']}", flush=True)

    with open(_p(args.out_jsonl), "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    solved = [r for r in results if r["solved"]]
    rescues = [r for r in results if r["is_rescue_candidate"]]
    summary = {"generated_by": "scripts/fli2_run_live_deployment_eval.py",
               "num_actions_run": len(results),
               "num_theorems": len({r["theorem"] for r in results}),
               "status_histogram": dict(Counter(r["status"] for r in results)),
               "candidate_solves": len(solved),
               "rescue_candidates": len(rescues),
               "robust_rescue_candidates": sum(1 for r in rescues if r.get("robust")),
               "unknown_name": sum(1 for r in results if r["status"] == "unknown_name"),
               "type_error": sum(1 for r in results if r["status"] == "type_error"),
               "infra_error": sum(1 for r in results if r["status"] == "infra_error"),
               "theorems_with_setup_error": sorted({r["theorem"] for r in results if r["setup_error"]}),
               "rescue_candidate_targets": sorted({r["theorem"] for r in rescues})}
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI2 live deployment summary", "",
          f"- actions run: {summary['num_actions_run']} over {summary['num_theorems']} theorems",
          f"- candidate solves: {summary['candidate_solves']} | **rescue candidates (solved, no "
          f"control): {summary['rescue_candidates']}** (robust {summary['robust_rescue_candidates']})",
          f"- status: {summary['status_histogram']}",
          f"- unknown_name {summary['unknown_name']} | type_error {summary['type_error']} | "
          f"infra {summary['infra_error']}",
          f"- rescue candidate theorems: {summary['rescue_candidate_targets']}", ""]
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli2-eval] DONE actions={len(results)} solves={len(solved)} "
          f"rescue_candidates={len(rescues)} robust={summary['robust_rescue_candidates']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--case-json")
    ap.add_argument("--open-timeout", type=int, default=120)
    ap.add_argument("--timeout-per-tactic", type=int, default=20)
    ap.add_argument("--plan")
    ap.add_argument("--out-jsonl")
    ap.add_argument("--out-summary-json")
    ap.add_argument("--out-summary-md")
    ap.add_argument("--trace-dir")
    args = ap.parse_args()
    if args.worker:
        r = worker(json.loads(args.case_json), args.open_timeout, args.timeout_per_tactic)
        json.dump(r, open(args.worker_out, "w"), ensure_ascii=False)
        return
    driver(args)


if __name__ == "__main__":
    main()
