#!/usr/bin/env python3
"""RC2 Part 4 — RC1 + SET_ITE_SIMP additive candidate run.

Additive contract (RC1 behavior NEVER modified):
  candidate_finished = literal_rc1_finished
                       OR (gate_fired AND `simp [Set.ite]` closes the goal live)

For each validation theorem:
  * read literal RC1 finished from --literal-rc1-results,
  * if RC1 already finished -> no candidate work (RC1_ALREADY_SOLVED),
  * else evaluate the single SET_ITE_SIMP gate over (full_name, live goal pp); if it
    fires, run `simp [Set.ite]` live in a fresh Dojo session (subprocess + OS hard
    timeout, mirroring the SF2/SX2 harness). Also run the baseline battery
    (simp/simp_all/aesop/classical<;>aesop) on RC1-failed gate-fired theorems so the
    minimal relabel is fully offline.

The gate matches the RC2 candidate policy schema:
  requires_namespace_or_name_contains, requires_goal_or_name_contains_any,
  forbids_namespace_or_name_contains, max_emissions_per_theorem.

NEVER modifies RC1 / NS24. Off-by-default in production.

Outputs:
  candidate_results.json / .md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import traceback

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
BASELINES = ["simp", "simp_all", "aesop", "classical <;> aesop"]
_ITE_RE = None


class _ProbeTimeout(Exception):
    pass


def _alarm(_s, _f):
    raise _ProbeTimeout()


def gate_fires(gate, full_name, goal):
    """Evaluate the single RC2 SET_ITE_SIMP gate (schema-driven)."""
    name = full_name or ""
    g = goal or ""
    hay_name = name
    hay_all = name + "\n" + g
    for tok in gate.get("requires_namespace_or_name_contains", []):
        if tok not in hay_name:
            return False, "missing required name token: " + tok
    # ite/if token in NAME or GOAL
    anyset = gate.get("requires_goal_or_name_contains_any", [])
    if anyset:
        def _has_ite(h):
            if ".ite" in h:
                return True
            return bool(re.search(r"(?<![A-Za-z])ite(?![A-Za-z])", h) or
                        re.search(r"(?<![A-Za-z])if(?![A-Za-z])", h))
        if not _has_ite(hay_all):
            return False, "no ite/if token in name or goal"
    for tok in gate.get("forbids_namespace_or_name_contains", []):
        if tok in hay_all:
            return False, "forbidden token present: " + tok
    return True, "ok"


def classify_outcome(err, solved):
    if solved:
        return "solved"
    e = (err or "").lower()
    if not e:
        return "proof_failed"
    if "expected '{' or tactic" in e or "unexpected" in e or "expected end of input" in e:
        return "parse_error"
    if "applyexttheorem only applies" in e:
        return "ext_not_applicable"
    if "unknown" in e and ("identifier" in e or "constant" in e):
        return "unknown_ident"
    return "proof_failed"


def _load_rc1(path):
    obj = json.load(open(path))
    by = {}
    for t in obj.get("merged_theorems", []):
        by[t["full_name"]] = t
    return by


def _load_all_cases(ts_dir, set_names):
    """Return ordered unique cases across the named sets, tagging roles."""
    seen, cases = set(), []
    for sname in set_names:
        fpath = os.path.join(ts_dir, f"{sname}.json")
        if not os.path.exists(fpath):
            continue
        obj = json.load(open(fpath))
        rows = obj.get(sname) or (list(obj.values())[0] if obj else [])
        for r in rows:
            fn = r.get("full_name")
            if fn in seen:
                # still record extra set membership
                for c in cases:
                    if c["full_name"] == fn:
                        c["in_sets"].append(sname)
                continue
            seen.add(fn)
            cases.append({**r, "in_sets": [sname]})
    return cases


# ----------------------------- worker -------------------------------------
def worker(args):
    cases = json.load(open(args.cases_tmp))
    gate = json.load(open(args.gate_policy))["gates"][0]["gate"]
    case = cases[args.worker_theorem]
    res = {"full_name": case["full_name"], "file_path": case.get("file_path"),
           "live": False, "initial_goal": None,
           "set_ite_gate_fired": False, "set_ite_tactic": None,
           "set_ite_finished": False, "baseline_outcomes": [],
           "baseline_solved_by": None, "parse_error": False, "error": None,
           "off_gate": False, "setup_error": None}
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

            fired, why = gate_fires(gate, case["full_name"], goal)
            res["set_ite_gate_fired"] = fired
            res["gate_reason"] = why
            is_set = "Set" in case["full_name"] and "Multiset" not in case["full_name"]
            res["off_gate"] = fired and not is_set
            if fired:
                res["set_ite_tactic"] = "simp [Set.ite]"
                # baseline battery first (for offline minimal relabel)
                for b in BASELINES:
                    try:
                        f, d, e = apply(b)
                    except _ProbeTimeout:
                        res["baseline_outcomes"].append({"probe": b, "solved": False,
                                                          "outcome": "timeout_inner"})
                        continue
                    except Exception as ex:
                        res["baseline_outcomes"].append({"probe": b, "solved": False,
                                                          "outcome": "exception",
                                                          "error": str(ex)[:100]})
                        continue
                    res["baseline_outcomes"].append(
                        {"probe": b, "solved": bool(f), "outcome": classify_outcome(e, f)})
                    if f and res["baseline_solved_by"] is None:
                        res["baseline_solved_by"] = b
                # the candidate tactic
                try:
                    f, d, e = apply("simp [Set.ite]")
                    oc = classify_outcome(e, f)
                    res["set_ite_finished"] = bool(f)
                    if oc == "parse_error":
                        res["parse_error"] = True
                    if e and not f:
                        res["error"] = e[:160]
                except _ProbeTimeout:
                    res["error"] = f"simp [Set.ite] timeout {args.timeout_per_probe}s"
    except Exception as e:
        res["setup_error"] = f"{type(e).__name__}: {str(e)[:200]}\n" + \
            traceback.format_exc()[-300:]
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


# ----------------------------- driver -------------------------------------
def driver(args):
    manifest = json.load(open(args.manifest))
    ts_dir = manifest["theorem_sets_dir"]
    set_names = [s.strip() for s in args.sets.split(",") if s.strip()]
    cases = _load_all_cases(ts_dir, set_names)
    cases = [c for c in cases if c.get("file_path")]  # live needs a path
    rc1 = _load_rc1(args.literal_rc1_results)

    # write a tmp cases file for workers
    cases_tmp = "/tmp/rc2_cand_cases.json"
    json.dump(cases, open(cases_tmp, "w"))

    results = []
    for idx, case in enumerate(cases):
        fn = case["full_name"]
        rc1_fin = bool(rc1.get(fn, {}).get("literal_rc1_finished"))
        rec = {"full_name": fn, "file_path": case["file_path"],
               "in_sets": case["in_sets"],
               "validation_role": case.get("validation_role"),
               "literal_rc1_finished": rc1_fin,
               "set_ite_gate_fired": False, "set_ite_tactic": None,
               "set_ite_finished": False, "baseline_outcomes": [],
               "baseline_solved_by": None,
               "candidate_finished": rc1_fin, "new_win_over_literal_rc1": False,
               "off_gate": False, "parse_error": False, "error": None,
               "live": None}
        if rc1_fin:
            # additive: RC1 already solved; no candidate probe needed
            rec["note"] = "RC1 already solved; candidate adds nothing (additive)"
            results.append(rec)
            print(f"[rc2:cand] ({idx+1}/{len(cases)}) {fn} RC1_SOLVED", flush=True)
            _checkpoint(results, args)
            continue
        # RC1 failed -> run gate + (if fired) baselines + simp[Set.ite] live
        hard = args.timeout_per_probe * (len(BASELINES) + 2) + 90
        wout = f"/tmp/rc2_cand_t{idx}.json"
        if os.path.exists(wout):
            os.remove(wout)
        cmd = [sys.executable, _TIMEOUT_HELPER, str(hard), sys.executable,
               os.path.abspath(__file__), "--worker-theorem", str(idx),
               "--worker-out", wout, "--cases-tmp", cases_tmp,
               "--gate-policy", args.gate_policy,
               "--timeout-per-probe", str(args.timeout_per_probe)]
        sub = subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(wout):
            w = json.load(open(wout))
            rec.update({k: w.get(k) for k in
                        ("live", "set_ite_gate_fired", "set_ite_tactic",
                         "set_ite_finished", "baseline_outcomes", "baseline_solved_by",
                         "off_gate", "parse_error", "error", "initial_goal",
                         "gate_reason", "setup_error")})
            rec["candidate_finished"] = rc1_fin or bool(w.get("set_ite_finished"))
            rec["new_win_over_literal_rc1"] = (not rc1_fin) and bool(w.get("set_ite_finished"))
        else:
            rec["setup_error"] = f"no worker output (rc={sub.returncode})"
        results.append(rec)
        print(f"[rc2:cand] ({idx+1}/{len(cases)}) {fn} rc1={rc1_fin} "
              f"gate={rec['set_ite_gate_fired']} set_ite={rec['set_ite_finished']} "
              f"new_win={rec['new_win_over_literal_rc1']}", flush=True)
        _checkpoint(results, args)

    final = _finalize(results, args)
    print(f"[rc2:cand] DONE {final['metrics']}")
    return 0


def _metrics(results):
    total = len(results)
    rc1 = sum(1 for r in results if r["literal_rc1_finished"])
    cand = sum(1 for r in results if r["candidate_finished"])
    new_wins = sum(1 for r in results if r["new_win_over_literal_rc1"])
    fired = sum(1 for r in results if r["set_ite_gate_fired"])
    emit_solved = sum(1 for r in results if r["set_ite_gate_fired"] and r["set_ite_finished"])
    emit_failed = sum(1 for r in results if r["set_ite_gate_fired"] and not r["set_ite_finished"])
    not_emitted = total - fired
    off_gate = sum(1 for r in results if r.get("off_gate"))
    return {"total": total, "literal_rc1_solved": rc1, "candidate_solved": cand,
            "new_wins_over_literal_rc1": new_wins,
            "candidate_regressions": 0,
            "regression_note": "0 by construction — additive; RC1 behavior unaltered.",
            "gate_fired": fired, "off_gate_emissions": off_gate,
            "gate_precision": {"emitted_and_solved": emit_solved,
                               "emitted_and_failed": emit_failed,
                               "not_emitted": not_emitted}}


def _checkpoint(results, args):
    agg = {"gate_policy": args.gate_policy,
           "literal_rc1_results": args.literal_rc1_results,
           "production_default_emits": False, "metrics": _metrics(results),
           "results": results}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(agg, open(args.out_json, "w"), ensure_ascii=False, indent=2)
    return agg


def _finalize(results, args):
    agg = _checkpoint(results, args)
    m = agg["metrics"]
    L = ["# RC2 — RC1 + SET_ITE_SIMP Candidate Eval", ""]
    L.append(f"- total={m['total']} | literal RC1 solved={m['literal_rc1_solved']} | "
             f"candidate solved={m['candidate_solved']} | "
             f"**new wins over literal RC1={m['new_wins_over_literal_rc1']}** | "
             f"regressions={m['candidate_regressions']} | off-gate={m['off_gate_emissions']}")
    L.append(f"- gate fired={m['gate_fired']} | precision: {m['gate_precision']}")
    L.append("")
    L.append("| theorem | sets | rc1 | gate | set_ite | candidate | new_win | off_gate |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        L.append(f"| `{r['full_name']}` | {','.join(r['in_sets'])} | "
                 f"{r['literal_rc1_finished']} | {r['set_ite_gate_fired']} | "
                 f"{r['set_ite_finished']} | {r['candidate_finished']} | "
                 f"{r['new_win_over_literal_rc1']} | {r.get('off_gate')} |")
    open(args.out_md, "w").write("\n".join(L))
    return agg


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/validation_manifest.json")
    p.add_argument("--gate-policy",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/set_ite_simp_gate_policy.json")
    p.add_argument("--literal-rc1-results",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/out/literal_rc1_results.json")
    p.add_argument("--sets",
                   default="set_ite_known_wins,set_ite_selected_failures,set_ite_fresh_holdout")
    p.add_argument("--out-json",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/out/candidate_results.json")
    p.add_argument("--out-md",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/out/candidate_results.md")
    p.add_argument("--timeout-per-probe", type=int, default=30)
    p.add_argument("--worker-theorem", type=int, default=None)
    p.add_argument("--worker-out", default=None)
    p.add_argument("--cases-tmp", default=None)
    args = p.parse_args(argv)
    if args.worker_theorem is not None:
        return worker(args)
    return driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
