#!/usr/bin/env python3
"""RC2 Hardening Part 2 — forensic analysis of the +4 perturbation wins.

For each of the +4 NEEDS_REVIEW theorems (Set.ite_inter, Set.ite_inter_self,
Set.ite_compl, Set.ite_inter_compl_self):

  1. Trace summary: did `simp [Set.ite]` appear as an ADVANCING step in the RC2 run,
     and what finishing tactic/step closed it? What did RC1 do (nearest path)?
  2. Live direct probes (subprocess + per-probe SIGALRM, one Dojo session/theorem):
       simp [Set.ite]
       simp [Set.ite] <;> aesop
       simp [Set.ite] <;> simp_all
       simp [Set.ite] <;> try aesop
       aesop
       simp_all
       simp [Set.ext_iff]
       simp [Set.ite, Set.ext_iff]
       ext x <;> simp [Set.ite]
     -> the AUTHORITATIVE attribution evidence.

candidate_role: direct_close | enabling_step | search_order_perturbation | unknown
credit_status:  credited | excluded | needs_review | sx3_sequence_candidate

A theorem closed by `simp [Set.ite] <;> aesop` (but NOT by bare aesop/simp_all and
NOT single-shot simp[Set.ite]) is an SX3 depth-2 sequence candidate — NOT an RC2
credited single-shot win. NEVER modifies RC1 / NS24.

Outputs:
  perturbation_forensics.json / .md
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import traceback

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")

TARGETS = ["Set.ite_inter", "Set.ite_inter_self", "Set.ite_compl", "Set.ite_inter_compl_self"]
PROBES = [
    "simp [Set.ite]",
    "simp [Set.ite] <;> aesop",
    "simp [Set.ite] <;> simp_all",
    "simp [Set.ite] <;> try aesop",
    "aesop",
    "simp_all",
    "simp [Set.ext_iff]",
    "simp [Set.ite, Set.ext_iff]",
    "ext x <;> simp [Set.ite]",
]
BASELINES = {"aesop", "simp_all", "simp", "classical <;> aesop"}


class _T(Exception):
    pass


def _alarm(_s, _f):
    raise _T()


def _outcome(err, fin):
    if fin:
        return "solved"
    e = (err or "").lower()
    if not e:
        return "proof_failed"
    if "expected '{' or tactic" in e or "unexpected" in e:
        return "parse_error"
    if "unknown" in e and ("identifier" in e or "constant" in e):
        return "unknown_ident"
    if "applyexttheorem" in e:
        return "ext_not_applicable"
    return "proof_failed"


def _trace_summary(run_dirs, full_name):
    """Find a traces.jsonl with this theorem; summarize advancing/finishing steps."""
    for base in run_dirs:
        if not os.path.isdir(base):
            continue
        for root, _d, files in os.walk(base):
            if "traces.jsonl" not in files:
                continue
            path = os.path.join(root, "traces.jsonl")
            rows = []
            with open(path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if d.get("full_name") == full_name:
                        rows.append(d)
            if not rows:
                continue
            advancing = [(r.get("step"), r.get("tactic")) for r in rows
                         if r.get("result_kind") == "TacticState"]
            finished = [(r.get("step"), r.get("tactic")) for r in rows
                        if r.get("proof_finished")]
            simp_ite_advances = any("simp [Set.ite]" == t for _s, t in advancing)
            return {"trace_path": path,
                    "num_step_records": len(rows),
                    "advancing_steps": advancing[:20],
                    "finishing_step": finished[0] if finished else None,
                    "simp_set_ite_advances": simp_ite_advances}
    return {"trace_path": None, "num_step_records": 0, "advancing_steps": [],
            "finishing_step": None, "simp_set_ite_advances": None}


# ----------------------------- worker (live probes) -----------------------
def worker(args):
    cfg = json.load(open(args.cases_tmp))[args.worker_theorem]
    res = {"full_name": cfg["full_name"], "live": False, "probe_results": [],
           "setup_error": None}
    try:
        sys.path.insert(0, _REPO)
        import env as _env
        from core_types import TheoremConfig as _TC
        from lean_dojo import Dojo as _Dojo
        repo = _env.make_repo()
        thm = _env.make_theorem(repo, _TC(file_path=cfg["file_path"],
                                          full_name=cfg["full_name"]))
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _alarm)
        with _Dojo(thm) as (dojo, s0):
            res["live"] = True
            res["initial_goal"] = getattr(s0, "pp", None) or getattr(s0, "state", None)
            for pr in PROBES:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(args.timeout_per_probe)
                try:
                    out = _env.run_transition(dojo, thm, s0, pr)
                    rec = getattr(out, "record", None)
                    fin = bool(getattr(out, "is_finished", False))
                    err = getattr(rec, "error_message", None) if rec else None
                except _T:
                    res["probe_results"].append({"probe": pr, "solved": False,
                                                 "outcome": "timeout_inner"})
                    continue
                except Exception as e:
                    res["probe_results"].append({"probe": pr, "solved": False,
                                                 "outcome": "exception",
                                                 "error": str(e)[:120]})
                    continue
                finally:
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(0)
                res["probe_results"].append({"probe": pr, "solved": bool(fin),
                                             "outcome": _outcome(err, fin)})
    except Exception as e:
        res["setup_error"] = f"{type(e).__name__}: {str(e)[:160]}\n" + \
            traceback.format_exc()[-200:]
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


def _classify(probe_map, trace):
    """Decide candidate_role + credit_status from live probes + trace."""
    def solved(p):
        return probe_map.get(p, {}).get("solved") is True
    single_shot = solved("simp [Set.ite]")
    seq_aesop = solved("simp [Set.ite] <;> aesop")
    seq_simpall = solved("simp [Set.ite] <;> simp_all")
    seq_try = solved("simp [Set.ite] <;> try aesop")
    bare_aesop = solved("aesop")
    bare_simpall = solved("simp_all")
    if single_shot:
        return "direct_close", "credited", "single-shot simp [Set.ite] closes it"
    if bare_aesop or bare_simpall:
        return "search_order_perturbation", "excluded", (
            "a bare baseline (aesop/simp_all) closes it single-shot -> RC2's win is a "
            "search-order reorder finding the baseline; not SET_ITE-attributable")
    if seq_aesop or seq_simpall or seq_try:
        return "enabling_step", "sx3_sequence_candidate", (
            "simp [Set.ite] then aesop/simp_all closes it (depth-2 sequence), but bare "
            "baselines and single-shot simp[Set.ite] do NOT -> SX3 sequence candidate, "
            "not an RC2 single-shot credited win")
    return "search_order_perturbation", "excluded", (
        "neither single-shot simp[Set.ite], nor simp[Set.ite]<;>baseline, nor bare "
        "baselines close it from state0 -> RC2 win is a search-order artifact; excluded")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rc1-results",
                   default="project/evolve/experiments/rc2/out/rc1_baseline_results.json")
    p.add_argument("--rc2-results",
                   default="project/evolve/experiments/rc2/out/rc2_candidate_results.json")
    p.add_argument("--comparison",
                   default="project/evolve/experiments/rc2/out/rc2_comparison.json")
    p.add_argument("--out-json",
                   default="project/evolve/experiments/rc2_hardening/out/perturbation_forensics.json")
    p.add_argument("--out-md",
                   default="project/evolve/experiments/rc2_hardening/out/perturbation_forensics.md")
    p.add_argument("--timeout-per-probe", type=int, default=40)
    p.add_argument("--worker-theorem", type=int, default=None)
    p.add_argument("--worker-out", default=None)
    p.add_argument("--cases-tmp", default=None)
    args = p.parse_args(argv)
    if args.worker_theorem is not None:
        return worker(args)

    # resolve file_path for each target from the comparison's source sets
    fp = {}
    for resf in (args.rc1_results, args.rc2_results):
        try:
            d = json.load(open(resf))
        except Exception:
            continue
        for s in d.get("per_surface", []):
            for t in s.get("theorems", []):
                if t.get("full_name") in TARGETS and t.get("file_path"):
                    fp[t["full_name"]] = t["file_path"]
    cases = [{"full_name": fn, "file_path": fp.get(fn, "Mathlib/Data/Set/Basic.lean")}
             for fn in TARGETS]
    cases_tmp = "/tmp/rc2h_forensic_cases.json"
    json.dump(cases, open(cases_tmp, "w"))

    rc1_runs = [os.path.join(_REPO, "project/evolve/experiments/rc2/out/rc1_runs")]
    rc2_runs = [os.path.join(_REPO, "project/evolve/experiments/rc2/out/rc2_candidate_runs")]

    results = []
    for idx, c in enumerate(cases):
        fn = c["full_name"]
        wout = f"/tmp/rc2h_forensic_t{idx}.json"
        if os.path.exists(wout):
            os.remove(wout)
        hard = args.timeout_per_probe * (len(PROBES) + 1) + 90
        cmd = [sys.executable, _TIMEOUT_HELPER, str(hard), sys.executable,
               os.path.abspath(__file__), "--worker-theorem", str(idx),
               "--worker-out", wout, "--cases-tmp", cases_tmp,
               "--timeout-per-probe", str(args.timeout_per_probe)]
        print(f"[rc2h:forensic] ({idx+1}/{len(cases)}) {fn} probing live ...", flush=True)
        subprocess.run(cmd, capture_output=True, text=True)
        w = json.load(open(wout)) if os.path.exists(wout) else {"probe_results": [],
                                                                "live": False}
        probe_map = {pr["probe"]: pr for pr in w.get("probe_results", [])}
        rc1_tr = _trace_summary(rc1_runs, fn)
        rc2_tr = _trace_summary(rc2_runs, fn)
        role, credit, reason = _classify(probe_map, rc2_tr)
        results.append({
            "full_name": fn, "live": w.get("live"),
            "initial_goal": w.get("initial_goal"),
            "rc1_finished": False, "rc2_finished": True,
            "set_ite_emitted": rc2_tr.get("simp_set_ite_advances"),
            "set_ite_directly_closed": probe_map.get("simp [Set.ite]", {}).get("solved"),
            "rc2_finishing_step": rc2_tr.get("finishing_step"),
            "rc2_simp_set_ite_advances": rc2_tr.get("simp_set_ite_advances"),
            "rc1_trace_path": rc1_tr.get("trace_path"),
            "rc2_trace_path": rc2_tr.get("trace_path"),
            "direct_probes": {pr["probe"]: pr["outcome"] for pr in w.get("probe_results", [])},
            "candidate_role": role, "credit_status": credit, "reason": reason,
        })
        print(f"            -> role={role} credit={credit} "
              f"single_shot={probe_map.get('simp [Set.ite]',{}).get('solved')} "
              f"seq_aesop={probe_map.get('simp [Set.ite] <;> aesop',{}).get('solved')} "
              f"bare_aesop={probe_map.get('aesop',{}).get('solved')}", flush=True)

    hist = {}
    for r in results:
        hist[r["credit_status"]] = hist.get(r["credit_status"], 0) + 1
    out = {"targets": TARGETS, "probe_ladder": PROBES,
           "credit_status_histogram": hist,
           "sx3_sequence_candidates": [r["full_name"] for r in results
                                       if r["credit_status"] == "sx3_sequence_candidate"],
           "excluded": [r["full_name"] for r in results if r["credit_status"] == "excluded"],
           "credited": [r["full_name"] for r in results if r["credit_status"] == "credited"],
           "results": results,
           "note": "Live direct probes are authoritative. None of the +4 are RC2 "
                   "single-shot credited wins unless simp[Set.ite] closes them alone."}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), ensure_ascii=False, indent=2)

    L = ["# RC2 Hardening — Perturbation-Win Forensics (+4)", ""]
    L.append(f"- credit-status histogram: `{hist}`")
    L.append(f"- SX3 sequence candidates: {out['sx3_sequence_candidates']}")
    L.append(f"- excluded (search-order artifacts): {out['excluded']}")
    L.append("")
    L.append("| theorem | single-shot simp[Set.ite] | simp[Set.ite]<;>aesop | bare aesop | "
             "simp_all | role | credit |")
    L.append("|---|---|---|---|---|---|---|")
    for r in results:
        dp = r["direct_probes"]
        L.append(f"| `{r['full_name']}` | {dp.get('simp [Set.ite]')} | "
                 f"{dp.get('simp [Set.ite] <;> aesop')} | {dp.get('aesop')} | "
                 f"{dp.get('simp_all')} | {r['candidate_role']} | **{r['credit_status']}** |")
    L.append("")
    for r in results:
        L.append(f"## `{r['full_name']}`")
        L.append(f"- goal: `{(r.get('initial_goal') or '')[:160]}`")
        L.append(f"- candidate_role: **{r['candidate_role']}** | credit: **{r['credit_status']}**")
        L.append(f"- reason: {r['reason']}")
        L.append(f"- RC2 finishing step: {r['rc2_finishing_step']} | "
                 f"simp[Set.ite] advances in RC2 trace: {r['rc2_simp_set_ite_advances']}")
        L.append(f"- direct probes: {r['direct_probes']}")
        L.append("")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[rc2h:forensic] hist={hist} sx3={out['sx3_sequence_candidates']} "
          f"excluded={out['excluded']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
