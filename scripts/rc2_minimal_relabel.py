#!/usr/bin/env python3
"""RC2 Part 6 — NS23 minimal-sufficient relabel for every RC2 new win.

For each new win in rc2_comparison.json, decide whether it is a genuine SET_ITE_SIMP
win using the baseline ladder (simp / simp_all / aesop / classical<;>aesop /
simp [Set.ite]). Baseline outcomes are sourced from the RC2-validation candidate run
(which ran exactly this live ladder on the Set.ite theorems); any new win not covered
there is run LIVE here in a fresh Dojo session (subprocess + OS timeout).

Classification:
  TRUE_SET_ITE_SIMP_WIN     literal RC1 failed, all baselines failed, simp [Set.ite] solved
  BASELINE_DUPLICATE        a simpler baseline also closed it
  RC1_ALREADY_SOLVED        RC1 finished (not a win) — should not appear among new wins
  PARSER_ARTIFACT           prior failure was a parser/runner artifact
  UNEXPECTED_WIN_NEEDS_REVIEW  a non-Set.ite win, or no baseline evidence available

Outputs:
  rc2_minimal_relabel_results.json / .md
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
BASELINES = ["simp", "simp_all", "aesop", "classical <;> aesop"]


class _T(Exception):
    pass


def _alarm(_s, _f):
    raise _T()


def _prior_baselines(prior_path):
    """full_name -> {baseline_solved_by, all_failed, file_path} from RC2-validation."""
    out = {}
    if not (prior_path and os.path.exists(prior_path)):
        return out
    for r in json.load(open(prior_path)).get("results", []):
        if r.get("baseline_outcomes"):
            solved = [b["probe"] for b in r["baseline_outcomes"] if b.get("solved")]
            out[r["full_name"]] = {"baseline_solved_by": (solved[0] if solved else None),
                                   "all_failed": len(solved) == 0,
                                   "file_path": r.get("file_path"),
                                   "set_ite_finished": r.get("set_ite_finished")}
    return out


def _live_baselines(worker_out, full_name, file_path, timeout):
    """Run the baseline ladder + simp[Set.ite] live for one theorem (subprocess)."""
    code = f'''
import json,signal,sys,os
sys.path.insert(0,{_REPO!r})
res={{"full_name":{full_name!r},"baseline_solved_by":None,"all_failed":True,"set_ite_finished":False,"live":False,"err":None}}
try:
    import env as E
    from core_types import TheoremConfig as TC
    from lean_dojo import Dojo
    repo=E.make_repo(); thm=E.make_theorem(repo,TC(file_path={file_path!r},full_name={full_name!r}))
    def al(s,f): raise TimeoutError()
    signal.signal(signal.SIGALRM,al)
    with Dojo(thm) as (d,s0):
        res["live"]=True
        def ap(t):
            signal.alarm({timeout})
            try: o=E.run_transition(d,thm,s0,t)
            finally: signal.alarm(0)
            return bool(getattr(o,"is_finished",False))
        for b in {BASELINES!r}:
            try:
                if ap(b):
                    res["baseline_solved_by"]=b; res["all_failed"]=False; break
            except Exception: pass
        try: res["set_ite_finished"]=ap("simp [Set.ite]")
        except Exception: pass
except Exception as e:
    res["err"]=str(e)[:200]
json.dump(res,open({worker_out!r},"w"))
'''
    cmd = [sys.executable, _TIMEOUT_HELPER, str(timeout * (len(BASELINES) + 3) + 90),
           sys.executable, "-c", code]
    subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(worker_out):
        return json.load(open(worker_out))
    return None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--comparison", required=True)
    p.add_argument("--prior-candidate",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/out/candidate_results.json")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--timeout-per-probe", type=int, default=30)
    args = p.parse_args(argv)

    comp = json.load(open(args.comparison))
    new_wins = comp.get("new_win_classification", [])
    prior = _prior_baselines(args.prior_candidate)

    rows, hist = [], {}
    # dedupe by theorem (a win may appear on multiple surfaces)
    seen = {}
    for w in new_wins:
        fn = w["full_name"]
        if fn in seen:
            continue
        seen[fn] = w
    for fn, w in seen.items():
        base = prior.get(fn)
        source = "prior_candidate_baselines"
        if base is None:
            wo = f"/tmp/rc2_relabel_{abs(hash(fn))%99999}.json"
            base = _live_baselines(wo, fn, None, args.timeout_per_probe) or {}
            source = "live"
        all_failed = base.get("all_failed", None)
        solved_by = base.get("baseline_solved_by")
        single_shot = base.get("set_ite_finished")  # simp[Set.ite] solved single-shot
        if not str(fn).startswith("Set.ite"):
            cls = "UNEXPECTED_WIN_NEEDS_REVIEW"
            reason = "new win is not a Set.ite theorem"
        elif solved_by:
            cls = "BASELINE_DUPLICATE"
            reason = f"baseline `{solved_by}` also closes it"
        elif single_shot and all_failed:
            cls = "TRUE_SET_ITE_SIMP_WIN"
            reason = "literal RC1 failed, all baselines failed, single-shot simp [Set.ite] solved"
        elif all_failed and not single_shot:
            # gate fired, baselines fail, but simp[Set.ite] does NOT close single-shot:
            # the RC2 full-wrapper win came via a multi-step search path (e.g. aesop@2)
            # that is a deterministic search-PERTURBATION side effect of adding the
            # action to priority_templates, not logically attributable to simp[Set.ite].
            cls = "UNEXPECTED_WIN_NEEDS_REVIEW"
            reason = ("multi-step/search-perturbation win (simp [Set.ite] alone does NOT "
                      "close it); NOT credited to SET_ITE_SIMP")
        else:
            cls = "UNEXPECTED_WIN_NEEDS_REVIEW"
            reason = f"no conclusive single-shot evidence ({base})"
        hist[cls] = hist.get(cls, 0) + 1
        rows.append({"full_name": fn, "surface": w.get("surface"),
                     "comparison_class": w.get("classification"),
                     "baseline_source": source, "baseline_solved_by": solved_by,
                     "all_baselines_failed": all_failed,
                     "attribution": cls, "reason": reason})

    true_wins = sorted(r["full_name"] for r in rows if r["attribution"] == "TRUE_SET_ITE_SIMP_WIN")
    out = {"attribution_histogram": hist,
           "true_set_ite_simp_win_count": len(true_wins),
           "true_set_ite_simp_wins": true_wins,
           "credited_delta": len(true_wins),
           "policy": "Promotion credits only TRUE_SET_ITE_SIMP_WIN. Any "
                     "BASELINE_DUPLICATE / UNEXPECTED is excluded from the delta.",
           "rows": rows}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), ensure_ascii=False, indent=2)

    L = ["# RC2 — Minimal-Sufficient Relabel of New Wins", ""]
    L.append(f"- attribution histogram: `{hist}`")
    L.append(f"- **TRUE_SET_ITE_SIMP_WIN = {len(true_wins)}** (credited delta): {true_wins}")
    L.append(f"- {out['policy']}")
    L.append("")
    L.append("| theorem | surface | baseline_solved_by | all_failed | attribution |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        L.append(f"| `{r['full_name']}` | {r['surface']} | {r['baseline_solved_by']} | "
                 f"{r['all_baselines_failed']} | **{r['attribution']}** |")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[rc2:relabel] new_wins={len(rows)} hist={hist} "
          f"TRUE_SET_ITE_SIMP_WIN={len(true_wins)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
