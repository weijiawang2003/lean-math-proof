#!/usr/bin/env python3
"""SF3 Part 3 — deep live probe ladder for Multiset.toFinset_eq_singleton_iff.

Robust subprocess-per-probe design: the DRIVER spawns one WORKER subprocess per
probe under an OS-level hard timeout (scripts/run_with_timeout.py), so a single
recursion-bomb tactic can never stall the whole run. Each worker opens its own
LeanDojo session, runs ONE probe as a single tactic block from the theorem's
initial state, and (if it closes the goal) runs the minimality battery. Results
are exchanged via per-probe JSON files (not stdout), defeating console flakiness.

Every outcome is classified:
  solved | parse_error | max_recursion | ext_not_applicable | proof_failed |
  solved_but_not_a_win | subprocess_timeout | worker_crash | session_dead | other
A solve is recorded as ``minimality_status: unconfirmed`` + ``requires_ns23_relabel``.
Never fabricates results; never modifies RC1 / production configs.

Outputs:
  project/evolve/experiments/sf3/out/singleton_iff/probe_results.json
  project/evolve/experiments/sf3/out/singleton_iff/probe_results.md
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import traceback

LADDER = "project/evolve/experiments/sf3/singleton_iff_probe_ladder.json"
OUT_JSON = "project/evolve/experiments/sf3/out/singleton_iff/probe_results.json"
OUT_MD = "project/evolve/experiments/sf3/out/singleton_iff/probe_results.md"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")


class _ProbeTimeout(Exception):
    pass


def _alarm(_s, _f):
    raise _ProbeTimeout()


def classify(err, solved):
    if solved:
        return "solved"
    e = (err or "").lower()
    if not e:
        return "proof_failed"
    if ("expected end of input" in e or "expected '{' or tactic" in e
            or "unexpected token" in e):
        return "parse_error"
    if "maximum recursion depth" in e:
        return "max_recursion"
    if "applyexttheorem only applies" in e:
        return "ext_not_applicable"
    if "no goals" in e:
        return "no_goals"
    return "proof_failed"


def flatten(ladder, max_probes):
    out = []
    for fam, items in ladder.get("families", {}).items():
        for it in items:
            out.append({"family": fam, "probe": it["probe"], "label": it.get("label"),
                        "known_bad": it.get("known_bad", False)})
    # cheap-first: multi-line / aesop last
    out.sort(key=lambda pr: (2 if "\n" in pr["probe"] else 0) + (1 if "aesop" in pr["probe"] else 0))
    return out[:max_probes]


# ----------------------------- worker -------------------------------------
def worker(args):
    ladder = json.load(open(args.ladder))
    target = args.theorem or ladder["target"]
    probes = flatten(ladder, args.max_probes)
    pr = probes[args.worker]
    prefix = ladder.get("minimality_prefix_probes", ["simp", "simp_all", "aesop"])
    fp_candidates = [ladder.get("file_path_frontier_backfill"), ladder.get("file_path_real")]
    fp_candidates = [x for x in fp_candidates if x]

    res = {"index": args.worker, **pr, "solved": False, "outcome": "worker_crash",
           "file_path_used": None, "error": None}
    try:
        sys.path.insert(0, _REPO)
        import env as _env
        from core_types import TheoremConfig as _TC
        from lean_dojo import Dojo as _Dojo
        repo = _env.make_repo()
        thm = None
        last = None
        for fp in fp_candidates:
            try:
                thm = _env.make_theorem(repo, _TC(file_path=fp, full_name=target))
                res["file_path_used"] = fp
                break
            except Exception as e:
                last = f"{fp}: {type(e).__name__}: {str(e)[:120]}"
        if thm is None:
            res["error"] = f"make_theorem failed: {last}"
            json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
            return 0
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _alarm)
        with _Dojo(thm) as (dojo, state0):
            res["initial_goal"] = getattr(state0, "pp", None) or getattr(state0, "state", None)

            def apply(tac, tmo):
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(tmo)
                try:
                    out = _env.run_transition(dojo, thm, state0, tac)
                finally:
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(0)
                rec = getattr(out, "record", None)
                return (bool(getattr(out, "is_finished", False)),
                        bool(getattr(out, "session_dead", False)),
                        getattr(rec, "error_message", None) if rec else None,
                        getattr(rec, "state_pp", None) if rec else None)

            try:
                fin, dead, err, st = apply(pr["probe"], args.per_probe_timeout)
            except _ProbeTimeout:
                res["outcome"] = "timeout_inner"
                res["error"] = f"exceeded {args.per_probe_timeout}s (inner alarm)"
                json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
                return 0
            is_win = (fin and not pr.get("known_bad")
                      and pr.get("label") != "parse_capability_only"
                      and "sorry" not in pr["probe"])
            res["solved"] = bool(fin)
            res["outcome"] = ("solved_but_not_a_win" if (fin and not is_win)
                              else classify(err, fin))
            if err and not fin:
                res["error"] = err[:240]
            if st and not fin:
                res["produced_induction_tangle"] = bool(
                    ("• {a}" in st or "::ₘ" in st) and "↔" in st)
            if dead:
                res["session_dead"] = True
            if is_win:
                mini = []
                for mp in prefix:
                    try:
                        f2, _, _, _ = apply(mp, args.per_probe_timeout)
                    except Exception:
                        f2 = False
                    mini.append({"probe": mp, "solved": f2})
                res["minimality_status"] = "unconfirmed"
                res["requires_ns23_relabel"] = True
                res["minimality_prefix_results"] = mini
    except Exception as e:
        res["outcome"] = "worker_crash"
        res["error"] = f"{type(e).__name__}: {str(e)[:200]}\n" + traceback.format_exc()[-400:]
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


# ----------------------------- driver -------------------------------------
def driver(args):
    ladder = json.load(open(args.ladder))
    target = args.theorem or ladder["target"]
    probes = flatten(ladder, args.max_probes)
    hard = args.per_probe_timeout + 45  # Dojo-open margin
    outcomes, solved_probe, initial_goal = [], None, None

    def checkpoint():
        res = {"target": target, "num_probes": len(probes),
               "num_solved": sum(1 for o in outcomes if o.get("solved")),
               "solved_probe": solved_probe, "initial_goal": initial_goal,
               "outcomes": outcomes, "outcome_histogram": hist(outcomes),
               "design": "subprocess-per-probe; OS hard timeout via run_with_timeout",
               "note": "No solve is a confirmed win; NS23 minimal-sufficient relabel + "
                       "deterministic reproduction required before promotion. RC1/"
                       "production configs not modified."}
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        json.dump(res, open(args.out_json, "w"), ensure_ascii=False, indent=2)
        return res

    for idx in range(len(probes)):
        wout = f"/tmp/sf3_probe_w{idx}.json"
        if os.path.exists(wout):
            os.remove(wout)
        cmd = [sys.executable, _TIMEOUT_HELPER, str(hard), sys.executable,
               os.path.abspath(__file__), "--worker", str(idx), "--worker-out", wout,
               "--ladder", args.ladder, "--per-probe-timeout", str(args.per_probe_timeout),
               "--max-probes", str(args.max_probes)]
        if args.theorem:
            cmd += ["--theorem", args.theorem]
        rc = subprocess.run(cmd, capture_output=True, text=True).returncode
        if os.path.exists(wout):
            try:
                rec = json.load(open(wout))
            except Exception as e:
                rec = {"index": idx, **probes[idx], "solved": False,
                       "outcome": "worker_crash", "error": f"unreadable worker out: {e}"}
        else:
            rec = {"index": idx, **probes[idx], "solved": False,
                   "outcome": "subprocess_timeout",
                   "error": f"no worker output (rc={rc}); OS-killed at {hard}s"}
        if initial_goal is None and rec.get("initial_goal"):
            initial_goal = rec["initial_goal"]
        outcomes.append(rec)
        if rec.get("solved") and rec.get("outcome") == "solved" and solved_probe is None:
            solved_probe = {"probe": rec["probe"], "family": rec["family"],
                            "label": rec.get("label"),
                            "minimality_status": rec.get("minimality_status", "unconfirmed"),
                            "requires_ns23_relabel": rec.get("requires_ns23_relabel", True),
                            "minimality_prefix_results": rec.get("minimality_prefix_results")}
        checkpoint()

    final = checkpoint()
    write_md(final, args.out_md)
    print(f"[sf3:probe] probes={len(probes)} solved={final['num_solved']} "
          f"solved_probe={solved_probe['probe'] if solved_probe else None}")
    print(f"[sf3:probe] histogram={final['outcome_histogram']}")
    return 0


def hist(outcomes):
    h = {}
    for o in outcomes:
        h[o.get("outcome", "?")] = h.get(o.get("outcome", "?"), 0) + 1
    return h


def write_md(r, path):
    L = []
    a = L.append
    a(f"# SF3 Deep Probe Ladder — `{r['target']}`")
    a("")
    a(f"- probes: {r['num_probes']} | solved: {r['num_solved']} | "
      f"histogram: `{r['outcome_histogram']}`")
    a(f"- design: {r['design']}")
    if r.get("solved_probe"):
        sp = r["solved_probe"]
        a(f"- **SOLVED** by (`{sp['family']}`, label={sp['label']}): `{sp['probe']}`")
        a(f"  - minimality (unconfirmed, needs NS23): {sp.get('minimality_prefix_results')}")
    else:
        a("- **No probe closed the goal.**")
    a("")
    a("## Per-probe outcomes")
    a("")
    a("| family | outcome | solved | tangle | probe | error |")
    a("|---|---|---|---|---|---|")
    for o in r["outcomes"]:
        err = (o.get("error") or "").replace("\n", " ")[:70]
        a(f"| {o.get('family', '')} | {o.get('outcome')} | {o.get('solved')} | "
          f"{o.get('produced_induction_tangle', '')} | "
          f"`{o.get('probe', '')[:55].replace(chr(10), ' / ')}` | {err} |")
    a("")
    a("> " + r["note"])
    open(path, "w").write("\n".join(L))


def single_session(args):
    """Open ONE Dojo and run every probe from state0 with a per-probe SIGALRM and
    checkpoint-after-each. Amortizes the (slow) cold Dojo open across all probes —
    the right design when open cost >> per-tactic cost."""
    ladder = json.load(open(args.ladder))
    target = args.theorem or ladder["target"]
    probes = flatten(ladder, args.max_probes)
    prefix = ladder.get("minimality_prefix_probes", ["simp", "simp_all", "aesop"])
    fp_candidates = [x for x in [ladder.get("file_path_frontier_backfill"),
                                 ladder.get("file_path_real")] if x]
    outcomes, solved_probe, initial_goal = [], None, None
    meta = {"available": False, "file_path_used": None, "setup_error": None}

    def checkpoint():
        res = {"target": target, "num_probes": len(probes),
               "num_solved": sum(1 for o in outcomes if o.get("solved")),
               "solved_probe": solved_probe, "initial_goal": initial_goal,
               "live": meta, "outcomes": outcomes, "outcome_histogram": hist(outcomes),
               "design": "single Dojo session; per-probe SIGALRM + checkpoint",
               "note": "No solve is a confirmed win; NS23 minimal-sufficient relabel + "
                       "deterministic reproduction required before promotion. RC1/"
                       "production configs not modified."}
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        json.dump(res, open(args.out_json, "w"), ensure_ascii=False, indent=2)
        return res

    try:
        sys.path.insert(0, _REPO)
        import env as _env
        from core_types import TheoremConfig as _TC
        from lean_dojo import Dojo as _Dojo
        repo = _env.make_repo()
        thm, last = None, None
        for fp in fp_candidates:
            try:
                thm = _env.make_theorem(repo, _TC(file_path=fp, full_name=target))
                meta["file_path_used"] = fp
                break
            except Exception as e:
                last = f"{fp}: {type(e).__name__}: {str(e)[:120]}"
        if thm is None:
            meta["setup_error"] = f"make_theorem failed: {last}"
            checkpoint()
            print(f"[sf3:single] setup failed: {meta['setup_error']}")
            return 0
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _alarm)
        with _Dojo(thm) as (dojo, state0):
            meta["available"] = True
            initial_goal = getattr(state0, "pp", None) or getattr(state0, "state", None)

            def apply(tac, tmo):
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(tmo)
                try:
                    out = _env.run_transition(dojo, thm, state0, tac)
                finally:
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(0)
                rec = getattr(out, "record", None)
                return (bool(getattr(out, "is_finished", False)),
                        bool(getattr(out, "session_dead", False)),
                        getattr(rec, "error_message", None) if rec else None,
                        getattr(rec, "state_pp", None) if rec else None)

            for pr in probes:
                try:
                    fin, dead, err, st = apply(pr["probe"], args.per_probe_timeout)
                except _ProbeTimeout:
                    outcomes.append({**pr, "solved": False, "outcome": "timeout_killed",
                                     "error": f"exceeded {args.per_probe_timeout}s; "
                                              "Dojo may be unusable, stopping"})
                    checkpoint()
                    break
                except Exception as e:
                    outcomes.append({**pr, "solved": False, "outcome": "exception",
                                     "error": f"{type(e).__name__}: {str(e)[:160]}"})
                    checkpoint()
                    continue
                is_win = (fin and not pr.get("known_bad")
                          and pr.get("label") != "parse_capability_only"
                          and "sorry" not in pr["probe"])
                rec = {**pr, "solved": bool(fin),
                       "outcome": ("solved_but_not_a_win" if (fin and not is_win)
                                   else classify(err, fin))}
                if err and not fin:
                    rec["error"] = err[:240]
                if st and not fin:
                    rec["produced_induction_tangle"] = bool(
                        ("• {a}" in st or "::ₘ" in st) and "↔" in st)
                outcomes.append(rec)
                checkpoint()
                if dead:
                    outcomes.append({"probe": "<session_dead>", "solved": False,
                                     "outcome": "session_dead",
                                     "error": "REPL crashed; remaining probes untested"})
                    checkpoint()
                    break
                if is_win and solved_probe is None:
                    mini = []
                    for mp in prefix:
                        try:
                            f2, _, _, _ = apply(mp, args.per_probe_timeout)
                        except Exception:
                            f2 = False
                        mini.append({"probe": mp, "solved": f2})
                    solved_probe = {"probe": pr["probe"], "family": pr["family"],
                                    "label": pr.get("label"),
                                    "minimality_status": "unconfirmed",
                                    "requires_ns23_relabel": True,
                                    "minimality_prefix_results": mini}
                    checkpoint()
    except Exception as e:
        meta["setup_error"] = (meta.get("setup_error") or "") + \
            f"\nsession error: {type(e).__name__}: {str(e)[:200]}"
        checkpoint()

    final = checkpoint()
    write_md(final, args.out_md)
    print(f"[sf3:single] live={meta['available']} file={meta.get('file_path_used')} "
          f"probes={len(probes)} solved={final['num_solved']} "
          f"solved_probe={solved_probe['probe'] if solved_probe else None}")
    print(f"[sf3:single] histogram={final['outcome_histogram']}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ladder", default=LADDER)
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-md", default=OUT_MD)
    p.add_argument("--max-probes", type=int, default=80)
    p.add_argument("--theorem", default=None)
    p.add_argument("--per-probe-timeout", type=int, default=60)
    p.add_argument("--mode", choices=["single", "driver"], default="single",
                   help="single = one Dojo for all probes (default; amortizes open); "
                        "driver = subprocess-per-probe with OS hard timeout")
    p.add_argument("--worker", type=int, default=None, help="internal: run one probe by index")
    p.add_argument("--worker-out", default=None, help="internal: worker result path")
    args = p.parse_args(argv)
    if args.worker is not None:
        return worker(args)
    if args.mode == "driver":
        return driver(args)
    return single_session(args)


if __name__ == "__main__":
    raise SystemExit(main())
