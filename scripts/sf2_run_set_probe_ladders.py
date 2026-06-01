#!/usr/bin/env python3
"""SF2 Part 4 — live LeanDojo probe runner for the selected Set failures.

Design (mirrors the proven sf3 singleton runner, extended to many theorems):
  * DRIVER iterates over selected cases; for each it spawns ONE worker subprocess
    under an OS-level hard timeout (scripts/run_with_timeout.py). A worker crash or
    hang on one theorem can never stall the whole run.
  * WORKER opens ONE Dojo session for its theorem (amortising the slow cold open),
    runs every gated probe from the initial state with a per-probe SIGALRM, and on
    the FIRST genuine win runs the minimality battery. Results go to a per-theorem
    JSON file, defeating console flakiness.

Probe gating per case: Family A always; B/D for equality+membership; C for iff;
E always; F (source_inspired) for that theorem; G (negative controls) only with
--enable-neg-controls. Cheap-first ordering so the first win is near-minimal.

Outcome vocabulary: solved | solved_but_not_a_win | parse_error | max_recursion |
ext_not_applicable | no_goals | proof_failed | timeout_inner | subprocess_timeout
| session_dead | worker_crash.

Per-theorem gap classification (honest, conservative):
  tactic_gap        — a non-baseline probe (not in RC1's simp/aesop battery) solves
                      it while the baselines fail.
  routing_gap       — a plain baseline (simp/simp_all/aesop/classical<;>aesop) solves
                      it, yet RC1 reported all tactics errored (mis-ordered/gated).
  search_depth_gap  — only a multi-step / rw-bridge / source-inspired probe solves.
  needs_missing_lemma / needs_deeper_search — nothing closes the goal.

No solve is a confirmed win; every win is `minimality_status: unconfirmed` +
`requires_ns23_relabel`. Never modifies RC1 / production configs.

Outputs:
  probe_results.json
  probe_results.md
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import traceback

CASES = "project/evolve/experiments/sf2/out/set_cluster_deep_dive/selected_cases.json"
LADDERS = "project/evolve/experiments/sf2/set_probe_ladders.json"
OUT_JSON = "project/evolve/experiments/sf2/out/set_cluster_deep_dive/probe_results.json"
OUT_MD = "project/evolve/experiments/sf2/out/set_cluster_deep_dive/probe_results.md"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")

BASELINE_PROBES = {"simp", "simp_all", "aesop", "classical <;> aesop",
                   "simp [Set.ext_iff]", "simp_all [Set.ext_iff]"}


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


def cheap_key(probe):
    """cheap-first: plain simp < simp_all < aesop < multi-line/<;>/rw."""
    score = 0
    if "<;>" in probe:
        score += 2
    if "rw " in probe or "rw[" in probe:
        score += 2
    if "aesop" in probe:
        score += 3
    if "ext" in probe or "by_cases" in probe or "rcases" in probe:
        score += 1
    if "\n" in probe:
        score += 4
    return score


def gated_probes(ladder, case, enable_neg):
    shape = case.get("primary_goal_shape", "equality")
    gating = ladder.get("shape_gating", {})
    probes = []
    for fam, items in ladder.get("families", {}).items():
        allowed = gating.get(fam)
        if allowed is not None and shape not in allowed:
            continue
        for it in items:
            probes.append({"family": fam, "probe": it["probe"],
                           "label": it.get("label"), "source_inspired": False,
                           "risk": "none"})
    for it in ladder.get("source_inspired", {}).get(case["full_name"], []):
        probes.append({"family": "F_source_inspired", "probe": it["probe"],
                       "label": it.get("note"), "source_inspired": True,
                       "risk": it.get("theorem_specific_risk", "unknown")})
    if enable_neg:
        for it in ladder.get("negative_controls", []):
            probes.append({"family": "G_neg_control", "probe": it["probe"],
                           "label": it.get("label"), "source_inspired": False,
                           "risk": "diagnostic"})
    # dedupe identical probe strings (keep first/cheapest family), cheap-first sort
    seen, uniq = set(), []
    for pr in sorted(probes, key=lambda p: cheap_key(p["probe"])):
        if pr["probe"] in seen:
            continue
        seen.add(pr["probe"])
        uniq.append(pr)
    return uniq


def classify_gap(case, winning, minimality):
    """Decide the gap type from the winning probe + minimality battery."""
    if winning is None:
        # nothing solved: distinguish rw-heavy (needs deeper search) from lemma
        return "needs_deeper_search"
    wp = winning["probe"]
    base_solved = any(m["solved"] and m["probe"] in BASELINE_PROBES for m in (minimality or []))
    if wp in BASELINE_PROBES or base_solved:
        return "routing_gap"
    if winning.get("source_inspired") and ("rw " in wp or "rw[" in wp or "<;>" in wp and ("rcases" in wp or "by_cases" in wp)):
        return "search_depth_gap"
    if "rw " in wp or "rw[" in wp:
        return "search_depth_gap"
    # a short non-baseline tactic (e.g. simp [Set.ite]) solved it
    return "tactic_gap"


# ----------------------------- worker -------------------------------------
def worker(args):
    cases = json.load(open(args.cases))["selected"]
    ladder = json.load(open(args.ladders))
    case = cases[args.worker_theorem]
    probes = gated_probes(ladder, case, args.enable_neg_controls)[:args.max_probes_per_theorem]
    prefix = ladder.get("minimality_prefix_probes",
                        ["simp", "simp_all", "aesop", "classical <;> aesop"])

    res = {"full_name": case["full_name"], "file_path": case["file_path"],
           "cluster_id": case["cluster_id"], "primary_goal_shape": case["primary_goal_shape"],
           "live": False, "initial_goal": None, "num_probes": len(probes),
           "outcomes": [], "solved_by_probe": False, "winning_probe": None,
           "winning_family": None, "minimal_sufficient_probe": None,
           "minimality_status": "unconfirmed", "minimality_results": None,
           "requires_ns23_relabel": False, "classification": None, "setup_error": None}
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
                        getattr(rec, "error_message", None) if rec else None)

            winning = None
            for pr in probes:
                try:
                    fin, dead, err = apply(pr["probe"], args.timeout_per_probe)
                except _ProbeTimeout:
                    res["outcomes"].append({**pr, "solved": False,
                                            "outcome": "timeout_inner",
                                            "error": f"exceeded {args.timeout_per_probe}s"})
                    continue
                except Exception as e:
                    res["outcomes"].append({**pr, "solved": False, "outcome": "exception",
                                            "error": f"{type(e).__name__}: {str(e)[:160]}"})
                    continue
                oc = classify_outcome(err, fin)
                rec = {**pr, "solved": bool(fin), "outcome": oc}
                if err and not fin:
                    rec["error"] = err[:200]
                res["outcomes"].append(rec)
                if dead:
                    res["outcomes"].append({"probe": "<session_dead>", "solved": False,
                                            "outcome": "session_dead"})
                    break
                if fin and winning is None:
                    winning = pr
                    res["solved_by_probe"] = True
                    res["winning_probe"] = pr["probe"]
                    res["winning_family"] = pr["family"]
                    res["requires_ns23_relabel"] = True
                    # minimality battery (cheapest sufficient)
                    mini = []
                    for mp in prefix:
                        try:
                            f2, d2, _ = apply(mp, args.timeout_per_probe)
                        except Exception:
                            f2, d2 = False, False
                        mini.append({"probe": mp, "solved": bool(f2)})
                        if d2:
                            break
                    res["minimality_results"] = mini
                    solved_min = [m["probe"] for m in mini if m["solved"]]
                    res["minimal_sufficient_probe"] = (
                        solved_min[0] if solved_min else pr["probe"])
                    break  # first win is enough to classify the gap
            res["classification"] = classify_gap(case, winning, res["minimality_results"])
    except Exception as e:
        res["setup_error"] = f"{type(e).__name__}: {str(e)[:200]}\n" + \
            traceback.format_exc()[-300:]
        res["classification"] = "trace_insufficient"
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


# ----------------------------- driver -------------------------------------
def driver(args):
    cases = json.load(open(args.cases))["selected"]
    ladder = json.load(open(args.ladders))
    results = []

    def checkpoint():
        agg = {"num_theorems": len(cases), "num_live": sum(1 for r in results if r.get("live")),
               "num_solved": sum(1 for r in results if r.get("solved_by_probe")),
               "gap_histogram": _hist(results, "classification"),
               "results": results,
               "note": "Live LeanDojo probes. No solve is a confirmed win; NS23 "
                       "minimal-sufficient relabel + deterministic reproduction "
                       "required before any promotion. RC1/production configs untouched.",
               "ladders": args.ladders}
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        json.dump(agg, open(args.out_json, "w"), ensure_ascii=False, indent=2)
        return agg

    for idx in range(len(cases)):
        probes = gated_probes(ladder, cases[idx], args.enable_neg_controls)
        nprobes = min(len(probes), args.max_probes_per_theorem)
        # OS hard timeout: per-probe budget * probes + battery + Dojo-open margin
        hard = args.timeout_per_probe * (nprobes + len(
            ladder.get("minimality_prefix_probes", [1, 2, 3, 4])) + 1) + 90
        wout = f"/tmp/sf2_set_probe_t{idx}.json"
        if os.path.exists(wout):
            os.remove(wout)
        cmd = [sys.executable, _TIMEOUT_HELPER, str(hard), sys.executable,
               os.path.abspath(__file__), "--worker-theorem", str(idx),
               "--worker-out", wout, "--cases", args.cases, "--ladders", args.ladders,
               "--max-probes-per-theorem", str(args.max_probes_per_theorem),
               "--timeout-per-probe", str(args.timeout_per_probe)]
        if args.enable_neg_controls:
            cmd.append("--enable-neg-controls")
        print(f"[sf2:probe] ({idx+1}/{len(cases)}) {cases[idx]['full_name']} "
              f"probes={nprobes} hard={hard}s", flush=True)
        rc = subprocess.run(cmd, capture_output=True, text=True).returncode
        if os.path.exists(wout):
            try:
                rec = json.load(open(wout))
            except Exception as e:
                rec = {"full_name": cases[idx]["full_name"], "cluster_id": cases[idx]["cluster_id"],
                       "live": False, "solved_by_probe": False,
                       "classification": "trace_insufficient",
                       "setup_error": f"unreadable worker out: {e}"}
        else:
            rec = {"full_name": cases[idx]["full_name"], "cluster_id": cases[idx]["cluster_id"],
                   "live": False, "solved_by_probe": False,
                   "classification": "trace_insufficient",
                   "setup_error": f"no worker output (rc={rc}); OS-killed at {hard}s"}
        results.append(rec)
        print(f"            -> live={rec.get('live')} solved={rec.get('solved_by_probe')} "
              f"win={rec.get('winning_probe')} gap={rec.get('classification')}", flush=True)
        checkpoint()

    final = checkpoint()
    write_md(final, args.out_md)
    print(f"[sf2:probe] DONE theorems={final['num_theorems']} live={final['num_live']} "
          f"solved={final['num_solved']} gaps={final['gap_histogram']}")
    return 0


def _hist(results, key):
    h = {}
    for r in results:
        v = r.get(key)
        h[v] = h.get(v, 0) + 1
    return h


def write_md(agg, path):
    L = ["# SF2 Set Cluster — Live Probe Ladder Results", ""]
    L.append(f"- theorems: {agg['num_theorems']} | live: {agg['num_live']} | "
             f"solved: {agg['num_solved']}")
    L.append(f"- gap histogram: `{agg['gap_histogram']}`")
    L.append("")
    L.append("| theorem | shape | live | solved | gap | winning probe | min-sufficient |")
    L.append("|---|---|---|---|---|---|---|")
    for r in agg["results"]:
        L.append(f"| `{r['full_name']}` | {r.get('primary_goal_shape','')} | "
                 f"{r.get('live')} | {r.get('solved_by_probe')} | "
                 f"{r.get('classification')} | "
                 f"`{(r.get('winning_probe') or '')[:40]}` | "
                 f"`{(r.get('minimal_sufficient_probe') or '')[:30]}` |")
    L.append("")
    for r in agg["results"]:
        L.append(f"## `{r['full_name']}`")
        if r.get("setup_error"):
            L.append(f"- setup_error: {r['setup_error'][:200]}")
        if r.get("initial_goal"):
            L.append(f"- initial goal: `{r['initial_goal'][:200]}`")
        L.append(f"- classification: **{r.get('classification')}** | "
                 f"solved={r.get('solved_by_probe')} | win=`{r.get('winning_probe')}`")
        if r.get("minimality_results"):
            L.append(f"- minimality: {r['minimality_results']}")
        L.append("")
        L.append("| family | outcome | solved | risk | probe | error |")
        L.append("|---|---|---|---|---|---|")
        for o in r.get("outcomes", []):
            err = (o.get("error") or "").replace("\n", " ")[:50]
            L.append(f"| {o.get('family','')} | {o.get('outcome')} | {o.get('solved')} "
                     f"| {o.get('risk','')} | `{o.get('probe','')[:48]}` | {err} |")
        L.append("")
    L.append("> " + agg["note"])
    open(path, "w").write("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default=CASES)
    p.add_argument("--ladders", default=LADDERS)
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-md", default=OUT_MD)
    p.add_argument("--max-probes-per-theorem", type=int, default=40)
    p.add_argument("--timeout-per-probe", type=int, default=30)
    p.add_argument("--enable-neg-controls", action="store_true")
    p.add_argument("--worker-theorem", type=int, default=None)
    p.add_argument("--worker-out", default=None)
    args = p.parse_args(argv)
    if args.worker_theorem is not None:
        return worker(args)
    return driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
