#!/usr/bin/env python3
"""SF2 Part 4 — live probe runner for the SF1 Multiset failure(s).

Runs each probe tactic as a SINGLE explicit tactic block from the theorem's
initial Lean state via the repo's LeanDojo helpers (env.make_repo /
env.make_theorem / env.run_transition + lean_dojo.Dojo), restricted to the 3
SF1 Multiset theorems. It NEVER modifies the RC1 wrapper or any production
config and NEVER adds probes to production.

If LeanDojo is unavailable it records the exact error and still emits the planned
per-probe commands (no fabricated results). Any solve is marked
``minimality_status: unconfirmed`` + ``requires_ns23_relabel: true``.

Output:
  project/evolve/experiments/sf2/out/multiset_seed/probe_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

CASES = "project/evolve/experiments/sf2/out/multiset_seed/failure_cases.json"
LADDER = "project/evolve/experiments/sf2/multiset_probe_ladder.json"
OUT = "project/evolve/experiments/sf2/out/multiset_seed/probe_results.json"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fill(probe, thm):
    return (probe.replace("{var2}", thm.get("secondary_var", "n"))
                 .replace("{var}", thm.get("primary_var", "s")))


def _build_probe_list(thm_entry, base_probes, max_probes):
    probes = []
    for p in (thm_entry.get("specific_probes", []) + base_probes):
        f = _fill(p, thm_entry)
        if f not in probes:
            probes.append(f)
    return probes[:max_probes]


def _run_one_theorem(env, TheoremConfig, Dojo, repo, case, probes, prefix_probes, only_first_n=None):
    fp, fn = case.get("file_path"), case["full_name"]
    outcomes = []
    try:
        cfg = TheoremConfig(file_path=fp, full_name=fn)
        thm = env.make_theorem(repo, cfg)
    except Exception as e:
        return outcomes, None, f"make_theorem failed: {type(e).__name__}: {str(e)[:200]}"
    try:
        with Dojo(thm) as (dojo, state0):
            def apply(tac):
                out = env.run_transition(dojo, thm, state0, tac)
                return bool(getattr(out, "is_finished", False)), bool(getattr(out, "session_dead", False)), \
                    getattr(getattr(out, "record", None), "error_message", None)
            for tac in probes:
                fin, dead, err = apply(tac)
                outcomes.append({"probe": tac, "solved": fin,
                                 **({"error": err[:200]} if (err and not fin) else {})})
                if dead:
                    outcomes.append({"probe": "<session_dead>", "solved": False,
                                     "error": "Lean REPL crashed; remaining probes untested"})
                    return outcomes, None, "session_dead"
                if fin:
                    mini = []
                    for mp in prefix_probes:
                        try:
                            f2, d2, _ = apply(mp)
                        except Exception:
                            f2 = False
                        mini.append({"probe": mp, "solved": f2})
                    return outcomes, {"probe": tac, "minimality": mini}, None
        return outcomes, None, None
    except Exception as e:
        return outcomes, None, f"dojo error: {type(e).__name__}: {str(e)[:300]}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SF2: run probe ladder (live).")
    p.add_argument("--cases", default=CASES)
    p.add_argument("--ladder", default=LADDER)
    p.add_argument("--out", default=OUT)
    p.add_argument("--max-probes-per-theorem", type=int, default=20)
    p.add_argument("--only-failures", action="store_true",
                   help="Run probes only for theorems with rc1_solved=false.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cases = json.load(open(args.cases)).get("cases", [])
    ladder = json.load(open(args.ladder))
    base = ladder.get("base_probes", [])
    prefix = ladder.get("minimality_prefix_probes", ["simp", "simp_all", "aesop"])
    by_name = {t["full_name"]: t for t in ladder.get("theorems", [])}

    planned = {}
    for c in cases:
        ent = by_name.get(c["full_name"], {"primary_var": "s", "secondary_var": "n"})
        planned[c["full_name"]] = _build_probe_list(ent, base, args.max_probes_per_theorem)

    live = {"attempted": True, "available": False, "setup_error": None}
    env = TheoremConfig = Dojo = repo = None
    try:
        sys.path.insert(0, _REPO)
        import env as _env
        from core_types import TheoremConfig as _TC
        from lean_dojo import Dojo as _Dojo
        env, TheoremConfig, Dojo = _env, _TC, _Dojo
        repo = env.make_repo()
        live["available"] = True
    except Exception as e:
        live["setup_error"] = f"{type(e).__name__}: {e}\n" + traceback.format_exc()[-800:]

    results = []
    if live["available"]:
        for c in cases:
            if args.only_failures and c.get("rc1_solved"):
                results.append({"full_name": c["full_name"], "file_path": c.get("file_path"),
                                "skipped": "rc1 already solved (only-failures mode)",
                                "probe_outcomes": [], "solved_probe": None})
                continue
            ent = by_name.get(c["full_name"], {"primary_var": "s", "secondary_var": "n"})
            probes = planned[c["full_name"]]
            prefix_f = [_fill(p, ent) for p in prefix]
            outcomes, solved, err = _run_one_theorem(
                env, TheoremConfig, Dojo, repo, c, probes, prefix_f)
            rec = {"full_name": c["full_name"], "file_path": c.get("file_path"),
                   "rc1_solved": c.get("rc1_solved"),
                   "probe_outcomes": outcomes, "run_error": err}
            if solved:
                rec["solved_probe"] = {"full_name": c["full_name"], "probe": solved["probe"],
                                       "solved": True, "minimality_status": "unconfirmed",
                                       "requires_ns23_relabel": True,
                                       "minimality_prefix_results": solved["minimality"]}
            else:
                rec["solved_probe"] = None
            results.append(rec)
    else:
        for c in cases:
            results.append({"full_name": c["full_name"], "file_path": c.get("file_path"),
                            "probe_outcomes": [], "solved_probe": None,
                            "run_error": "live infra unavailable; see live.setup_error"})

    out = {"live": live, "planned_probes": planned, "results": results,
           "note": "No probe result is a confirmed production win; any solve needs NS23 minimal-"
                   "sufficient relabel + deterministic reproduction before promotion. RC1/production "
                   "configs were not modified."}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)

    n_solved = sum(1 for r in results if r.get("solved_probe"))
    print(f"[sf2:probe] live_available={live['available']} theorems={len(results)} "
          f"probe_solved={n_solved} -> {args.out}")
    if not live["available"]:
        print(f"[sf2:probe] live infra unavailable: {(live['setup_error'] or '')[:160]}")
    for r in results:
        sp = r.get("solved_probe")
        nsolv = sum(1 for o in (r.get('probe_outcomes') or []) if o.get('solved'))
        print(f"  {r['full_name']}: solved_probe={sp['probe'] if sp else None} "
              f"(#solved_outcomes={nsolv}) run_error={r.get('run_error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
