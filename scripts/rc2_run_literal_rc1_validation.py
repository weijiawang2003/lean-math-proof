#!/usr/bin/env python3
"""RC2 Part 3 — literal RC1 baseline run (NOT a proxy).

Runs the unmodified RC1 production wrapper + NS24 router over the validation Set
sets via the non-invasive runtime-registration path
(scripts/sf1_run_eval.py -> eval_rollout_all.py), exactly the RC1 production
command:

  --policy-type hybrid_evolved --route-config <ns24_router.json>
  --strategy-config <rc1_production_wrapper.json> --top-k 8 --max-steps 8

Parses each run's eval-<hash>/metrics.json `per_theorem` list (authoritative
`finished` key) into a merged per-theorem result. Writes the exact commands to a
replayable .sh and keeps run logs.

NEVER modifies RC1 / NS24 configs.

Outputs (under --out-dir):
  literal_rc1_results.json      merged per-theorem (deduped by full_name + per-set)
  literal_rc1_commands.sh
  literal_rc1_logs/<set>.log
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")

# sets to run literal RC1 on (live-runnable Set sets)
RC1_SETS = ["set_ite_known_wins", "set_ite_selected_failures", "set_ite_fresh_holdout"]


def _latest_metrics(out_dir):
    """Find the newest eval-*/metrics.json under out_dir."""
    best, best_mt = None, -1
    for root, _dirs, files in os.walk(out_dir):
        if "metrics.json" in files:
            pth = os.path.join(root, "metrics.json")
            mt = os.path.getmtime(pth)
            if mt > best_mt:
                best, best_mt = pth, mt
    return best


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/validation_manifest.json")
    p.add_argument("--route-config", default="project/evolve/routing/ns24_router.json")
    p.add_argument("--strategy-config",
                   default="project/evolve/experiments/rc1/rc1_production_wrapper.json")
    p.add_argument("--out-dir",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/out/literal_rc1")
    p.add_argument("--results-json",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/out/literal_rc1_results.json")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--timeout-per-set", type=int, default=1200)
    p.add_argument("--sets", default=",".join(RC1_SETS))
    args = p.parse_args(argv)

    manifest = json.load(open(args.manifest))
    ts_dir = manifest["theorem_sets_dir"]
    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.join(os.path.dirname(args.results_json), "literal_rc1_logs")
    os.makedirs(log_dir, exist_ok=True)

    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    commands = ["#!/usr/bin/env bash",
                "# Literal RC1 validation — replayable commands. RC1/NS24 untouched.",
                "set -uo pipefail", f"cd {_REPO}", ""]
    per_set = {}
    for sname in sets:
        set_file = os.path.join(ts_dir, f"{sname}.json")
        run_out = os.path.join(args.out_dir, sname)
        os.makedirs(run_out, exist_ok=True)
        register = f"rc2_{sname}"
        cmd = [sys.executable, os.path.join("scripts", "sf1_run_eval.py"),
               "--theorem-set-file", set_file, "--register-name", register,
               "--", "--policy-type", "hybrid_evolved",
               "--route-config", args.route_config,
               "--strategy-config", args.strategy_config,
               "--top-k", str(args.top_k), "--max-steps", str(args.max_steps),
               "--out-dir", run_out]
        wrapped = [sys.executable, _TIMEOUT_HELPER, str(args.timeout_per_set)] + cmd
        commands.append(f"# {sname}")
        commands.append(" ".join(cmd))
        commands.append("")
        log_path = os.path.join(log_dir, f"{sname}.log")
        print(f"[rc2:rc1] running literal RC1 on {sname} (timeout {args.timeout_per_set}s) ...",
              flush=True)
        t0 = time.time()
        with open(log_path, "w") as lf:
            rc = subprocess.run(wrapped, stdout=lf, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        mpath = _latest_metrics(run_out)
        rows = []
        if mpath:
            m = json.load(open(mpath))
            for t in m.get("per_theorem", []):
                rows.append({
                    "full_name": t.get("full_name"),
                    "file_path": t.get("file_path"),
                    "finished": bool(t.get("finished")),
                    "available": t.get("available"),
                    "has_error": t.get("has_error"),
                    "num_steps": t.get("num_steps"),
                    "winning_tactic": t.get("winning_tactic"),
                    "winning_tactic_origin": t.get("winning_tactic_origin"),
                    "error_message": (t.get("error_message") or "")[:200],
                    "goal_shape": t.get("goal_shape"),
                })
        per_set[sname] = {"return_code": rc, "elapsed_sec": round(dt, 1),
                          "metrics_path": mpath, "log_path": log_path,
                          "num_theorems": len(rows),
                          "num_finished": sum(1 for r in rows if r["finished"]),
                          "theorems": rows}
        print(f"            -> rc={rc} elapsed={dt:.0f}s finished="
              f"{per_set[sname]['num_finished']}/{len(rows)} metrics={bool(mpath)}",
              flush=True)

    # merged per-theorem (union; record which set + finished)
    merged = {}
    for sname, rep in per_set.items():
        for r in rep["theorems"]:
            fn = r["full_name"]
            cur = merged.setdefault(fn, {"full_name": fn, "file_path": r["file_path"],
                                         "literal_rc1_finished": False,
                                         "winning_tactic": None, "winning_tactic_origin": None,
                                         "in_sets": [], "per_set_finished": {}})
            cur["in_sets"].append(sname)
            cur["per_set_finished"][sname] = r["finished"]
            if r["finished"]:
                cur["literal_rc1_finished"] = True
                cur["winning_tactic"] = r["winning_tactic"]
                cur["winning_tactic_origin"] = r["winning_tactic_origin"]
            cur.setdefault("error_message", r.get("error_message"))
            cur.setdefault("goal_shape", r.get("goal_shape"))

    out = {
        "rc1_command": {"policy_type": "hybrid_evolved", "route_config": args.route_config,
                        "strategy_config": args.strategy_config, "top_k": args.top_k,
                        "max_steps": args.max_steps},
        "sets_run": sets,
        "per_set_summary": {s: {"finished": per_set[s]["num_finished"],
                                "total": per_set[s]["num_theorems"],
                                "return_code": per_set[s]["return_code"],
                                "elapsed_sec": per_set[s]["elapsed_sec"]}
                            for s in sets},
        "per_set": per_set,
        "merged_theorems": list(merged.values()),
        "note": "Literal RC1 via unmodified rc1_production_wrapper.json + ns24_router.json. "
                "Authoritative solved flag = per_theorem `finished`. RC1/NS24 untouched.",
    }
    os.makedirs(os.path.dirname(args.results_json), exist_ok=True)
    json.dump(out, open(args.results_json, "w"), ensure_ascii=False, indent=2)
    cmd_sh = os.path.join(os.path.dirname(args.results_json), "literal_rc1_commands.sh")
    open(cmd_sh, "w").write("\n".join(commands))
    print(f"[rc2:rc1] DONE sets={sets} -> {args.results_json}")
    for s in sets:
        ss = out["per_set_summary"][s]
        print(f"   {s:28s} finished={ss['finished']}/{ss['total']} rc={ss['return_code']} "
              f"({ss['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
