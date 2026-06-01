#!/usr/bin/env python3
"""RC2 Parts 3/4 — full-wrapper benchmark runner for RC1 or RC2 candidate.

Runs eval_rollout_all.py (mode = full_wrapper_eval) on every benchmark surface with
the given --strategy-config (RC1 production wrapper, or the composed RC2 candidate
wrapper). Registered surfaces use eval_rollout_all --theorem-set directly; file
surfaces use the non-invasive runtime registration via scripts/sf1_run_eval.py. Both
paths produce eval-<hash>/metrics.json, parsed on the authoritative `finished` key.

  --policy rc1|rc2_candidate    (label only; behavior set by --strategy-config)
  --reuse <results.json>        merge prior literal-RC1 per-surface results (skip rerun
                                of surfaces present there, identical command)

NEVER modifies RC1 / NS24 configs. RC2 wrapper is a separate composed artifact.

Outputs (under derived dir of --out):
  <out>                         merged per-surface + per-theorem results
  <stem>_commands.sh
  <stem>_logs/<surface>.log
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


def _latest_metrics(out_dir):
    best, mt = None, -1
    for root, _d, files in os.walk(out_dir):
        if "metrics.json" in files:
            pth = os.path.join(root, "metrics.json")
            m = os.path.getmtime(pth)
            if m > mt:
                best, mt = pth, m
    return best


def _parse_metrics(mpath):
    rows = []
    if not mpath:
        return rows
    m = json.load(open(mpath))
    for t in m.get("per_theorem", []):
        rows.append({"full_name": t.get("full_name"), "file_path": t.get("file_path"),
                     "finished": bool(t.get("finished")), "available": t.get("available"),
                     "has_error": t.get("has_error"), "num_steps": t.get("num_steps"),
                     "winning_tactic": t.get("winning_tactic"),
                     "winning_tactic_origin": t.get("winning_tactic_origin"),
                     "goal_shape": t.get("goal_shape"),
                     "error_message": (t.get("error_message") or "")[:160]})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--policy", default="rc1", choices=["rc1", "rc2_candidate"])
    p.add_argument("--route-config", default="project/evolve/routing/ns24_router.json")
    p.add_argument("--strategy-config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--timeout-per-surface", type=int, default=1800)
    p.add_argument("--reuse", default=None,
                   help="prior results.json to reuse surfaces from (identical command)")
    p.add_argument("--only", default=None, help="comma list of surface names to run")
    args = p.parse_args(argv)

    manifest = json.load(open(args.manifest))
    surfaces = manifest["surfaces"]
    if args.only:
        keep = set(args.only.split(","))
        surfaces = [s for s in surfaces if s["name"] in keep]

    reuse = {}
    if args.reuse and os.path.exists(args.reuse):
        rprev = json.load(open(args.reuse))
        for s in rprev.get("per_surface", []):
            reuse[s["name"]] = s

    stem = os.path.splitext(args.out)[0]
    log_dir = stem + "_logs"
    os.makedirs(log_dir, exist_ok=True)
    base_out = os.path.join(os.path.dirname(args.out), f"{args.policy}_runs")
    os.makedirs(base_out, exist_ok=True)

    commands = ["#!/usr/bin/env bash", f"# RC2 benchmark — policy={args.policy}. "
                "RC1/NS24 untouched.", "set -uo pipefail", f"cd {_REPO}", ""]
    per_surface = []
    for s in surfaces:
        name = s["name"]
        # negative controls: non-live (file_path null) -> skip live, handled by off-gate scan
        if s["role"] == "negative_control" and s.get("live_runnable", 0) == 0:
            per_surface.append({"name": name, "role": s["role"], "skipped": "dry_offgate_only",
                                "num_theorems": 0, "num_finished": 0, "theorems": []})
            continue
        if name in reuse:
            rep = dict(reuse[name])
            rep["reused"] = True
            per_surface.append(rep)
            print(f"[rc2:{args.policy}] REUSE {name} finished="
                  f"{rep.get('num_finished')}/{rep.get('num_theorems')}", flush=True)
            continue

        run_out = os.path.join(base_out, name)
        os.makedirs(run_out, exist_ok=True)
        if s["kind"] == "registered":
            cmd = [sys.executable, "eval_rollout_all.py", "--theorem-set", name,
                   "--policy-type", "hybrid_evolved", "--route-config", args.route_config,
                   "--strategy-config", args.strategy_config,
                   "--top-k", str(args.top_k), "--max-steps", str(args.max_steps),
                   "--out-dir", run_out]
        else:
            cmd = [sys.executable, os.path.join("scripts", "sf1_run_eval.py"),
                   "--theorem-set-file", s["path_or_registered_name"],
                   "--register-name", f"rc2bench_{name}",
                   "--", "--policy-type", "hybrid_evolved", "--route-config", args.route_config,
                   "--strategy-config", args.strategy_config,
                   "--top-k", str(args.top_k), "--max-steps", str(args.max_steps),
                   "--out-dir", run_out]
        commands += [f"# {name} ({s['role']})", " ".join(cmd), ""]
        wrapped = [sys.executable, _TIMEOUT_HELPER, str(args.timeout_per_surface)] + cmd
        log_path = os.path.join(log_dir, f"{name}.log")
        print(f"[rc2:{args.policy}] running {name} ({s['role']}, size={s.get('size')}) ...",
              flush=True)
        t0 = time.time()
        with open(log_path, "w") as lf:
            rc = subprocess.run(wrapped, stdout=lf, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        mpath = _latest_metrics(run_out)
        rows = _parse_metrics(mpath)
        rep = {"name": name, "role": s["role"], "kind": s["kind"],
               "return_code": rc, "elapsed_sec": round(dt, 1), "metrics_path": mpath,
               "log_path": log_path, "num_theorems": len(rows),
               "num_finished": sum(1 for r in rows if r["finished"]),
               "contains_set_ite": s.get("contains_set_ite"),
               "expected_rc1": s.get("expected_rc1"),
               "expected_rc2_delta": s.get("expected_rc2_delta"),
               "theorems": rows, "reused": False}
        per_surface.append(rep)
        print(f"            -> rc={rc} {dt:.0f}s finished={rep['num_finished']}/{len(rows)}",
              flush=True)

    out = {"policy": args.policy, "strategy_config": args.strategy_config,
           "route_config": args.route_config, "top_k": args.top_k,
           "max_steps": args.max_steps, "mode": "full_wrapper_eval",
           "per_surface": per_surface,
           "totals": {"surfaces": len([s for s in per_surface if not s.get("skipped")]),
                      "theorems": sum(s.get("num_theorems", 0) for s in per_surface),
                      "finished": sum(s.get("num_finished", 0) for s in per_surface)},
           "note": f"policy={args.policy}; full-wrapper eval_rollout_all; authoritative "
                   "`finished` key; RC1/NS24 untouched."}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    open(stem + "_commands.sh", "w").write("\n".join(commands))
    print(f"[rc2:{args.policy}] DONE totals={out['totals']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
