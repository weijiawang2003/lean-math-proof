#!/usr/bin/env python3
"""RC4D Part 11 — full canonical floor benchmark.

Runs BOTH literal RC2 and the RC4D candidate wrapper over the full canonical floor sets
(demo_v1, nat_defs_medium, nat_defs_large_v5) through the real eval_rollout_all search at the
RC2-release verification config (hybrid_evolved, top-k 8, max-steps 8), and compares solved
counts. RC4D promotion requires NO regression on any floor (RC4D solved ≥ RC2 solved per set).
Each (floor_set, wrapper) runs in a worker subprocess under an OS hard timeout; the registered
theorem set is used directly (full set, not a sample).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
FLOORS = ("demo_v1", "nat_defs_medium", "nat_defs_large_v5")


def _p(*a):
    return os.path.join(_REPO, *a)


def worker(args):
    out = {"set": args.floor_set, "wrapper": args.wrapper}
    try:
        sys.path.insert(0, _REPO)
        import eval_rollout_all as E
        os.makedirs(args.out_dir, exist_ok=True)
        sys.argv = ["eval_rollout_all.py", "--theorem-set", args.floor_set,
                    "--policy-type", args.policy_type, "--route-config", args.route_config,
                    "--strategy-config", args.wrapper, "--top-k", str(args.top_k),
                    "--max-steps", str(args.max_steps), "--out-dir", args.out_dir]
        try:
            E.main()
        except SystemExit:
            pass
        mfiles = sorted(glob.glob(os.path.join(args.out_dir, "eval-*", "metrics.json")),
                        key=os.path.getmtime)
        if mfiles:
            m = json.load(open(mfiles[-1]))
            per = m.get("per_theorem", [])
            out["n"] = len(per)
            out["available"] = sum(1 for r in per if r.get("available"))
            out["solved"] = sum(1 for r in per if r.get("finished"))
            out["solved_names"] = sorted(r["full_name"] for r in per if r.get("finished"))
        else:
            out["error"] = "no metrics"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    json.dump(out, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


def run_one(floor_set, wrapper, tag, args):
    wout = _p(args.out_dir_root, f"{tag}_{floor_set}.json")
    ckpt = wout
    if os.path.exists(ckpt):
        try:
            return json.load(open(ckpt))
        except Exception:
            pass
    cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout),
           sys.executable, os.path.abspath(__file__), "--worker", "--worker-out", wout,
           "--floor-set", floor_set, "--wrapper", wrapper,
           "--out-dir", _p(args.out_dir_root, f"runs_{tag}_{floor_set}"),
           "--route-config", args.route_config, "--policy-type", args.policy_type,
           "--top-k", str(args.top_k), "--max-steps", str(args.max_steps)]
    print(f"[rc4d-floor] {tag} {floor_set} ...", flush=True)
    subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.load(open(wout))
    except Exception:
        return {"set": floor_set, "wrapper": wrapper, "error": "worker_unreadable"}


def driver(args):
    os.makedirs(_p(args.out_dir_root), exist_ok=True)
    rows = []
    floors_ok = True
    for fs in FLOORS:
        rc2 = run_one(fs, args.rc2_wrapper, "rc2", args)
        rc4d = run_one(fs, args.rc4d_wrapper, "rc4d", args)
        rc2_solved = rc2.get("solved")
        rc4d_solved = rc4d.get("solved")
        rc2_set = set(rc2.get("solved_names", []))
        rc4d_set = set(rc4d.get("solved_names", []))
        regressed = sorted(rc2_set - rc4d_set)
        gained = sorted(rc4d_set - rc2_set)
        floor_pass = (rc4d_solved is not None and rc2_solved is not None
                      and rc4d_solved >= rc2_solved and not regressed)
        floors_ok = floors_ok and floor_pass
        rows.append({"floor": fs, "n": rc2.get("n"),
                     "rc2_solved": rc2_solved, "rc4d_solved": rc4d_solved,
                     "delta": (rc4d_solved - rc2_solved) if (rc4d_solved is not None and rc2_solved is not None) else None,
                     "regressed_theorems": regressed, "gained_theorems": gained,
                     "floor_pass": floor_pass,
                     "rc2_error": rc2.get("error"), "rc4d_error": rc4d.get("error")})

    out = {"generated_by": "scripts/rc4d_full_floor_benchmark.py",
           "rc2_wrapper": args.rc2_wrapper, "rc4d_wrapper": args.rc4d_wrapper,
           "policy_type": args.policy_type, "top_k": args.top_k, "max_steps": args.max_steps,
           "floors": rows, "all_floors_pass": floors_ok,
           "total_regressions": sum(len(r["regressed_theorems"]) for r in rows)}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC4D full canonical floor benchmark", "",
          f"- config: {args.policy_type}, top-k {args.top_k}, max-steps {args.max_steps}",
          f"- **all floors pass (RC4D ≥ RC2, no regression): {floors_ok}**",
          f"- total regressions: {out['total_regressions']}", "",
          "| floor | n | RC2 | RC4D | delta | regressed | gained | pass |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['floor']} | {r['n']} | {r['rc2_solved']} | {r['rc4d_solved']} | "
                  f"{r['delta']} | {len(r['regressed_theorems'])} | {len(r['gained_theorems'])} | "
                  f"{r['floor_pass']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4d-floor] all_pass={floors_ok} rows={[(r['floor'], r['rc2_solved'], r['rc4d_solved']) for r in rows]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--floor-set")
    ap.add_argument("--wrapper")
    ap.add_argument("--out-dir")
    ap.add_argument("--rc2-wrapper", default="project/evolve/experiments/rc2_release/rc2_production_wrapper.json")
    ap.add_argument("--rc4d-wrapper", default="project/evolve/experiments/rc4_candidates/composition_rc4d/rc4d_candidate_wrapper.json")
    ap.add_argument("--route-config", default="project/evolve/routing/ns24_router.json")
    ap.add_argument("--policy-type", default="hybrid_evolved")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--out-json", default="project/evolve/experiments/rc4_candidates/composition_rc4d/out/full_floor_benchmark.json")
    ap.add_argument("--out-md", default="project/evolve/experiments/rc4_candidates/composition_rc4d/out/full_floor_benchmark.md")
    ap.add_argument("--out-dir-root", default="project/evolve/experiments/rc4_candidates/composition_rc4d/out/floor_bench")
    ap.add_argument("--hard-timeout", type=int, default=2400)
    args = ap.parse_args()
    if args.worker:
        sys.exit(worker(args))
    driver(args)


if __name__ == "__main__":
    main()
