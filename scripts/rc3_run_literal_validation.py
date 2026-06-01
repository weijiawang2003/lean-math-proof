#!/usr/bin/env python3
"""RC3 literal-wrapper validation runner.

Runs the production eval harness (`eval_rollout_all.main`) on the RC3 validation
theorem surface, for either the literal RC2 baseline or the RC3 candidate, and
emits a normalized per-theorem results JSON.

It reuses the SF1 in-process registration trick (patch `get_theorems` /
`list_theorem_sets` that `eval_rollout_all` imported) so NO file on disk is
modified and protected configs / production defaults are untouched. The only
difference between the RC2 and RC3 runs is `--strategy-config`.

Theorems are gathered from the manifest's `live_comparison_surface` sets (or an
explicit `--theorem-sets` list), restricted to rows with a non-null `file_path`,
and **deduped by full_name** (so `sx3_fresh_win` ⊂ `sx3_set_ite_holdout` is run
once). Roles are merged so each theorem keeps the most specific role.

Usage:
  python3 scripts/rc3_run_literal_validation.py \
    --manifest project/evolve/experiments/rc3_validation/validation_manifest.json \
    --policy rc2 \
    --strategy-config project/evolve/experiments/rc2_release/rc2_production_wrapper.json \
    --route-config project/evolve/routing/ns24_router.json \
    --out-json project/evolve/experiments/rc3_validation/out/literal_rc2_results.json
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

# role specificity for dedupe merge (higher = keep)
_ROLE_RANK = {
    "deferred_known": 5,
    "fresh_win": 4,
    "fresh_holdout": 3,
    "set_cluster_failure": 2,
    "negative_control": 1,
    "canonical_smoke": 0,
}


def _load_set_rows(path):
    obj = json.load(open(path))
    rows = []
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                rows.extend(v)
    elif isinstance(obj, list):
        rows = obj
    return rows


def _gather(manifest_path, explicit_sets):
    man = json.load(open(manifest_path))
    if explicit_sets:
        set_files = explicit_sets
    else:
        names = set(man.get("live_comparison_surface", {}).get("sets", []))
        set_files = [t["file"] for t in man["theorem_sets"] if t["name"] in names]
    by_name = {}
    for sf in set_files:
        for r in _load_set_rows(sf):
            fp = r.get("file_path")
            fn = r.get("full_name")
            if not fp or not fn:
                continue
            prev = by_name.get(fn)
            if prev is None or _ROLE_RANK.get(r.get("role", ""), -1) > _ROLE_RANK.get(prev.get("role", ""), -1):
                by_name[fn] = r
    return list(by_name.values()), set_files


def _build_configs(rows):
    from core_types import TheoremConfig
    fields = {f.name for f in dataclasses.fields(TheoremConfig)}
    tcs = []
    for r in rows:
        kwargs = {k: v for k, v in r.items() if k in fields and v is not None}
        kwargs.setdefault("file_path", r["file_path"])
        kwargs.setdefault("full_name", r["full_name"])
        tcs.append(TheoremConfig(**{k: v for k, v in kwargs.items() if k in fields}))
    return tcs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--theorem-sets", nargs="*", default=None,
                   help="explicit theorem-set JSON files (override manifest surface)")
    p.add_argument("--policy", required=True, help="label, e.g. rc2 / rc3_candidate")
    p.add_argument("--strategy-config", required=True)
    p.add_argument("--route-config", required=True)
    p.add_argument("--policy-type", default="hybrid_evolved")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--register-name", default=None)
    args = p.parse_args(argv)

    rows, set_files = _gather(args.manifest, args.theorem_sets)
    tcs = _build_configs(rows)
    role_by = {r["full_name"]: r.get("role") for r in rows}
    if not tcs:
        print("[rc3_run] ERROR: no runnable theorems", file=sys.stderr)
        return 3

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.out_json), f"{args.policy}_runs")
    os.makedirs(out_dir, exist_ok=True)
    name = args.register_name or f"rc3val_{args.policy}"

    import eval_rollout_all as E
    _get, _list = E.get_theorems, E.list_theorem_sets
    E.list_theorem_sets = lambda: (list(_list()) + [name]) if name not in _list() else list(_list())
    E.get_theorems = lambda n: tcs if n == name else _get(n)

    fwd = ["eval_rollout_all.py", "--theorem-set", name,
           "--policy-type", args.policy_type,
           "--route-config", args.route_config,
           "--strategy-config", args.strategy_config,
           "--top-k", str(args.top_k), "--max-steps", str(args.max_steps),
           "--out-dir", out_dir]
    print(f"[rc3_run] policy={args.policy} theorems={len(tcs)} -> {out_dir}")
    print(f"[rc3_run] sets: {set_files}")
    sys.argv = fwd
    rc = 0
    try:
        ret = E.main()
        rc = ret if isinstance(ret, int) else 0
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 0

    # locate freshest metrics.json under out_dir
    metrics_files = sorted(glob.glob(os.path.join(out_dir, "eval-*", "metrics.json")),
                           key=os.path.getmtime)
    if not metrics_files:
        print("[rc3_run] ERROR: no metrics.json produced", file=sys.stderr)
        return 4
    mpath = metrics_files[-1]
    metrics = json.load(open(mpath))
    run_dir = os.path.dirname(mpath)
    trace_path = os.path.join(run_dir, "traces.jsonl")

    per = []
    for t in metrics.get("per_theorem", []):
        fn = t["full_name"]
        per.append({
            "full_name": fn,
            "file_path": t.get("file_path"),
            "role": role_by.get(fn),
            "available": t.get("available"),
            "finished": bool(t.get("finished")),
            "has_error": bool(t.get("has_error")),
            "error_message": t.get("error_message"),
            "num_steps": t.get("num_steps"),
            "tactics_used": t.get("tactics_used"),
            "winning_tactic": t.get("winning_tactic"),
            "winning_tactic_origin": t.get("winning_tactic_origin"),
            "skip_reason": t.get("skip_reason"),
            "open_flake": (not t.get("available")) and t.get("skip_reason") not in (None, ""),
        })
    out = {
        "policy": args.policy,
        "strategy_config": args.strategy_config,
        "route_config": args.route_config,
        "policy_type": args.policy_type,
        "top_k": args.top_k, "max_steps": args.max_steps,
        "theorem_sets": set_files,
        "num_theorems": len(per),
        "num_finished": sum(1 for r in per if r["finished"]),
        "num_available": sum(1 for r in per if r["available"]),
        "return_code": rc,
        "metrics_path": mpath,
        "trace_path": trace_path,
        "run_dir": run_dir,
        "per_theorem": per,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)
    print(f"[rc3_run] wrote {args.out_json}: finished {out['num_finished']}/{out['num_theorems']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
