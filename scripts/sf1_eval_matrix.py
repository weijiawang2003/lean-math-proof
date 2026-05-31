#!/usr/bin/env python3
"""SF1 stage (e): eval-matrix wiring + live-eval unblocker.

Builds raw / ns9 / rc1 / experimental evaluation for SF1 batches and can now run
a REAL RC1/NS9 smoke eval by:
  1. joining each batch to backfilled file_paths (frontier_with_paths.jsonl),
  2. emitting eval-compatible theorem-set files under
     project/evolve/experiments/sf1/theorem_sets/, and
  3. delegating to eval_rollout_all.py via scripts/sf1_run_eval.py, which registers
     the SF1 set by name at runtime (no edit to eval_rollout_all.py / tasks.py /
     protected configs).

Principles: never fake results; --dry-run emits commands only; --run-real runs
ONLY supported, file_path-complete configs; records rc / logs / parsed metrics.

Outputs:
  runnable_theorem_sets_manifest.json
  eval_matrix_manifest.json
  eval_matrix_results.json
  eval_commands.sh
  <out-dir>/runs/<set>__<policy>/...  (run dirs + *.stdout/*.stderr logs)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import sf1_common as C
    _read = C.read_json_or_jsonl
    _ensure = C.ensure_parent_dir
except Exception:  # pragma: no cover
    def _read(path):
        rows = []
        if not os.path.isfile(path):
            return rows
        with open(path, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        try:
            obj = json.loads(txt)
            return obj if isinstance(obj, list) else [obj]
        except json.JSONDecodeError:
            for line in txt.splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _ensure(path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        return path

RC1_WRAPPER = "project/evolve/experiments/rc1/rc1_production_wrapper.json"
NS24_ROUTER = "project/evolve/routing/ns24_router.json"
EVAL_SCRIPT = "eval_rollout_all.py"
RUN_EVAL = "scripts/sf1_run_eval.py"
TIMEOUT_HELPER = "scripts/run_with_timeout.py"
TS_DIR = "project/evolve/experiments/sf1/theorem_sets"
OUT_REAL = "project/evolve/experiments/sf1/out/real"

# batch -> runnable set name / file
RUNNABLE_TARGETS = [
    ("sf1_multiset_holdout", "sf1_multiset_holdout_runnable"),
    ("sf1_balanced_mini", "sf1_balanced_mini_runnable"),
    ("sf1_frontier_all", "sf1_frontier_runnable_subset"),
]


def _name(row):
    if isinstance(row, dict):
        for k in ("decl_name", "full_name", "name", "theorem"):
            v = row.get(k)
            if isinstance(v, str) and v:
                return v
    return None


def load_path_map(frontier_with_paths):
    """name -> {file_path, namespace, line} from backfilled frontier."""
    m = {}
    for r in _read(frontier_with_paths):
        n = _name(r)
        fp = r.get("file_path") if isinstance(r, dict) else None
        if n and fp:
            m[n] = {"file_path": fp, "namespace": r.get("namespace"),
                    "line": r.get("line")}
    return m


def detect_eval_capabilities():
    caps = {"script_present": os.path.isfile(EVAL_SCRIPT),
            "run_eval_wrapper_present": os.path.isfile(RUN_EVAL),
            "help_ok": False, "flags": []}
    if not caps["script_present"]:
        return caps
    try:
        r = subprocess.run([sys.executable, EVAL_SCRIPT, "--help"],
                           capture_output=True, text=True, timeout=120)
        caps["help_ok"] = r.returncode == 0
        txt = (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # pragma: no cover
        txt = f"<help failed: {e}>"
    for flag in ["--theorem-set", "--policy-type", "--route-config",
                 "--strategy-config", "--ckpt-dir", "--top-k", "--max-steps", "--out-dir"]:
        if flag in txt:
            caps["flags"].append(flag)
    return caps


def discover_ns9_config():
    for cand in ["project/evolve/ns9_runs/baseline/strategy_config.json"]:
        if os.path.isfile(cand):
            return cand
    hits = sorted(glob.glob("project/evolve/ns9_runs/**/strategy_config.json", recursive=True))
    return hits[0] if hits else None


def prepare_runnable_sets(path_map, batches_dir, max_theorems_per_batch=0):
    """Write eval-compatible theorem-set files; return manifest list."""
    os.makedirs(TS_DIR, exist_ok=True)
    manifest = []
    for batch_name, eval_name in RUNNABLE_TARGETS:
        bpath = os.path.join(batches_dir, batch_name + ".jsonl")
        if not os.path.isfile(bpath):
            continue
        rows = _read(bpath)
        kept, dropped = [], 0
        for r in rows:
            n = _name(r)
            info = path_map.get(n)
            if info and info.get("file_path"):
                kept.append({"file_path": info["file_path"], "full_name": n,
                             "namespace": info.get("namespace")})
            else:
                dropped += 1
        if max_theorems_per_batch and len(kept) > max_theorems_per_batch:
            # deterministic: batches are already deterministically ordered
            kept = kept[:max_theorems_per_batch]
        out_path = os.path.join(TS_DIR, eval_name + ".json")
        json.dump({eval_name: kept}, open(out_path, "w"), ensure_ascii=False, indent=2)
        manifest.append({
            "name": eval_name, "path": out_path, "size": len(kept),
            "source_batch": batch_name, "dropped_missing_path": dropped,
            "schema": "{set_name: [{file_path, full_name, namespace}]}",
            "registered": False,  # registered at runtime by sf1_run_eval.py
            "eval_name": eval_name,
        })
    _ensure(os.path.join(OUT_REAL, "x"))
    json.dump({"theorem_sets_dir": TS_DIR, "sets": manifest},
              open(os.path.join(OUT_REAL, "runnable_theorem_sets_manifest.json"), "w"),
              ensure_ascii=False, indent=2)
    return manifest


def build_command(policy, ts_file, run_dir, top_k, max_steps, ns9_cfg):
    """Return (cmd_list_or_None, supported, reason). Uses sf1_run_eval wrapper."""
    base = [sys.executable, RUN_EVAL, "--theorem-set-file", ts_file,
            "--out-dir", run_dir, "--top-k", str(top_k), "--max-steps", str(max_steps)]
    if policy == "rc1":
        return (base + ["--policy-type", "hybrid_evolved",
                        "--route-config", NS24_ROUTER,
                        "--strategy-config", RC1_WRAPPER],
                True, "RC1 production stack via runtime-registered SF1 set")
    if policy == "ns9":
        if ns9_cfg:
            return (base + ["--policy-type", "hybrid_evolved",
                            "--strategy-config", ns9_cfg],
                    True, f"NS9 strategy config: {ns9_cfg}")
        return None, False, "NS9 strategy config not discoverable"
    if policy == "raw":
        return None, False, ("raw baseline needs a confirmed base-policy ckpt + "
                             "no-wrapper mode; not safely discoverable")
    if policy == "experimental":
        return None, False, "no safe SF1 experimental config enabled this stage"
    return None, False, f"unknown policy '{policy}'"


# --- metrics.json schema (live eval_rollout_all writer) ---------------------
# Top-level aggregate (scalars/dicts): total_theorems, available, proved,
#   errored, exhausted, skipped, success_rate, proved_by_origin, ...
# per_theorem: list of dicts, each with the AUTHORITATIVE solved flag `finished`
#   (bool), plus full_name, file_path, available, has_error, num_steps,
#   tactics_used, winning_tactic, winning_tactic_origin, error_message, ...
#
# SCHEMA NOTE / BUGFIX: the authoritative per-theorem solved flag is `finished`.
# Earlier code keyed on `proof_finished`/`solved`/`proved`, none of which exist
# per-theorem — so every theorem read as unsolved (the SF1 "RC1 0/3" undercount).
# We now read `finished` first and only fall back to the legacy keys (recording a
# warning) when `finished` is absent, so we never again silently undercount wins.
def _theorem_solved(t):
    """Return (solved_bool, warning_or_None) using `finished` as authoritative."""
    if "finished" in t:
        return bool(t.get("finished")), None
    for k in ("proof_finished", "solved", "proved"):
        if k in t:
            return bool(t.get(k)), f"used legacy key '{k}' ('finished' absent)"
    return False, "no solved flag found (finished/proof_finished/solved/proved absent)"


def parse_metrics(run_dir):
    hits = sorted(glob.glob(os.path.join(run_dir, "**", "metrics.json"), recursive=True))
    if not hits:
        return None
    try:
        m = json.load(open(hits[0]))
    except Exception:
        return None
    per = m.get("per_theorem") if isinstance(m, dict) else None
    solved = total = None
    per_out = []
    warnings = []
    if isinstance(per, list):
        total = len(per)
        solved = 0
        for t in per:
            ok, warn = _theorem_solved(t)
            if warn:
                warnings.append({"full_name": t.get("full_name") or t.get("theorem"),
                                 "warning": warn})
            solved += 1 if ok else 0
            per_out.append({"full_name": t.get("full_name") or t.get("theorem"),
                            "file_path": t.get("file_path"),
                            "solved": ok,
                            "finished": t.get("finished"),
                            "num_steps": t.get("num_steps"),
                            "winning_tactic": t.get("winning_tactic"),
                            "winning_tactic_origin": t.get("winning_tactic_origin"),
                            "error_message": t.get("error_message")})
    agg = {k: v for k, v in m.items() if not isinstance(v, (list, dict))} \
        if isinstance(m, dict) else {}
    # Aggregate cross-check: the live writer emits `proved` + `total_theorems`.
    agg_solved = agg.get("proved")
    if agg_solved is None:
        agg_solved = agg.get("num_proved") or agg.get("solved")
    agg_total = (agg.get("total_theorems") or agg.get("num_theorems")
                 or agg.get("total") or agg.get("available"))
    if solved is None:
        solved, total = agg_solved, agg_total
    elif agg_solved is not None and agg_solved != solved:
        warnings.append({"warning": f"per_theorem solved={solved} disagrees with "
                                    f"aggregate proved={agg_solved}"})
    return {"metrics_path": hits[0], "solved": solved, "total": total,
            "aggregate_solved": agg_solved, "aggregate_total": agg_total,
            "aggregate": agg, "per_theorem": per_out, "parse_warnings": warnings}


def repair_existing_results(results_path, out_path):
    """Re-derive solved/total/per_theorem for an existing eval_matrix_results.json
    using the corrected `finished`-key parser, WITHOUT re-running any eval.

    Preserves the pre-repair value per result (`pre_repair_solved`) and returns
    (data, changes) where changes lists every result whose solved count moved.
    """
    data = json.load(open(results_path))
    changes = []
    for res in data.get("results", []):
        if res.get("status") != "ran" or not res.get("run_dir"):
            continue
        mt = parse_metrics(res["run_dir"])
        if not mt:
            continue
        before = res.get("solved")
        res["pre_repair_solved"] = before
        res["solved"] = mt["solved"]
        res["total"] = mt["total"]
        res["per_theorem"] = mt["per_theorem"]
        res["aggregate_solved"] = mt.get("aggregate_solved")
        res["parse_warnings"] = mt.get("parse_warnings")
        res["repaired"] = True
        if before != mt["solved"]:
            changes.append({"theorem_set": res.get("theorem_set"),
                            "policy": res.get("policy"),
                            "before_solved": before, "after_solved": mt["solved"],
                            "total": mt["total"]})
    data["repaired_with_finished_key"] = True
    json.dump(data, open(out_path, "w"), ensure_ascii=False, indent=2)
    return data, changes


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SF1 (e): eval matrix + live unblocker.")
    p.add_argument("--batch-manifest",
                   default=os.path.join(OUT_REAL, "batch_manifest.json"))
    p.add_argument("--batches-dir", default="project/evolve/experiments/sf1/batches/real")
    p.add_argument("--frontier-with-paths",
                   default=os.path.join(OUT_REAL, "frontier_with_paths.jsonl"))
    p.add_argument("--out-dir", default=os.path.join(OUT_REAL, "eval"))
    p.add_argument("--policies", default="raw,ns9,rc1,experimental")
    p.add_argument("--prepare-runnable-sets", action="store_true")
    p.add_argument("--run-real", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--max-theorems-per-batch", type=int, default=0)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--timeout", type=int, default=1200)
    # Part-1 truth-repair mode: re-parse metrics with the corrected `finished`
    # key for an existing results file (no eval re-run).
    p.add_argument("--repair-existing-results", action="store_true")
    p.add_argument("--results", default=os.path.join(OUT_REAL, "eval_matrix_results.json"))
    p.add_argument("--out", default=os.path.join(OUT_REAL, "eval_matrix_results.corrected.json"))
    p.add_argument("--in-place", action="store_true",
                   help="With --repair-existing-results: also overwrite --results "
                        "after backing it up to *.prefix_bug.bak.json.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.repair_existing_results:
        if not os.path.isfile(args.results):
            print(f"[sf1:repair] ERROR: results file not found: {args.results}",
                  file=sys.stderr)
            return 2
        data, changes = repair_existing_results(args.results, args.out)
        if args.in_place:
            bak = args.results.replace(".json", ".prefix_bug.bak.json")
            if not os.path.isfile(bak):
                json.dump(json.load(open(args.results)), open(bak, "w"),
                          ensure_ascii=False, indent=2)
            json.dump(data, open(args.results, "w"), ensure_ascii=False, indent=2)
            print(f"[sf1:repair] in-place: backed up original -> {bak}; "
                  f"rewrote {args.results}")
        print(f"[sf1:repair] corrected results -> {args.out}")
        print(f"[sf1:repair] {len(changes)} result(s) changed by the finished-key fix:")
        for c in changes:
            print(f"           {c['theorem_set']}/{c['policy']}: "
                  f"solved {c['before_solved']} -> {c['after_solved']} / {c['total']}")
        return 0
    caps = detect_eval_capabilities()
    ns9_cfg = discover_ns9_config()
    path_map = load_path_map(args.frontier_with_paths)
    os.makedirs(args.out_dir, exist_ok=True)

    # Always (re)prepare runnable sets when we have backfilled paths; harmless
    # and keeps --run-real self-contained.
    runnable = []
    if path_map and (args.prepare_runnable_sets or args.run_real or True):
        runnable = prepare_runnable_sets(path_map, args.batches_dir,
                                         args.max_theorems_per_batch)

    run_real = args.run_real and not args.dry_run
    # selection order: as in RUNNABLE_TARGETS; cap by --max-batches
    sel = [m for m in runnable if m["size"] > 0]
    if args.max_batches:
        sel = sel[: args.max_batches]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    matrix, results, cmd_lines = [], [], ["#!/usr/bin/env bash", "set -uo pipefail", ""]
    for m in (sel if sel else [{"name": None, "path": None, "size": 0,
                                "source_batch": None}]):
        for policy in policies:
            set_name = m["name"]
            ts_file = m["path"]
            run_dir = os.path.join(args.out_dir,
                                   f"rc1_smoke_{set_name}" if policy == "rc1"
                                   else f"{policy}_{set_name}")
            cmd, supported, reason = build_command(
                policy, ts_file, run_dir, args.top_k, args.max_steps, ns9_cfg)
            empty = (m["size"] == 0)
            entry = {"policy": policy, "batch": m["source_batch"],
                     "theorem_set": set_name, "theorem_set_file": ts_file,
                     "size": m["size"], "supported": supported, "reason": reason,
                     "command": " ".join(cmd) if cmd else None}
            matrix.append(entry)
            if cmd:
                cmd_lines += [f"# batch={m['source_batch']} set={set_name} "
                              f"policy={policy} supported={supported} size={m['size']}",
                              (" ".join(cmd)) if supported and not empty
                              else f"# NOT-RUN: {reason if not supported else 'empty set'}",
                              ""]

            res = {"policy": policy, "batch": m["source_batch"], "theorem_set": set_name,
                   "ran": False, "return_code": None, "run_dir": run_dir,
                   "status": None}
            if not supported:
                res["status"] = "unsupported"
            elif empty or not ts_file:
                res["status"] = "blocked_empty_set"
            elif not run_real:
                res["status"] = "dry_run"
            else:
                os.makedirs(run_dir, exist_ok=True)
                run_cmd = ([sys.executable, TIMEOUT_HELPER, str(args.timeout)] + cmd
                           if os.path.isfile(TIMEOUT_HELPER) else cmd)
                so = os.path.join(run_dir, "eval.stdout.log")
                se = os.path.join(run_dir, "eval.stderr.log")
                try:
                    r = subprocess.run(run_cmd, capture_output=True, text=True)
                    open(so, "w").write(r.stdout or "")
                    open(se, "w").write(r.stderr or "")
                    res.update({"ran": True, "return_code": r.returncode,
                                "stdout_log": so, "stderr_log": se})
                    mt = parse_metrics(run_dir)
                    if r.returncode == 0 and mt:
                        res["status"] = "ran"
                        res["metrics_path"] = mt["metrics_path"]
                        res["solved"] = mt["solved"]
                        res["total"] = mt["total"]
                        res["aggregate_solved"] = mt.get("aggregate_solved")
                        res["parse_warnings"] = mt.get("parse_warnings")
                        res["per_theorem"] = mt["per_theorem"]
                    else:
                        res["status"] = "failed"
                        res["error_summary"] = (r.stderr or "")[-1200:] or \
                            f"return_code={r.returncode}, no metrics.json produced"
                except Exception as e:  # pragma: no cover
                    res["status"] = "failed"
                    res["error_summary"] = f"{type(e).__name__}: {e}"
            results.append(res)

    json.dump({"eval_capabilities": caps, "ns9_config": ns9_cfg,
               "path_map_size": len(path_map), "run_real": run_real,
               "policies": policies, "runnable_sets": runnable, "matrix": matrix},
              open(os.path.join(OUT_REAL, "eval_matrix_manifest.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump({"run_real": run_real, "results": results},
              open(os.path.join(OUT_REAL, "eval_matrix_results.json"), "w"),
              ensure_ascii=False, indent=2)
    cmds_out = os.path.join(OUT_REAL, "eval_commands.sh")
    open(cmds_out, "w").write("\n".join(cmd_lines) + "\n")
    os.chmod(cmds_out, 0o755)

    ran = [r for r in results if r["status"] == "ran"]
    failed = [r for r in results if r["status"] == "failed"]
    print(f"[sf1:eval] {'RUN-REAL' if run_real else 'DRY-RUN'} runnable_sets="
          f"{[(m['name'], m['size']) for m in runnable]} path_map={len(path_map)}")
    print(f"[sf1:eval] matrix={len(matrix)} ran={len(ran)} failed={len(failed)} "
          f"ns9_cfg={ns9_cfg}")
    for r in ran:
        print(f"           RAN {r['theorem_set']}/{r['policy']} rc={r['return_code']} "
              f"solved={r.get('solved')}/{r.get('total')}")
    for r in failed:
        print(f"           FAILED {r['theorem_set']}/{r['policy']} rc={r['return_code']} "
              f"-- {str(r.get('error_summary',''))[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
