#!/usr/bin/env python3
"""SF4 Part 2 — collect the literal-RC2 failure pool.

Aggregates RC2 failures from existing artifacts, dedupes by full_name, EXCLUDES
anything literal RC2 solved (incl. SX3 production-subsumed cases), and writes a
failure pool (rows with file_path) + an unresolved file (rows without file_path).

Primary authoritative source: literal RC2 results (rc3_validation) — theorems with
finished==false are *confirmed* RC2 failures (with file_path + a trace to mine
last_goal/last_error). Secondary: the SF1 frontier candidate list (unconfirmed —
tagged for live confirmation in Part 3) and SX3 holdout/cluster results.

No live Lean is run here (pure artifact aggregation).
"""
from __future__ import annotations

import argparse
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LITERAL_RC2 = "project/evolve/experiments/rc3_validation/out/literal_rc2_results.json"
FRONTIER = "project/evolve/experiments/sf1/out/real/frontier_with_paths.jsonl"
SX3_HOLDOUT = "project/evolve/experiments/sx3/out/sx3_fresh_holdout_results.json"
SX3_CLUSTER = "project/evolve/experiments/sx3/out/sx3_set_cluster_results.json"


def _ns_of(full_name):
    return full_name.split(".")[0] if "." in full_name else ""


def _last_trace_info(trace_rows):
    """Return (last_goal, last_error) from a theorem's trace rows."""
    if not trace_rows:
        return None, None
    last_err = None
    last_goal = None
    for r in trace_rows:
        if r.get("state_pp"):
            last_goal = r["state_pp"]
        if r.get("error_message"):
            last_err = r["error_message"]
    return last_goal, last_err


def _load_trace(path):
    rows = {}
    if not path or not os.path.isfile(path):
        return rows
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        rows.setdefault(rec.get("full_name"), []).append(rec)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--literal-rc2", default=LITERAL_RC2)
    p.add_argument("--frontier", default=FRONTIER)
    p.add_argument("--out-pool", required=True)
    p.add_argument("--out-unresolved",
                   default="project/evolve/experiments/sf4/cases/rc2_failure_pool_unresolved.jsonl")
    p.add_argument("--out-summary-json", required=True)
    p.add_argument("--out-summary-md", required=True)
    args = p.parse_args(argv)

    rc2_solved = set()      # exclude these (RC2 already solves / production-subsumed)
    rows_by = {}            # full_name -> row

    def add(full_name, **kw):
        if not full_name:
            return
        row = rows_by.get(full_name)
        if row is None:
            row = {"full_name": full_name, "file_path": None, "namespace": _ns_of(full_name),
                   "source_surface": None, "rc2_finished": None, "last_goal": None,
                   "last_error": None, "trace_path": None, "tags": []}
            rows_by[full_name] = row
        for k, v in kw.items():
            if k == "tags":
                for t in v:
                    if t not in row["tags"]:
                        row["tags"].append(t)
            elif v is not None and (row.get(k) in (None, "") or k == "rc2_finished"):
                row[k] = v

    # ---- primary: literal RC2 results (authoritative) ----
    src_counts = {}
    if os.path.isfile(args.literal_rc2):
        d = json.load(open(args.literal_rc2))
        trace = _load_trace(d.get("trace_path"))
        for r in d.get("per_theorem", []):
            fn = r["full_name"]
            if r.get("finished"):
                rc2_solved.add(fn)
                continue
            lg, le = _last_trace_info(trace.get(fn))
            add(fn, file_path=r.get("file_path"), namespace=_ns_of(fn),
                source_surface="literal_rc2_results", rc2_finished=False,
                last_goal=lg, last_error=le, trace_path=d.get("trace_path"),
                tags=["rc2_confirmed_failure", "role:" + str(r.get("role"))])
            src_counts["literal_rc2"] = src_counts.get("literal_rc2", 0) + 1

    # ---- secondary: SF1 frontier (unconfirmed candidates) ----
    if os.path.isfile(args.frontier):
        for line in open(args.frontier):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            fn = r.get("name") or r.get("full_name")
            if not fn or fn in rc2_solved:
                continue
            add(fn, file_path=r.get("file_path"), namespace=r.get("namespace") or _ns_of(fn),
                source_surface=r.get("source_module") or "sf1_frontier",
                tags=["frontier_unconfirmed"])
            src_counts["sf1_frontier"] = src_counts.get("sf1_frontier", 0) + 1

    # ---- enrichment: SX3 holdout/cluster (rc2_baseline_finished where present) ----
    for path, tag in [(SX3_HOLDOUT, "sx3_holdout"), (SX3_CLUSTER, "sx3_cluster")]:
        if not os.path.isfile(path):
            continue
        d = json.load(open(path))
        for r in d.get("results", []):
            fn = r.get("full_name")
            if not fn:
                continue
            base_fin = r.get("rc2_baseline_finished")
            if base_fin is True:
                rc2_solved.add(fn)
                continue
            if fn in rows_by:
                rows_by[fn]["tags"].append(tag) if tag not in rows_by[fn]["tags"] else None
                if rows_by[fn].get("last_goal") is None and r.get("initial_goal"):
                    rows_by[fn]["last_goal"] = r.get("initial_goal")

    # drop anything that turned out RC2-solved
    for fn in list(rows_by):
        if fn in rc2_solved:
            del rows_by[fn]

    pool, unresolved = [], []
    for row in rows_by.values():
        (pool if row.get("file_path") else unresolved).append(row)
    pool.sort(key=lambda r: (r["namespace"], r["full_name"]))
    unresolved.sort(key=lambda r: r["full_name"])

    os.makedirs(os.path.dirname(args.out_pool), exist_ok=True)
    with open(args.out_pool, "w") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")
    with open(args.out_unresolved, "w") as f:
        for r in unresolved:
            f.write(json.dumps(r) + "\n")

    confirmed = [r for r in pool if "rc2_confirmed_failure" in r["tags"]]
    unconfirmed = [r for r in pool if "rc2_confirmed_failure" not in r["tags"]]
    by_ns = {}
    for r in pool:
        by_ns[r["namespace"]] = by_ns.get(r["namespace"], 0) + 1

    summary = {
        "sources": {"literal_rc2": args.literal_rc2, "frontier": args.frontier,
                    "sx3_holdout": SX3_HOLDOUT, "sx3_cluster": SX3_CLUSTER},
        "source_counts": src_counts,
        "excluded_rc2_solved": sorted(rc2_solved),
        "num_excluded_rc2_solved": len(rc2_solved),
        "pool_size": len(pool),
        "pool_with_file_path": len(pool),
        "unresolved_size": len(unresolved),
        "num_confirmed_rc2_failures": len(confirmed),
        "num_frontier_unconfirmed": len(unconfirmed),
        "by_namespace": by_ns,
        "confirmed_failures": [r["full_name"] for r in confirmed],
    }
    os.makedirs(os.path.dirname(args.out_summary_json), exist_ok=True)
    json.dump(summary, open(args.out_summary_json, "w"), indent=2)

    L = ["# SF4 RC2 failure pool", "",
         f"- pool size (with file_path): **{len(pool)}**",
         f"- confirmed RC2 failures (from literal_rc2): **{len(confirmed)}**",
         f"- frontier unconfirmed candidates: **{len(unconfirmed)}**",
         f"- unresolved (no file_path): {len(unresolved)}",
         f"- excluded (RC2 already solves): {len(rc2_solved)}", "",
         "## By namespace", ""]
    for ns, n in sorted(by_ns.items(), key=lambda kv: -kv[1]):
        L.append(f"- {ns or '(none)'}: {n}")
    L += ["", "## Confirmed RC2 failures (priority pool)", ""]
    for r in confirmed:
        L.append(f"- `{r['full_name']}` ({r['file_path']})"
                 + (f" — err: {str(r['last_error'])[:80]}" if r.get("last_error") else ""))
    L += ["", "## Excluded (RC2 already solves / production-subsumed)", ""]
    for fn in sorted(rc2_solved):
        L.append(f"- `{fn}`")
    open(args.out_summary_md, "w").write("\n".join(L))
    print(f"[sf4-collect] pool={len(pool)} confirmed={len(confirmed)} "
          f"unconfirmed={len(unconfirmed)} unresolved={len(unresolved)} excluded={len(rc2_solved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
