#!/usr/bin/env python3
"""SF1 live-eval unblocker — deterministic file_path backfill for frontier rows.

Artifact-mode frontier rows carry name/namespace only (statement=null, file_path
missing). Live eval (LeanDojo) needs a real `file_path` per theorem. This script
recovers `file_path` by exact-name join against existing repo sources, never
guessing when a name maps to multiple distinct files.

Sources (in priority order; first that resolves a name unambiguously wins):
  - project/discovered_theorems.json        (full_name -> file_path[, line])
  - project/evolve/routing/*_theorem_sets.json / *_eval_sets.json
  - tasks.THEOREM_SETS                       (imported; TheoremConfig rows)
  - project/evolve/eval_runs/**/metrics.json (per_theorem full_name -> file_path)
    [scanned only for still-unresolved names; early-exits when all covered]

Outputs:
  frontier_with_paths.jsonl        resolved rows + path fields
  frontier_unresolved_paths.jsonl  rows with no/ambiguous path
  path_backfill_report.json        coverage + examples

Determinism: pure joins; the eval_runs scan is ordered by sorted path so the
first-seen file_path for a name is stable across runs.

SAFETY: read-only against all sources (incl. protected configs); writes only
under the SF1 out dir.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    import sf1_common as C
    _read = C.read_json_or_jsonl
    _write = C.write_jsonl
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
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return rows

    def _ensure(path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        return path

    def _write(rows, path):
        _ensure(path)
        n = 0
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        return n

DISCOVERED = "project/discovered_theorems.json"
ROUTING_GLOBS = ["project/evolve/routing/*_theorem_sets.json",
                 "project/evolve/routing/*_eval_sets.json"]
EVAL_RUNS_GLOB = "project/evolve/eval_runs/**/metrics.json"
RUNS_GLOB = "runs/**/metrics.json"


def _frontier_name(row):
    if isinstance(row, dict):
        for k in ("decl_name", "full_name", "name", "theorem"):
            v = row.get(k)
            if isinstance(v, str) and v:
                return v
    return None


def _module_of(file_path):
    if not file_path:
        return None
    fp = file_path[:-5] if file_path.endswith(".lean") else file_path
    return fp.replace("/", ".")


def _add(mapping, name, fp, line, source, src_of, line_of):
    if not name or not fp:
        return
    mapping[name].add(fp)
    if name not in src_of:
        src_of[name] = source
    if line is not None and name not in line_of:
        line_of[name] = line


def _scan_record(rec, mapping, src_of, line_of, source, needed):
    """Recurse a json record, adding (full_name -> file_path)."""
    stack = [rec]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            fn = node.get("full_name") or node.get("decl_name") or node.get("name") \
                or node.get("theorem")
            fp = node.get("file_path") or node.get("file")
            ln = node.get("line")
            if isinstance(fn, str) and isinstance(fp, str) and (needed is None or fn in needed):
                _add(mapping, fn, fp, ln, source, src_of, line_of)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


def build_base_map(needed, verbose=False):
    mapping = defaultdict(set)
    src_of, line_of = {}, {}
    cov = {}

    before = sum(len(v) for v in mapping.values())
    for rec in _read(DISCOVERED):
        _scan_record(rec, mapping, src_of, line_of, "discovered_theorems", needed)
    cov["discovered_theorems"] = len([n for n in needed if n in mapping])

    for g in ROUTING_GLOBS:
        for path in sorted(glob.glob(g)):
            try:
                _scan_record(json.load(open(path)), mapping, src_of, line_of,
                             "routing:" + os.path.basename(path), needed)
            except Exception:
                continue
    cov["routing"] = len([n for n in needed if n in mapping])

    # tasks.THEOREM_SETS (imported)
    try:
        sys.path.insert(0, _REPO)
        import tasks  # noqa
        for setname, rows in getattr(tasks, "THEOREM_SETS", {}).items():
            for tc in rows:
                fn = getattr(tc, "full_name", None)
                fp = getattr(tc, "file_path", None)
                if isinstance(fn, str) and isinstance(fp, str) and fn in needed:
                    _add(mapping, fn, fp, None, "tasks.THEOREM_SETS", src_of, line_of)
    except Exception as e:  # pragma: no cover
        if verbose:
            print(f"[backfill] tasks import skipped: {e}", file=sys.stderr)
    cov["tasks"] = len([n for n in needed if n in mapping])
    return mapping, src_of, line_of, cov


def scan_eval_runs(needed_unresolved, mapping, src_of, line_of, verbose=False):
    """Scan eval-run metrics for still-unresolved names; early-exit when done."""
    remaining = set(needed_unresolved)
    scanned = 0
    for g in (EVAL_RUNS_GLOB, RUNS_GLOB):
        for path in sorted(glob.glob(g, recursive=True)):
            if not remaining:
                break
            scanned += 1
            try:
                m = json.load(open(path))
            except Exception:
                continue
            per = m.get("per_theorem") if isinstance(m, dict) else None
            if isinstance(per, list):
                for t in per:
                    fn = t.get("full_name") or t.get("theorem") or t.get("name")
                    fp = t.get("file_path") or t.get("file")
                    if isinstance(fn, str) and isinstance(fp, str) and fn in remaining:
                        _add(mapping, fn, fp, t.get("line"),
                             "eval_runs:" + os.path.basename(os.path.dirname(path)),
                             src_of, line_of)
                        if len(mapping.get(fn, ())) >= 1:
                            remaining.discard(fn)
    if verbose:
        print(f"[backfill] eval_runs scanned={scanned} still_unresolved={len(remaining)}")
    return scanned


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SF1 frontier file_path backfill.")
    p.add_argument("--frontier",
                   default="project/evolve/experiments/sf1/out/real/frontier.jsonl")
    p.add_argument("--out",
                   default="project/evolve/experiments/sf1/out/real/frontier_with_paths.jsonl")
    p.add_argument("--unresolved-out",
                   default="project/evolve/experiments/sf1/out/real/frontier_unresolved_paths.jsonl")
    p.add_argument("--report",
                   default="project/evolve/experiments/sf1/out/real/path_backfill_report.json")
    p.add_argument("--no-eval-runs", action="store_true",
                   help="Skip the (slower) eval_runs metrics fallback scan.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.frontier):
        print(f"[backfill] ERROR: frontier not found: {args.frontier}", file=sys.stderr)
        return 2
    rows = _read(args.frontier)
    names = [_frontier_name(r) for r in rows]
    needed = set(n for n in names if n)

    mapping, src_of, line_of, cov = build_base_map(needed, args.verbose)
    unresolved_after_base = [n for n in needed if n not in mapping]
    eval_scanned = 0
    if unresolved_after_base and not args.no_eval_runs:
        eval_scanned = scan_eval_runs(unresolved_after_base, mapping, src_of, line_of,
                                      args.verbose)
    cov["eval_runs_or_final"] = len([n for n in needed if n in mapping])

    resolved, unresolved = [], []
    n_exact = n_ambiguous = n_unresolved = 0
    for row, name in zip(rows, names):
        out = dict(row)
        paths = mapping.get(name, set()) if name else set()
        if len(paths) == 1:
            fp = next(iter(paths))
            out.update({
                "file_path": fp,
                "line": line_of.get(name),
                "source_module": _module_of(fp),
                "path_backfill_source": src_of.get(name),
                "path_backfill_confidence": "exact_name_unique_path",
                "path_backfill_notes": [],
            })
            resolved.append(out)
            n_exact += 1
        elif len(paths) > 1:
            out.update({
                "file_path": None,
                "path_backfill_source": None,
                "path_backfill_confidence": "ambiguous",
                "path_backfill_notes": [f"{len(paths)} distinct file_paths for name; "
                                        f"not guessing: {sorted(paths)[:4]}"],
            })
            unresolved.append(out)
            n_ambiguous += 1
        else:
            out.update({
                "file_path": None,
                "path_backfill_source": None,
                "path_backfill_confidence": "unresolved",
                "path_backfill_notes": ["no source provided a file_path for this name"],
            })
            unresolved.append(out)
            n_unresolved += 1

    _write(resolved, args.out)
    _write(unresolved, args.unresolved_out)

    from collections import Counter
    ns_resolved = Counter((r.get("namespace") or "?") for r in resolved)
    ns_unresolved = Counter((r.get("namespace") or "?") for r in unresolved)
    examples_unresolved = [r.get("name") for r in unresolved
                           if (r.get("namespace") or "").startswith("Set")][:10] \
        or [r.get("name") for r in unresolved][:10]

    resolved_names = {_frontier_name(r) for r in resolved}

    def _batch_runnable(batch_path):
        if not os.path.isfile(batch_path):
            return None
        b = _read(batch_path)
        bn = [_frontier_name(x) for x in b]
        have = sum(1 for x in bn if x in resolved_names)
        return {"size": len(bn), "with_path": have, "runnable": have > 0}

    bdir = "project/evolve/experiments/sf1/batches/real"
    report = {
        "total_frontier_rows": len(rows),
        "exact_matches": n_exact,
        "normalized_unambiguous_matches": 0,
        "ambiguous_matches": n_ambiguous,
        "unresolved": n_unresolved,
        "source_coverage_cumulative": cov,
        "eval_runs_files_scanned": eval_scanned,
        "resolved_by_namespace": dict(ns_resolved.most_common()),
        "unresolved_by_namespace": dict(ns_unresolved.most_common()),
        "examples_unresolved": examples_unresolved,
        "batch_runnability": {
            "sf1_balanced_mini": _batch_runnable(os.path.join(bdir, "sf1_balanced_mini.jsonl")),
            "sf1_multiset_holdout": _batch_runnable(os.path.join(bdir, "sf1_multiset_holdout.jsonl")),
            "sf1_frontier_all": _batch_runnable(os.path.join(bdir, "sf1_frontier_all.jsonl")),
        },
    }
    _ensure(args.report)
    json.dump(report, open(args.report, "w"), ensure_ascii=False, indent=2)

    print(f"[backfill] frontier={len(rows)} exact={n_exact} ambiguous={n_ambiguous} "
          f"unresolved={n_unresolved} (eval_runs_scanned={eval_scanned})")
    print(f"[backfill] resolved_by_ns={dict(ns_resolved.most_common())}")
    print(f"[backfill] batch_runnability={report['batch_runnability']}")
    print(f"[backfill] -> {args.out}\n           -> {args.unresolved_out}\n           -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
