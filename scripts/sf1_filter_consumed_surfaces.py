#!/usr/bin/env python3
"""SF1 stage (b): consumed-surface exclusion.

Subtract every theorem already consumed by a prior experiment (and, optionally,
every theorem already seen on a prior eval surface) from the raw catalog, so SF1
mines genuinely *open* frontier. This is the bookkeeping that hand-guided
namespace mining never had (it caused the NS21 transfer-ceiling artifact when the
same theorems were re-mined across experiments).

Inputs : catalog.jsonl (stage a) + the batch policy's consumed-surface config.
Outputs: frontier.jsonl (open theorems) + exclusion_ledger.json (what/why removed).

This stage is now wired for real: it builds the consumed-theorem set from actual
repo artifacts via ``sf1_common.load_consumed_decl_names`` — known theorem-set
JSON/JSONL/TXT files, prior eval surfaces, and (opt-in) benchmark/python sources.
``--dry-run`` still uses a deterministic in-memory consumed set so the pipeline
can be exercised without touching repo artifacts.

SAFETY
------
- Read-only with respect to production configs; only writes under the SF1 out dir.
- Missing consumed-set paths are skipped (recorded in the ledger), never raised.
- The RC1 wrapper / NS9 genome / NS24 router / REL1 reports are never written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sf1_common as C  # noqa: E402

SEED = 1729

# Deterministic placeholder consumed set used under --dry-run. In a real run
# these names are harvested from the configured repo artifacts.
_PLACEHOLDER_CONSUMED = {
    "Multiset.map_cons_placeholder",   # consumed by WX3
    "List.append_assoc_placeholder",   # consumed by WX2
    "Option.bind_some_placeholder",    # consumed by WX1
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SF1 (b): exclude consumed surfaces.")
    p.add_argument("--catalog", default="project/evolve/experiments/sf1/out/catalog.jsonl",
                   help="Input catalog JSONL from stage (a).")
    p.add_argument("--policy", default="project/evolve/experiments/sf1/sf1_batch_policy.json",
                   help="Batch policy JSON (provides the consumed-surface config).")
    p.add_argument("--out", default="project/evolve/experiments/sf1/out/frontier.jsonl",
                   help="Output JSONL of open frontier theorems.")
    p.add_argument("--ledger", default="project/evolve/experiments/sf1/out/exclusion_ledger.json",
                   help="Output JSON exclusion ledger.")
    p.add_argument("--consumed-out", default="project/evolve/experiments/sf1/out/consumed_decl_names.txt",
                   help="Optional dump of the resolved consumed decl-name set.")
    p.add_argument("--extra-source", action="append", default=[],
                   help="Additional consumed source (file/dir/glob); repeatable.")
    p.add_argument("--python-scan", action="store_true",
                   help="Also scan configured python_sources for decl-like literals.")
    p.add_argument("--dry-run", action="store_true",
                   help="Use a deterministic placeholder consumed set.")
    p.add_argument("--verbose", action="store_true", help="Log missing sources.")
    return p.parse_args(argv)


def load_policy(path):
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def resolve_consumed(policy, args):
    """Return (consumed_names:set, source_ledger:dict)."""
    if args.dry_run:
        return set(_PLACEHOLDER_CONSUMED), {
            "mode": "dry_run_placeholder",
            "per_path": {}, "missing": [], "scanned_paths": 0,
            "python_scanned": [], "total_names": len(_PLACEHOLDER_CONSUMED),
        }

    cfg = policy.get("exclude_consumed_surfaces", {}) or {}
    sources = list(cfg.get("consumed_sets", []) or [])
    sources += list(cfg.get("eval_surface_sets", []) or [])
    sources += list(args.extra_source or [])
    key_hints = cfg.get("key_hints")
    recursive = bool(cfg.get("recursive", True))

    py_sources = list(cfg.get("python_sources", []) or [])
    do_py = bool(args.python_scan or cfg.get("python_string_scan", False))

    names, ledger = C.load_consumed_decl_names(
        sources, key_hints=key_hints, recursive=recursive,
        python_string_scan=False, verbose=args.verbose)

    if do_py and py_sources:
        py_names, py_ledger = C.load_consumed_decl_names(
            py_sources, key_hints=key_hints, recursive=recursive,
            python_string_scan=True, verbose=args.verbose)
        names |= py_names
        ledger["per_path"].update(py_ledger["per_path"])
        ledger["python_scanned"] += py_ledger["python_scanned"]
        ledger["missing"] += py_ledger["missing"]
        ledger["scanned_paths"] += py_ledger["scanned_paths"]
        ledger["total_names"] = len(names)

    ledger["mode"] = "artifacts"
    return names, ledger


def main(argv=None):
    args = parse_args(argv)

    if not os.path.isfile(args.catalog):
        print(f"[sf1:filter] ERROR: catalog not found: {args.catalog}. "
              f"Run sf1_extract_mathlib_catalog.py first.", file=sys.stderr)
        return 2

    catalog = C.read_json_or_jsonl(args.catalog)
    policy = load_policy(args.policy)
    if not policy and not args.dry_run and args.verbose:
        print(f"[sf1:filter] WARN: policy not found: {args.policy}", file=sys.stderr)

    consumed, source_ledger = resolve_consumed(policy, args)

    frontier, excluded = [], []
    for row in catalog:
        # A catalog row's decl name may live under any of the standard keys.
        cand = C.extract_decl_names_from_record(row)
        name = cand[0] if cand else row.get("name") if isinstance(row, dict) else None
        if name and name in consumed:
            excluded.append(name)
        else:
            frontier.append(row)

    C.write_jsonl(frontier, args.out)

    if args.consumed_out and not args.dry_run:
        C.ensure_parent_dir(args.consumed_out)
        with open(args.consumed_out, "w", encoding="utf-8") as fh:
            for nm in sorted(consumed):
                fh.write(nm + "\n")

    ledger = {
        "seed": SEED,
        "dry_run": bool(args.dry_run),
        "catalog_count": len(catalog),
        "frontier_count": len(frontier),
        "excluded_count": len(excluded),
        "consumed_set_size": len(consumed),
        "excluded_names": sorted(set(excluded)),
        "source_ledger": source_ledger,
    }
    C.ensure_parent_dir(args.ledger)
    with open(args.ledger, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, ensure_ascii=False, indent=2)

    tag = "DRY-RUN " if args.dry_run else ""
    print(f"[sf1:filter] {tag}seed={SEED} consumed={len(consumed)} "
          f"({source_ledger.get('scanned_paths', 0)} paths) | "
          f"{len(catalog)} catalog -> {len(frontier)} frontier "
          f"({len(excluded)} excluded) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
