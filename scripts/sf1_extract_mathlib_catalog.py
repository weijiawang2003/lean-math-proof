#!/usr/bin/env python3
"""SF1 stage (a): theorem catalog extraction.

Emit one JSONL record per theorem:

    {"name", "namespace", "file", "line", "statement", "decl_kind", "source"}

This is the raw frontier the rest of the SF1 pipeline filters, classifies, and
batches. There are three modes:

- ``--dry-run``        : deterministic placeholder catalog (no I/O against repo
                         artifacts or the traced cache).
- ``--from-artifacts`` : REAL "mine from existing repo artifacts" mode. Harvest
                         theorem/decl names from the catalog sources configured
                         in the batch policy (eval surfaces, benchmark/python
                         sources, prior theorem-set files) and emit a catalog of
                         the theorems this project has actually touched. This is
                         the universe Stage (b) then trims down to open frontier.
- (default, live)      : drive the 18GB traced LeanDojo cache — NOT yet wired.

SAFETY
------
- Read-only with respect to all production configs (NS9 genome, NS24 router,
  RC1 production wrapper, REL1 reports). This script never writes there.

TODO (LeanDojo / Mathlib integration)
-------------------------------------
- Point ``--mathlib-cache`` at the 18GB traced cache and enumerate declarations
  via LeanDojo's traced-repo API; recover real ``file``/``line``/``statement``.
- macOS has no ``timeout``; wrap any subprocess in ``scripts/run_with_timeout.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sf1_common as C  # noqa: E402

SEED = 1729

# Deterministic placeholder frontier used under --dry-run.
_PLACEHOLDER_CATALOG = [
    {"name": "Multiset.map_cons_placeholder", "namespace": "Multiset", "file": "Mathlib/Data/Multiset/Basic.lean", "line": 101, "statement": "(a ::ₘ s).map f = f a ::ₘ s.map f", "decl_kind": "theorem"},
    {"name": "Multiset.sum_induction_placeholder", "namespace": "Multiset", "file": "Mathlib/Data/Multiset/Basic.lean", "line": 222, "statement": "∀ s : Multiset α, P s", "decl_kind": "theorem"},
    {"name": "Set.Finite.toFinset_union_placeholder", "namespace": "Set", "file": "Mathlib/Data/Set/Finite.lean", "line": 333, "statement": "(hs.union ht).toFinset = hs.toFinset ∪ ht.toFinset", "decl_kind": "theorem"},
    {"name": "Set.mem_inter_placeholder", "namespace": "Set", "file": "Mathlib/Data/Set/Basic.lean", "line": 404, "statement": "x ∈ s ∩ t ↔ x ∈ s ∧ x ∈ t", "decl_kind": "theorem"},
    {"name": "Finset.ext_placeholder", "namespace": "Finset", "file": "Mathlib/Data/Finset/Basic.lean", "line": 515, "statement": "s = t ↔ ∀ a, a ∈ s ↔ a ∈ t", "decl_kind": "theorem"},
    {"name": "List.append_assoc_placeholder", "namespace": "List", "file": "Mathlib/Data/List/Basic.lean", "line": 66, "statement": "(l₁ ++ l₂) ++ l₃ = l₁ ++ (l₂ ++ l₃)", "decl_kind": "theorem"},
    {"name": "Option.bind_some_placeholder", "namespace": "Option", "file": "Mathlib/Data/Option/Basic.lean", "line": 77, "statement": "o.bind some = o", "decl_kind": "theorem"},
    {"name": "Nat.add_comm_placeholder", "namespace": "Nat", "file": "Mathlib/Data/Nat/Basic.lean", "line": 12, "statement": "n + m = m + n", "decl_kind": "theorem"},
    {"name": "Int.emod_emod_placeholder", "namespace": "Int", "file": "Mathlib/Data/Int/Basic.lean", "line": 188, "statement": "a % b % b = a % b", "decl_kind": "theorem"},
    {"name": "Finsupp.support_zero_placeholder", "namespace": "Finsupp", "file": "Mathlib/Data/Finsupp/Basic.lean", "line": 240, "statement": "(0 : α →₀ M).support = ∅", "decl_kind": "theorem"},
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SF1 (a): extract theorem catalog.")
    p.add_argument("--mathlib-cache", default=None,
                   help="Path to the traced Mathlib/LeanDojo cache (live mode).")
    p.add_argument("--policy", default="project/evolve/experiments/sf1/sf1_batch_policy.json",
                   help="Batch policy JSON (provides catalog_sources for --from-artifacts).")
    p.add_argument("--from-artifacts", action="store_true",
                   help="Harvest the catalog from existing repo artifacts (real).")
    p.add_argument("--extra-source", action="append", default=[],
                   help="Additional catalog source (file/dir/glob); repeatable.")
    p.add_argument("--python-scan", action="store_true",
                   help="Also scan configured python catalog sources.")
    p.add_argument("--out", default="project/evolve/experiments/sf1/out/catalog.jsonl",
                   help="Output JSONL path for the theorem catalog.")
    p.add_argument("--limit", type=int, default=0,
                   help="Optional cap on number of theorems emitted (0 = no cap).")
    p.add_argument("--dry-run", action="store_true",
                   help="Emit a deterministic placeholder catalog; touch no cache.")
    p.add_argument("--verbose", action="store_true", help="Log missing sources.")
    return p.parse_args(argv)


def _namespace_of(name):
    return name.split(".")[0] if "." in name else "GENERAL_FRONTIER"


def harvest_from_artifacts(policy, args):
    """Build a catalog by harvesting decl names from configured artifacts."""
    cfg = policy.get("catalog_sources", {}) or {}
    sources = list(cfg.get("sets", []) or []) + list(args.extra_source or [])
    py_sources = list(cfg.get("python_sources", []) or [])
    key_hints = cfg.get("key_hints")
    recursive = bool(cfg.get("recursive", True))
    do_py = bool(args.python_scan or cfg.get("python_string_scan", False))

    names, ledger = C.load_consumed_decl_names(
        sources, key_hints=key_hints, recursive=recursive, verbose=args.verbose)
    if do_py and py_sources:
        py_names, py_ledger = C.load_consumed_decl_names(
            py_sources, key_hints=key_hints, recursive=recursive,
            python_string_scan=True, verbose=args.verbose)
        names |= py_names
        for k in ("per_path", "missing", "python_scanned"):
            ledger[k] = (ledger.get(k) or []) + py_ledger[k] if isinstance(
                ledger.get(k), list) else {**ledger.get(k, {}), **py_ledger[k]}
        ledger["scanned_paths"] = ledger.get("scanned_paths", 0) + py_ledger["scanned_paths"]

    rows = []
    for name in sorted(names):
        rows.append({
            "name": name,
            "namespace": _namespace_of(name),
            "file": None,
            "line": None,
            "statement": None,          # TODO: backfill from traced cache.
            "decl_kind": "theorem",
            "source": "artifacts",
        })
    return rows, ledger


def extract_catalog_live(mathlib_cache, limit):
    """Live traced-cache extraction (not yet implemented)."""
    raise NotImplementedError(
        "Live Mathlib catalog extraction is not wired yet. Use --dry-run for "
        "the scaffold, --from-artifacts to mine from repo artifacts, or "
        "implement traced-cache enumeration over --mathlib-cache.")


def main(argv=None):
    args = parse_args(argv)

    if args.dry_run:
        rows = list(_PLACEHOLDER_CATALOG)
        if args.limit:
            rows = rows[: args.limit]
        C.write_jsonl(rows, args.out)
        print(f"[sf1:extract] DRY-RUN seed={SEED} wrote {len(rows)} placeholder "
              f"theorems -> {args.out}")
        return 0

    if args.from_artifacts:
        policy = {}
        if os.path.isfile(args.policy):
            with open(args.policy, encoding="utf-8") as fh:
                policy = json.load(fh)
        rows, ledger = harvest_from_artifacts(policy, args)
        if args.limit:
            rows = rows[: args.limit]
        C.write_jsonl(rows, args.out)
        print(f"[sf1:extract] ARTIFACTS seed={SEED} harvested {len(rows)} "
              f"theorems from {ledger.get('scanned_paths', 0)} paths "
              f"({len(ledger.get('missing', []))} sources missing) -> {args.out}")
        return 0

    if not args.mathlib_cache or not os.path.exists(args.mathlib_cache):
        print(f"[sf1:extract] ERROR: --mathlib-cache path missing or not found: "
              f"{args.mathlib_cache!r}. Use --dry-run or --from-artifacts.",
              file=sys.stderr)
        return 2

    rows = extract_catalog_live(args.mathlib_cache, args.limit)
    C.write_jsonl(rows, args.out)
    print(f"[sf1:extract] wrote {len(rows)} theorems -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
