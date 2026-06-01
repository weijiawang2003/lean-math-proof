#!/usr/bin/env python3
"""SF1 stage (d): real deterministic batch generation over the open frontier.

Reads classified_frontier.jsonl (Stage C) and emits deterministic JSONL theorem
batches plus a manifest. Each batch row preserves the full classified record so
Stage E / F have everything they need. Seed comes from sf1_batch_policy.json
(default 1729); selection is by stable hash, so runs are reproducible.

Outputs:
  <out-dir>/sf1_frontier_all.jsonl
  <out-dir>/sf1_set_frontier.jsonl
  <out-dir>/sf1_multiset_holdout.jsonl
  <out-dir>/sf1_mx2_candidate.jsonl
  <out-dir>/sf1_wx3_candidate.jsonl
  <out-dir>/sf1_balanced_mini.jsonl
  <out-dir>/sf1_failure_driven_seed.jsonl
  batch_manifest.json

SAFETY: writes only under the SF1 batches/out dirs; never touches production cfg.
Existing files in <out-dir> are only overwritten when their name matches an SF1
generated batch name (the sf1_*.jsonl set above).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import sf1_common as C
    _read = C.read_json_or_jsonl
    _write = C.write_jsonl
    _ensure = C.ensure_parent_dir
    _stable = C.stable_hash
except Exception:  # pragma: no cover
    def _read(path):
        rows = []
        if not os.path.isfile(path):
            return rows
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
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

    def _stable(value):
        blob = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) \
            if isinstance(value, (dict, list, tuple)) else str(value)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

SEED = 1729
SF1_BATCH_NAMES = {
    "sf1_frontier_all", "sf1_set_frontier", "sf1_multiset_holdout",
    "sf1_mx2_candidate", "sf1_wx3_candidate", "sf1_balanced_mini",
    "sf1_failure_driven_seed",
}


def _det_sort(records, seed):
    """Deterministic order independent of input order."""
    return sorted(records, key=lambda r: _stable([seed, r.get("decl_name", "")]))


def _dedup(records):
    seen, out = set(), []
    for r in records:
        k = r.get("decl_name")
        if k and k not in seen:
            seen.add(k)
            out.append(r)
    return out


def _score(r, fam):
    return (r.get("candidate_family_scores") or {}).get(fam, 0.0)


def _dominant(records, key_fn, top=5):
    c = Counter()
    for r in records:
        c.update(key_fn(r))
    return dict(c.most_common(top))


def build_batches(records, seed):
    records = _dedup(records)
    ordered = _det_sort(records, seed)

    batches = {}
    batches["sf1_frontier_all"] = list(ordered)
    batches["sf1_set_frontier"] = [r for r in ordered if "has_set" in r.get("tags", [])]
    batches["sf1_multiset_holdout"] = [r for r in ordered if "has_multiset" in r.get("tags", [])]
    batches["sf1_mx2_candidate"] = [
        r for r in ordered
        if r.get("top_candidate_family") == "mx2_set_finite_tofinset_aesop"
        or _score(r, "mx2_set_finite_tofinset_aesop") >= 0.5]
    batches["sf1_wx3_candidate"] = [
        r for r in ordered
        if r.get("top_candidate_family") == "wx3_multiset_induction"
        or _score(r, "wx3_multiset_induction") >= 0.5]
    batches["sf1_failure_driven_seed"] = [
        r for r in ordered
        if _score(r, "future_failure_driven_lemma_candidate") >= 0.5
        or (r.get("classification_confidence", 1.0) <= 0.5
            and _score(r, "rc1_production_stack") >= 0.4)]

    # balanced mini: one representative per (namespace) round-robin, capped.
    by_ns = {}
    for r in ordered:
        by_ns.setdefault(r.get("namespace", "?"), []).append(r)
    mini, cap = [], 8
    # deterministic namespace order
    for ns in sorted(by_ns, key=lambda n: _stable([seed, "ns", n])):
        if by_ns[ns]:
            mini.append(by_ns[ns][0])
    mini = mini[:cap]
    batches["sf1_balanced_mini"] = mini

    return batches


INTENDED_USE = {
    "sf1_frontier_all": "full open frontier; baseline raw/ns9/rc1 sweep",
    "sf1_set_frontier": "Set-heavy surface; probe mx2 narrow Set.Finite aesop relevance",
    "sf1_multiset_holdout": "Multiset holdout; WX3 induction regression/extension guard",
    "sf1_mx2_candidate": "high mx2 score; targeted Set.Finite/toFinset aesop eval",
    "sf1_wx3_candidate": "high wx3 score; targeted Multiset induction eval",
    "sf1_balanced_mini": "small deterministic cross-namespace smoke batch (cheap eval)",
    "sf1_failure_driven_seed": "weak productive-family match / low confidence; SF2 lemma-discovery seed",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SF1 (d): real batch generation.")
    p.add_argument("--classified-frontier", "--classified",
                   default="project/evolve/experiments/sf1/out/real/classified_frontier.jsonl")
    p.add_argument("--policy",
                   default="project/evolve/experiments/sf1/sf1_batch_policy.json")
    p.add_argument("--out-dir",
                   default="project/evolve/experiments/sf1/batches/real")
    p.add_argument("--manifest-out",
                   default="project/evolve/experiments/sf1/out/real/batch_manifest.json")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.classified_frontier):
        print(f"[sf1:batch] ERROR: classified frontier not found: "
              f"{args.classified_frontier}", file=sys.stderr)
        return 2
    seed = SEED
    if os.path.isfile(args.policy):
        try:
            seed = int(json.load(open(args.policy)).get("deterministic_seed", SEED))
        except Exception:
            seed = SEED

    records = _read(args.classified_frontier)
    batches = build_batches(records, seed)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    for name in ["sf1_frontier_all", "sf1_set_frontier", "sf1_multiset_holdout",
                 "sf1_mx2_candidate", "sf1_wx3_candidate", "sf1_balanced_mini",
                 "sf1_failure_driven_seed"]:
        rows = batches.get(name, [])
        path = os.path.join(args.out_dir, name + ".jsonl")
        if os.path.exists(path) and name not in SF1_BATCH_NAMES:
            print(f"[sf1:batch] SKIP non-SF1 existing file: {path}", file=sys.stderr)
            continue
        _write(rows, path)
        manifest.append({
            "batch_name": name,
            "path": path,
            "size": len(rows),
            "sampling_policy": ("balanced_round_robin_by_namespace"
                                if name == "sf1_balanced_mini" else "deterministic_filter"),
            "dominant_namespaces": _dominant(rows, lambda r: [r.get("namespace", "?")]),
            "dominant_candidate_families": _dominant(rows, lambda r: [r.get("top_candidate_family", "?")]),
            "intended_use": INTENDED_USE.get(name, ""),
            "seed": seed,
        })

    _ensure(args.manifest_out)
    with open(args.manifest_out, "w", encoding="utf-8") as fh:
        json.dump({"seed": seed, "out_dir": args.out_dir, "batches": manifest},
                  fh, ensure_ascii=False, indent=2)

    print(f"[sf1:batch] seed={seed} wrote {len(manifest)} batches -> {args.out_dir}")
    for m in manifest:
        print(f"            {m['batch_name']}: {m['size']}  ns={m['dominant_namespaces']}")
    print(f"[sf1:batch] manifest -> {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
