#!/usr/bin/env python3
"""TR3 Part 4 — build/expand the retrieval index.

Reuses the SF5 lemma index (`sf5_lemma_index.jsonl`) verbatim and EXPANDS it by
scanning additional traced-Mathlib source directories relevant to the TR3
multi-namespace expansion (Nat / List / Multiset / Order / Algebra basics) plus the
project declaration catalog. The SF5 source-scan helpers are reused directly so the
record schema (full_name / file_path / namespace / name_tokens / statement_text /
decl_kind / source / features) is identical.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import sf5_build_lemma_index as SF5  # noqa: E402

# broader than SF5's focused set (SF5 = Set/Finset/Order.Monotone)
TR3_SOURCE_DIRS = [
    "Mathlib/Data/Set",
    "Mathlib/Data/Finset",
    "Mathlib/Data/Multiset",
    "Mathlib/Data/Nat",
    "Mathlib/Data/List",
    "Mathlib/Order/Monotone",
    "Mathlib/Order/Basic.lean",
    "Mathlib/Order/SetNotation.lean",
]


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf5-index",
                    default="project/evolve/experiments/sf5/out/sf5_lemma_index.jsonl")
    ap.add_argument("--out-index", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    ap.add_argument("--source-root", default=SF5._TRACED_ROOT)
    args = ap.parse_args()

    index = {}

    # 1) reuse SF5 index verbatim
    n_sf5 = 0
    if os.path.exists(_p(args.sf5_index)):
        for line in open(_p(args.sf5_index)):
            if not line.strip():
                continue
            rec = json.loads(line)
            index[rec["full_name"]] = rec
            n_sf5 += 1

    # 2) catalog (weak features) — fill any gaps
    n_catalog = 0
    cat = _p("project/discovered_theorems.json")
    if os.path.exists(cat):
        for th in json.load(open(cat)).get("theorems", []):
            fn = th["full_name"]
            if fn in index:
                continue
            ns = fn.rsplit(".", 1)[0] if "." in fn else ""
            index[fn] = {
                "full_name": fn, "file_path": th.get("file_path"), "namespace": ns,
                "name_tokens": SF5._tokens(fn), "statement_text": None,
                "decl_kind": "theorem", "source": "discovered_theorems",
                "features": SF5._features(fn),
            }
            n_catalog += 1

    # 3) expand source scan over broader dirs
    n_expand = 0
    root = args.source_root
    if root and os.path.isdir(root):
        for fp, rel in SF5._iter_source_files(root, TR3_SOURCE_DIRS):
            for full, rel_path, stmt, kind in SF5._scan_source_file(fp, rel):
                prev = index.get(full)
                if prev is not None and prev.get("statement_text"):
                    continue
                index[full] = {
                    "full_name": full, "file_path": rel_path,
                    "namespace": full.rsplit(".", 1)[0] if "." in full else "",
                    "name_tokens": SF5._tokens(full), "statement_text": stmt or None,
                    "decl_kind": kind, "source": "mathlib_source",
                    "features": SF5._features((stmt or "") + " " + full),
                }
                if prev is None:
                    n_expand += 1

    records = list(index.values())
    with_stmt = sum(1 for r in records if r.get("statement_text"))
    from collections import Counter
    by_source = Counter(r.get("source") for r in records)
    by_kind = Counter(r.get("decl_kind") for r in records)
    by_ns = Counter((r["namespace"].split(".")[0] if r.get("namespace") else "(root)")
                    for r in records)

    os.makedirs(os.path.dirname(_p(args.out_index)), exist_ok=True)
    with open(_p(args.out_index), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "generated_by": "scripts/tr3_build_retrieval_index.py",
        "reused_sf5_records": n_sf5, "added_catalog": n_catalog,
        "added_expansion_source": n_expand,
        "num_lemmas": len(records), "num_with_statement_text": with_stmt,
        "statement_coverage_pct": round(100.0 * with_stmt / max(1, len(records)), 1),
        "by_source": dict(by_source), "by_decl_kind": dict(by_kind),
        "top_namespaces": dict(by_ns.most_common(20)),
        "source_dirs_scanned": TR3_SOURCE_DIRS,
    }
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)

    md = ["# TR3 retrieval index — summary", "",
          f"- total indexed lemmas: **{len(records)}** (SF5 reuse {n_sf5}, "
          f"+catalog {n_catalog}, +expansion {n_expand})",
          f"- with statement text: **{with_stmt}** ({summary['statement_coverage_pct']}%)",
          f"- by source: {dict(by_source)}",
          f"- by decl kind: {dict(by_kind)}", "", "## Top namespaces", ""]
    for ns, c in by_ns.most_common(20):
        md.append(f"- {ns}: {c}")
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")

    print(f"[tr3-index] {len(records)} lemmas ({with_stmt} w/ stmt, "
          f"{summary['statement_coverage_pct']}%); by_source={dict(by_source)}")


if __name__ == "__main__":
    main()
