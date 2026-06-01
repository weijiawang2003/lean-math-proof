#!/usr/bin/env python3
"""TR6 Part 3 — build the fresh multi-namespace frontier.

Source-scans a curated multi-namespace file list from the traced Mathlib cache (cx1-style
regex extractor with proper namespace tracking; Dojo-resolvability verified by smoke test)
and merges project/discovered_theorems.json. Excludes the TR6 exclusion registry, keeps
only theorem/lemma decls with a real file_path whose statement looks proof-search-relevant,
computes feature flags, and dedups by full_name. The live RC2 step is the final
availability filter.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMIT = "29dcec074de168ac2bf835a77ef68bbe069194c5"
MATHLIB = Path.home() / ".cache" / "lean_dojo" / f"leanprover-community-mathlib4-{COMMIT}" / "mathlib4"

# curated multi-namespace file list (all confirmed present in the traced tree)
FILES = [
    # Finset extensions
    "Mathlib/Data/Finset/Card.lean", "Mathlib/Data/Finset/Image.lean",
    "Mathlib/Data/Finset/Lattice.lean", "Mathlib/Data/Finset/Powerset.lean",
    "Mathlib/Data/Finset/Union.lean", "Mathlib/Data/Finset/Preimage.lean",
    # List
    "Mathlib/Data/List/Basic.lean", "Mathlib/Data/List/Count.lean",
    "Mathlib/Data/List/Dedup.lean", "Mathlib/Data/List/Pairwise.lean",
    "Mathlib/Data/List/Range.lean",
    # Multiset
    "Mathlib/Data/Multiset/Basic.lean", "Mathlib/Data/Multiset/Dedup.lean",
    "Mathlib/Data/Multiset/Bind.lean",
    # Nat extensions
    "Mathlib/Data/Nat/Defs.lean", "Mathlib/Data/Nat/GCD/Basic.lean",
    "Mathlib/Data/Nat/ModEq.lean", "Mathlib/Data/Nat/Log.lean",
    # Set extensions
    "Mathlib/Data/Set/Function.lean", "Mathlib/Data/Set/Lattice.lean",
    # Order
    "Mathlib/Order/Basic.lean", "Mathlib/Order/Bounds/Basic.lean",
    # Other (Bool/Option/Int/Logic)
    "Mathlib/Data/Bool/Basic.lean", "Mathlib/Data/Option/Basic.lean",
    "Mathlib/Data/Int/Defs.lean", "Mathlib/Logic/Basic.lean",
]

DECL_RE = re.compile(
    r"^(?:@\[[^\]]+\]\s*)?(?:protected\s+|private\s+|noncomputable\s+)?"
    r"(theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_'\.]*)", re.MULTILINE)
_NS = re.compile(r"^namespace\s+([A-Za-z_][A-Za-z0-9_'\.]*)", re.MULTILINE)
_SEC = re.compile(r"^section(?:\s+([A-Za-z_][A-Za-z0-9_'\.]*))?", re.MULTILINE)
_END = re.compile(r"^end(?:\s+([A-Za-z_][A-Za-z0-9_'\.]*))?\s*$", re.MULTILINE)


def _strip(src):
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _ns_at(src, pos):
    """Namespace prefix at offset pos. Tracks namespace AND section stacks; only
    namespaces contribute to the prefix. Named `end X` pops whichever stack has X on
    top; unnamed `end` pops the innermost (section preferred). (cx1 semantics.)"""
    ev = []
    for m in _NS.finditer(src):
        if m.start() >= pos:
            break
        ev.append((m.start(), "ns_open", m.group(1)))
    for m in _SEC.finditer(src):
        if m.start() >= pos:
            break
        ev.append((m.start(), "sec_open", m.group(1)))
    for m in _END.finditer(src):
        if m.start() >= pos:
            break
        ev.append((m.start(), "end", m.group(1)))
    ev.sort(key=lambda e: e[0])
    ns_stack, sec_stack = [], []
    for _, kind, name in ev:
        if kind == "ns_open":
            ns_stack.append(name)
        elif kind == "sec_open":
            sec_stack.append(name)
        else:  # end
            if name is None:
                if sec_stack:
                    sec_stack.pop()
                elif ns_stack:
                    ns_stack.pop()
            else:
                if ns_stack and ns_stack[-1] == name:
                    ns_stack.pop()
                elif sec_stack and sec_stack[-1] == name:
                    sec_stack.pop()
                elif name in ns_stack:
                    # pop down to the matching namespace frame
                    while ns_stack and ns_stack[-1] != name:
                        ns_stack.pop()
                    if ns_stack:
                        ns_stack.pop()
    return ".".join(ns_stack)


def _statement(src, start):
    """Capture the signature text from the decl start up to ':=' / 'by' / blank line."""
    chunk = src[start:start + 600]
    # cut at the proof start
    for marker in (" :=", ":=\n", "\n\n"):
        idx = chunk.find(marker)
        if idx != -1:
            chunk = chunk[:idx]
            break
    return re.sub(r"\s+", " ", chunk).strip()[:400]


def _flags(text):
    t = text or ""
    low = t.lower()
    return {
        "has_iff": ("↔" in t) or (" iff" in low),
        "has_eq": ("=" in t and "↔" not in t.split("=")[0][-3:]) or (" eq" in low),
        "has_subset": ("⊆" in t) or ("⊂" in t) or ("subset" in low),
        "has_mem": ("∈" in t) or ("∉" in t) or ("mem" in low),
        "has_disjoint": "disjoint" in low,
        "has_card": "card" in low or ("#" in t),
        "has_map_filter": any(w in low for w in ("map", "filter", "fold", "image", "bind")),
        "has_tofinset": "tofinset" in low,
        "has_nat_arith": any(w in low for w in ("nat", "mod", "gcd", "dvd", "div"))
                         or ("ℕ" in t) or ("≤" in t) or ("<" in t),
        "has_order": any(w in low for w in ("monotone", "le_", "lt_", "order", "bound",
                                            "sup", "inf", "max", "min")) or ("≤" in t),
        "has_singleton": ("singleton" in low) or ("{" in t),
        "has_union_inter": ("∪" in t) or ("∩" in t) or ("union" in low) or ("inter" in low),
    }


def _interesting(fl):
    # proof-search-likely: needs at least one structural feature
    return any(fl[k] for k in ("has_iff", "has_eq", "has_subset", "has_mem", "has_disjoint",
                               "has_card", "has_map_filter", "has_tofinset", "has_order",
                               "has_union_inter"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclusion", required=True)
    ap.add_argument("--out-pool", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    excl = set(json.load(open(_p(args.exclusion)))["excluded_full_names"])

    rows = {}
    scanned_files = []
    # --- source scan ---
    for rel in FILES:
        fp = MATHLIB / rel
        if not fp.exists():
            continue
        scanned_files.append(rel)
        src = _strip(fp.read_text(encoding="utf-8", errors="ignore"))
        for m in DECL_RE.finditer(src):
            if "private" in src[m.start():m.start() + 30]:
                continue  # private decls are not addressable in LeanDojo
            nm = m.group(2)
            ns = _ns_at(src, m.start())
            full = f"{ns}.{nm}" if ns else nm
            if full in excl or full in rows:
                continue
            stmt = _statement(src, m.start())
            fl = _flags(stmt)
            if not _interesting(fl):
                continue
            rows[full] = {
                "full_name": full, "file_path": rel,
                "namespace": ns.split(".")[0] if ns else (full.split(".")[0] if "." in full else ""),
                "statement_text": stmt, "source": "source_scan",
                "features": fl, "freshness_status": "fresh",
            }
    # --- merge discovered_theorems (clean paths, may add a few non-excluded) ---
    dt = _p("project/discovered_theorems.json")
    if os.path.exists(dt):
        d = json.load(open(dt))
        for t in d.get("theorems", []):
            full = t.get("full_name")
            if not full or full in excl or full in rows:
                continue
            ns = full.split(".")[0] if "." in full else ""
            fl = _flags(full)  # no statement; use name
            rows[full] = {
                "full_name": full, "file_path": t.get("file_path"),
                "namespace": ns, "statement_text": None, "source": "discovered_theorems",
                "features": fl, "freshness_status": "fresh",
            }

    pool = [r for r in rows.values() if r.get("file_path")]
    pool.sort(key=lambda r: (r["namespace"], r["full_name"]))

    os.makedirs(os.path.dirname(_p(args.out_pool)), exist_ok=True)
    with open(_p(args.out_pool), "w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_ns = Counter(r["namespace"] for r in pool)
    by_src = Counter(r["source"] for r in pool)
    feat_dist = {k: sum(1 for r in pool if r["features"].get(k)) for k in
                 ("has_iff", "has_eq", "has_subset", "has_mem", "has_disjoint", "has_card",
                  "has_map_filter", "has_tofinset", "has_order", "has_union_inter")}
    summary = {
        "generated_by": "scripts/tr6_build_fresh_frontier.py",
        "num_fresh": len(pool), "num_excluded_applied": len(excl),
        "scanned_files": scanned_files,
        "by_namespace": dict(by_ns), "by_source": dict(by_src),
        "feature_distribution": feat_dist,
        "target_300_met": len(pool) >= 300,
    }
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR6 fresh frontier", "",
          f"- **{len(pool)} fresh candidates** (≥300 target: {summary['target_300_met']})",
          f"- by namespace: {dict(by_ns)}",
          f"- by source: {dict(by_src)}",
          f"- scanned {len(scanned_files)} traced files",
          f"- feature distribution: {feat_dist}"]
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")
    print(f"[tr6-frontier] {len(pool)} fresh candidates; by_ns={dict(by_ns)}")


def _p(*a):
    return os.path.join(_REPO, *a)


if __name__ == "__main__":
    main()
