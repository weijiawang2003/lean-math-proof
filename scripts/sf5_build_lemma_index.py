#!/usr/bin/env python3
"""SF5 Part 3 — build a Mathlib lemma candidate index.

Sources (best-effort, coverage reported, not required to be perfect):
  1. project/discovered_theorems.json          — declaration catalog (names/paths,
                                                  NO statement text -> weak features)
  2. local LeanDojo-traced Mathlib source       — real statement text for a focused set
                                                  of directories relevant to the targets
                                                  (Set / Finset / Order.Monotone / ...)

Each indexed lemma: full_name, file_path, namespace, name_tokens, statement_text
(optional), source, features. When statement text is unavailable we fall back to
name/token/path features only.
"""
from __future__ import annotations

import argparse
import json
import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRACED_ROOT = os.path.expanduser(
    "~/.cache/lean_dojo/leanprover-community-mathlib4-"
    "29dcec074de168ac2bf835a77ef68bbe069194c5/mathlib4")

# focused source directories (relative to traced root / Mathlib) relevant to the
# Set iff-equivalence + Set.ite-subset + Multiset.toFinset target clusters.
SOURCE_DIRS = [
    "Mathlib/Data/Set",
    "Mathlib/Data/Finset",
    "Mathlib/Order/Monotone",
    "Mathlib/Order/SetNotation.lean",
    "Mathlib/Data/Set/Basic.lean",
]

_DECL_RE = re.compile(r"^(\s*)(?:protected\s+|private\s+|noncomputable\s+|@\[[^\]]*\]\s*)*"
                      r"(theorem|lemma|def|abbrev)\s+([A-Za-z_][A-Za-z0-9_'!?.]*)")
_NS_RE = re.compile(r"^\s*namespace\s+([A-Za-z_][\w.]*)")
_END_RE = re.compile(r"^\s*end\s+([A-Za-z_][\w.]*)\s*$")


def _p(*a):
    return os.path.join(_REPO, *a)


def _tokens(name):
    # split on dots, camelCase, underscores
    last = name.split(".")[-1]
    parts = re.split(r"[._]", name)
    cam = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", last)
    toks = set()
    for s in parts + cam + [last]:
        s = s.strip().lower()
        if s:
            toks.add(s)
    return sorted(toks)


def _features(text):
    t = text or ""
    low = t.lower()
    return {
        "has_iff": ("↔" in t) or ("iff" in low),
        "has_subset": ("⊆" in t) or ("subset" in low),
        "has_ssubset": ("⊂" in t) or ("ssubset" in low),
        "has_monotone": ("monotone" in low),
        "has_strictmono": ("strictmono" in low) or ("strict_mono" in low),
        "has_set": ("set" in low) or ("∈" in t) or ("∪" in t) or ("∩" in t)
                   or ("⊆" in t) or ("{" in t),
        "has_singleton": ("singleton" in low),
        "has_insert": ("insert" in low),
        "has_compl": ("compl" in low) or ("ᶜ" in t),
        "has_pair": ("pair" in low),
        "has_pairwisedisjoint": ("pairwisedisjoint" in low) or ("pairwise_disjoint" in low),
        "has_ite": ("ite" in low),
        "has_union": ("union" in low) or ("∪" in t),
        "has_empty": ("empty" in low) or ("∅" in t),
    }


def _scan_source_file(fp, rel_path):
    """Yield (full_name, statement_text) for theorem/lemma decls in a .lean file."""
    try:
        lines = open(fp, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return
    ns_stack = []
    for i, ln in enumerate(lines):
        m_ns = _NS_RE.match(ln)
        if m_ns:
            ns_stack.append(m_ns.group(1))
            continue
        m_end = _END_RE.match(ln)
        if m_end:
            if ns_stack and (ns_stack[-1] == m_end.group(1)
                             or ns_stack[-1].endswith("." + m_end.group(1))):
                ns_stack.pop()
            continue
        m = _DECL_RE.match(ln)
        if not m:
            continue
        kind = m.group(2)
        short = m.group(3)
        ns = ".".join(ns_stack)
        full = f"{ns}.{short}" if ns else short
        # assemble signature up to `:=` (bounded lookahead)
        buf = []
        for j in range(i, min(i + 14, len(lines))):
            seg = lines[j]
            buf.append(seg)
            if ":=" in seg:
                break
        text = " ".join(s.strip() for s in buf)
        idx = text.find(":=")
        if idx != -1:
            text = text[:idx]
        text = re.sub(r"\s+", " ", text).strip()
        yield full, rel_path, text, kind


def _iter_source_files(root, dirs):
    seen = set()
    for d in dirs:
        full = os.path.join(root, d)
        if os.path.isfile(full) and full.endswith(".lean"):
            if full not in seen:
                seen.add(full)
                yield full, os.path.relpath(full, root)
        elif os.path.isdir(full):
            for dp, _dn, fns in os.walk(full):
                for fn in fns:
                    if fn.endswith(".lean"):
                        f = os.path.join(dp, fn)
                        if f not in seen:
                            seen.add(f)
                            yield f, os.path.relpath(f, root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-index", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    ap.add_argument("--source-root", default=_TRACED_ROOT)
    args = ap.parse_args()

    index = {}  # full_name -> record (source-text record wins over catalog-only)

    # --- catalog source (weak) ---
    cat_path = _p("project/discovered_theorems.json")
    n_catalog = 0
    if os.path.exists(cat_path):
        cat = json.load(open(cat_path))
        for th in cat.get("theorems", []):
            fn = th["full_name"]
            ns = fn.rsplit(".", 1)[0] if "." in fn else ""
            index[fn] = {
                "full_name": fn,
                "file_path": th.get("file_path"),
                "namespace": ns,
                "name_tokens": _tokens(fn),
                "statement_text": None,
                "decl_kind": "theorem",
                "source": "discovered_theorems",
                "features": _features(fn),
            }
            n_catalog += 1

    # --- source scan (rich) ---
    n_source = 0
    root = args.source_root
    src_ok = bool(root) and os.path.isdir(root)
    if src_ok:
        for fp, rel in _iter_source_files(root, SOURCE_DIRS):
            for full, rel_path, stmt, kind in _scan_source_file(fp, rel):
                rec = {
                    "full_name": full,
                    "file_path": rel_path,
                    "namespace": full.rsplit(".", 1)[0] if "." in full else "",
                    "name_tokens": _tokens(full),
                    "statement_text": stmt or None,
                    "decl_kind": kind,
                    "source": "mathlib_source",
                    "features": _features((stmt or "") + " " + full),
                }
                # source record (has statement) supersedes a catalog-only record
                prev = index.get(full)
                if prev is None or prev.get("statement_text") is None:
                    index[full] = rec
                    if prev is None:
                        n_source += 1

    records = list(index.values())
    with_stmt = sum(1 for r in records if r["statement_text"])
    by_source = {}
    by_ns = {}
    for r in records:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        ns0 = (r["namespace"].split(".")[0] if r["namespace"] else "(root)")
        by_ns[ns0] = by_ns.get(ns0, 0) + 1

    os.makedirs(os.path.dirname(_p(args.out_index)), exist_ok=True)
    with open(_p(args.out_index), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "generated_by": "scripts/sf5_build_lemma_index.py",
        "source_root": root,
        "source_root_available": src_ok,
        "source_dirs_scanned": SOURCE_DIRS,
        "num_lemmas": len(records),
        "num_with_statement_text": with_stmt,
        "statement_coverage_pct": round(100.0 * with_stmt / max(1, len(records)), 1),
        "by_source": by_source,
        "top_namespaces": dict(sorted(by_ns.items(), key=lambda kv: -kv[1])[:15]),
        "limitations": [
            "discovered_theorems has no statement text -> weak name/token/path features",
            "source scan is restricted to focused directories relevant to the targets",
            "signatures parsed lexically up to ':='; complex multi-line binders may truncate",
        ],
    }
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)

    md = [
        "# SF5 lemma index — summary",
        "",
        f"- total indexed lemmas: **{len(records)}**",
        f"- with statement text: **{with_stmt}** ({summary['statement_coverage_pct']}%)",
        f"- source root available: {src_ok}",
        "",
        "## By source",
        "",
    ]
    for k, v in by_source.items():
        md.append(f"- {k}: {v}")
    md += ["", "## Top namespaces", ""]
    for k, v in summary["top_namespaces"].items():
        md.append(f"- {k}: {v}")
    md += ["", "## Limitations", ""] + [f"- {x}" for x in summary["limitations"]]
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")

    print(f"[sf5-index] {len(records)} lemmas, {with_stmt} with statement "
          f"({summary['statement_coverage_pct']}%), sources={by_source}")


if __name__ == "__main__":
    main()
