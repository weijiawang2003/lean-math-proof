#!/usr/bin/env python3
"""SF2 Part 2 — extract Mathlib source context for each failure theorem.

For each failure case, locate the Lean source file (the declaration's official
Mathlib proof may be present — that is fine; we learn the proof *pattern*), and
extract: ±N context lines, the statement text, the existing proof, neighbouring
declaration names, imports, and nearby lemmas mentioning toFinset/disjoint/
nsmul/singleton.

Source files live in a mathlib checkout / the traced LeanDojo cache, not in this
repo. The extractor tries several candidate roots and a bounded `find`; if a file
cannot be located it records that precisely rather than inventing context.

Outputs:
  source_context.json
  source_context.md

SAFETY: read-only; writes only under project/evolve/experiments/sf2/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

CASES = "project/evolve/experiments/sf2/out/multiset_seed/failure_cases.json"
KEYWORDS = ("toFinset", "disjoint", "Disjoint", "nsmul", "singleton")


def _candidate_roots():
    roots = ["", ".", "mathlib4", ".lake/packages/mathlib",
             os.path.expanduser("~/.cache/lean_dojo")]
    # env hints
    for ev in ("LEAN_DOJO_CACHE_DIR", "MATHLIB_ROOT", "LEAN_SRC_PATH"):
        v = os.environ.get(ev)
        if v:
            roots.append(v)
    return [r for r in roots if r is not None]


def _locate(file_path, extra_root=None, timeout=60):
    """Return an absolute path to file_path's source, or None."""
    if not file_path:
        return None
    cands = []
    roots = _candidate_roots()
    if extra_root:
        roots.insert(0, extra_root)
    for r in roots:
        j = os.path.join(r, file_path) if r else file_path
        if os.path.isfile(j):
            cands.append(j)
    if cands:
        return cands[0]
    # bounded find fallback under the lean_dojo cache (basename + path suffix)
    cache = os.path.expanduser("~/.cache/lean_dojo")
    if os.path.isdir(cache):
        base = os.path.basename(file_path)
        try:
            r = subprocess.run(["find", cache, "-name", base, "-path",
                                "*/" + file_path], capture_output=True, text=True,
                               timeout=timeout)
            hits = [h for h in (r.stdout or "").splitlines() if h.strip()]
            if hits:
                return sorted(hits, key=len)[0]
        except Exception:
            pass
    return None


def _short_name(full_name):
    return full_name.split(".")[-1] if full_name else full_name


def _extract(full_name, src_path, context_lines):
    with open(src_path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    short = _short_name(full_name)
    # find declaration line: `theorem|lemma|def <short>` (Mathlib uses short name
    # inside `namespace Multiset`); fall back to full name.
    decl_re = re.compile(rf"^\s*(theorem|lemma|def|@\[.*\]\s*(theorem|lemma|def))\s+{re.escape(short)}\b")
    full_re = re.compile(rf"\b{re.escape(full_name)}\b")
    idx = None
    for i, ln in enumerate(lines):
        if decl_re.search(ln):
            idx = i
            break
    if idx is None:
        for i, ln in enumerate(lines):
            if full_re.search(ln) and re.search(r"\b(theorem|lemma|def)\b", ln):
                idx = i
                break
    imports = [ln.strip() for ln in lines[:80] if ln.strip().startswith("import")]
    nearby_kw = []
    for i, ln in enumerate(lines):
        if any(k in ln for k in KEYWORDS) and re.search(r"\b(theorem|lemma)\b", ln):
            m = re.search(r"\b(?:theorem|lemma)\s+([A-Za-z0-9_'.]+)", ln)
            if m:
                nearby_kw.append(m.group(1))
    result = {
        "source_found": True, "source_path": src_path,
        "declaration_line": (idx + 1) if idx is not None else None,
        "imports_head": imports[:20],
        "nearby_keyword_lemmas": sorted(set(nearby_kw))[:40],
    }
    if idx is None:
        result["statement_text"] = None
        result["proof_text"] = None
        result["context_block"] = None
        result["notes"] = ["declaration not located by name inside file"]
        return result
    lo = max(0, idx - context_lines)
    hi = min(len(lines), idx + context_lines + 1)
    block = "".join(lines[lo:hi])
    # statement = decl line through first ':=' or 'by' boundary
    stmt, proof = [], []
    in_proof = False
    for ln in lines[idx: min(len(lines), idx + 60)]:
        if not in_proof:
            stmt.append(ln)
            if ":=" in ln or re.search(r":=\s*by\b", ln) or ln.rstrip().endswith("by"):
                in_proof = True
            elif re.match(r"^\s*(theorem|lemma|def|namespace|end|@\[)", ln) and proof:
                break
        else:
            if re.match(r"^\s*(theorem|lemma|def|namespace|end|@\[)\S", ln):
                break
            proof.append(ln)
    # neighbouring declarations within the window
    neigh = []
    for ln in lines[lo:hi]:
        m = re.search(r"^\s*(?:@\[[^\]]*\]\s*)?(?:theorem|lemma|def)\s+([A-Za-z0-9_'.]+)", ln)
        if m:
            neigh.append(m.group(1))
    result.update({
        "statement_text": "".join(stmt).strip(),
        "proof_text": "".join(proof).strip() or None,
        "context_block": block,
        "neighboring_decls": neigh,
        "notes": [],
    })
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SF2: extract source context.")
    p.add_argument("--cases", default=CASES)
    p.add_argument("--context-lines", type=int, default=80)
    p.add_argument("--mathlib-root", default=None)
    p.add_argument("--out-json",
                   default="project/evolve/experiments/sf2/out/multiset_seed/source_context.json")
    p.add_argument("--out-md",
                   default="project/evolve/experiments/sf2/out/multiset_seed/source_context.md")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.cases):
        print(f"[sf2:src] ERROR: cases not found: {args.cases}", file=sys.stderr)
        return 2
    cases = json.load(open(args.cases)).get("cases", [])
    out = []
    for c in cases:
        fn, fp = c["full_name"], c.get("file_path")
        src = _locate(fp, args.mathlib_root)
        rec = {"full_name": fn, "file_path": fp}
        if not src:
            rec.update({"source_found": False, "source_path": None,
                        "notes": [f"source file not located for {fp!r}; tried cwd, "
                                  f"mathlib4, .lake, ~/.cache/lean_dojo and bounded find"]})
        else:
            rec.update(_extract(fn, src, args.context_lines))
        out.append(rec)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump({"context_lines": args.context_lines, "theorems": out},
              open(args.out_json, "w"), ensure_ascii=False, indent=2)

    md = ["# SF2 Multiset seed — source context", ""]
    for r in out:
        md.append(f"## `{r['full_name']}`")
        md.append("")
        md.append(f"- file: `{r.get('file_path')}`  | source_found: `{r.get('source_found')}`"
                  + (f"  | decl line: {r.get('declaration_line')}" if r.get('source_found') else ""))
        if r.get("source_found"):
            md.append(f"- nearby keyword lemmas: {r.get('nearby_keyword_lemmas')}")
            md.append("")
            md.append("Statement:")
            md.append("```lean")
            md.append((r.get("statement_text") or "<not located>").strip())
            md.append("```")
            if r.get("proof_text"):
                md.append("Existing Mathlib proof:")
                md.append("```lean")
                md.append(r["proof_text"].strip())
                md.append("```")
        else:
            md.append(f"- {r.get('notes')}")
        md.append("")
    open(args.out_md, "w").write("\n".join(md))

    found = sum(1 for r in out if r.get("source_found"))
    print(f"[sf2:src] {found}/{len(out)} sources located -> {args.out_json}")
    for r in out:
        print(f"  {r['full_name']}: found={r.get('source_found')} "
              f"path={r.get('source_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
