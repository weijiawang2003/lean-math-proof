#!/usr/bin/env python3
"""SF2 Part 2 — extract Mathlib source context + official proof for each Set failure.

We study the official proof to learn reusable PROBE families and to decide whether
a failure is a tactic/routing/search gap vs a genuine missing-lemma bridge. We do
NOT copy proofs into production.

For each selected case: locate the .lean source (traced LeanDojo cache / mathlib
checkout), extract the statement text, the official proof body, +/-N context lines,
neighbouring declaration names, imports, and nearby lemmas mentioning Set-shaped
keywords. Then classify the proof style.

Outputs:
  source_context.json
  source_context.md
Read-only on sources; writes only under sf2/out/set_cluster_deep_dive/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

CASES = "project/evolve/experiments/sf2/out/set_cluster_deep_dive/selected_cases.json"
OUT_JSON = "project/evolve/experiments/sf2/out/set_cluster_deep_dive/source_context.json"
OUT_MD = "project/evolve/experiments/sf2/out/set_cluster_deep_dive/source_context.md"

KEYWORDS = (
    "Set.ext", "ext ", "subset_def", "mem_setOf_eq", "mem_union", "mem_inter_iff",
    "mem_diff", "mem_compl_iff", "ite", "if_pos", "if_neg", "by_cases", "Decidable",
    "Classical", "SetLike.ext", "subset_antisymm", "constructor", "Set.ext_iff",
    "aesop",
)


def candidate_roots():
    roots = [
        os.path.expanduser("~/.cache/lean_dojo"),
        ".lake/packages/mathlib", "mathlib4", ".",
    ]
    for ev in ("LEAN_DOJO_CACHE_DIR", "MATHLIB_ROOT", "LEAN_SRC_PATH"):
        v = os.environ.get(ev)
        if v:
            roots.append(v)
    return [r for r in roots if r and os.path.isdir(r)]


def locate(file_path):
    """Return an absolute path to file_path's mathlib source, or None."""
    if not file_path:
        return None
    base = file_path  # e.g. Mathlib/Data/Set/Basic.lean
    for root in candidate_roots():
        # most specific first
        for pat in (os.path.join(root, "**", base),
                    os.path.join(root, "**", os.path.basename(base))):
            hits = glob.glob(pat, recursive=True)
            # prefer paths that contain the full logical suffix
            hits = sorted(hits, key=lambda h: (0 if h.endswith(base) else 1, len(h)))
            for h in hits:
                if os.path.isfile(h) and h.endswith(base):
                    return h
            if hits and hits[0].endswith(base):
                return hits[0]
    return None


DECL_RE = re.compile(r"^\s*(theorem|lemma|def|instance)\s+([A-Za-z0-9_'.]+)")


def find_decl(lines, short_name):
    """Return (start_idx, header_end_idx, proof_end_idx) for the decl, or None.
    header_end is the line index of `:= by` / `:=`; proof_end is exclusive."""
    pat = re.compile(rf"^\s*(?:@\[[^\]]*\]\s*)?(?:theorem|lemma)\s+{re.escape(short_name)}\b")
    for i, ln in enumerate(lines):
        if pat.search(ln):
            # header runs until a line containing ':=' (possibly multi-line sig)
            hend = i
            for j in range(i, min(i + 12, len(lines))):
                if ":=" in lines[j]:
                    hend = j
                    break
            # proof end: next top-level decl / #align / blank-before-decl
            pend = hend + 1
            k = hend + 1
            while k < len(lines):
                s = lines[k]
                if s.startswith("#align"):
                    pend = k
                    break
                if DECL_RE.match(s) and not s.startswith(" "):
                    pend = k
                    break
                if s.strip() == "" and k + 1 < len(lines) and (
                        DECL_RE.match(lines[k + 1]) or lines[k + 1].startswith("/--")
                        or lines[k + 1].startswith("section") or lines[k + 1].startswith("end ")):
                    pend = k
                    break
                k += 1
            else:
                pend = len(lines)
            return i, hend, pend
    return None


def classify_proof(stmt, proof):
    pt = proof.lower()
    has_ext = bool(re.search(r"\bext\b", proof)) or "set.ext" in pt
    has_bycases = "by_cases" in pt
    has_constructor = "constructor" in pt or "refine ⟨" in proof or "⟨fun" in proof
    has_antisymm = "antisymm" in pt
    has_simp_set = "set.ite" in pt or "set.ext_iff" in pt or bool(re.search(r"simp[^\n]*set\.", pt))
    has_aesop = "aesop" in pt
    has_rw = bool(re.search(r"\brw \[|\brw\[|\brewrite", proof))
    has_simp = bool(re.search(r"\bsimp\b", proof))
    has_obtain = "obtain" in pt or "rcases" in pt or "rfl |" in proof

    # decide a single dominant style
    if has_ext and has_bycases:
        style = "by_cases_ite_split"
    elif has_antisymm:
        style = "subset_antisymm"
    elif has_ext:
        style = "extensionality"
    elif has_rw and not has_simp:
        style = "rw_bridge"
    elif has_aesop and has_simp:
        style = "simp_only"  # simp [...] ; aesop closing
    elif has_aesop:
        style = "aesop_only"
    elif has_simp:
        style = "simp_only"
    elif has_constructor or has_obtain:
        style = "constructor_intro"
    elif ":= by" not in (stmt + proof) and ":=" in stmt:
        style = "manual_term_proof"
    else:
        style = "unknown"

    # likely reusable probe: the cheapest single-line tactic suggested by the proof
    if "simp [set.ite" in pt or "simp [Set.ite" in proof:
        probe = "simp [Set.ite]"
    elif has_simp_set and has_simp:
        m = re.search(r"simp(?:\s+only)?\s*\[[^\]]*\]", proof)
        probe = m.group(0) if m else "simp [Set.ext_iff]"
    elif has_ext and has_bycases:
        probe = "ext x <;> by_cases hx : x ∈ t <;> simp_all [Set.ite]"
    elif has_ext:
        probe = "ext x <;> simp_all"
    elif has_aesop and has_simp:
        m = re.search(r"simp(?:\s+only)?\s*\[[^\]]*\]", proof)
        probe = (m.group(0) + " <;> aesop") if m else "simp_all <;> aesop"
    elif has_rw:
        probe = "rw-bridge (theorem-specific; see official proof)"
    else:
        probe = "aesop"

    # missing-lemma heuristic: rw-bridge that depends on a *named* non-mem lemma
    # that is itself Set-specific and not a simp lemma → possible bridge.
    likely_missing = False
    notes = []
    if style == "rw_bridge":
        notes.append("rewrite-bridge proof: depends on specific named lemmas, "
                     "not closed by generic simp/aesop")
    if "simp [set.ite" in pt:
        notes.append("one-line `simp [Set.ite]`: RC1 simp does not unfold the "
                     "irreducible Set.ite by default -> tactic gap")
    if has_bycases and has_ext:
        notes.append("ext + by_cases on membership in t, then simp with hypotheses")
    if has_aesop:
        notes.append("official proof itself ends in aesop")

    return {
        "proof_style": style,
        "uses_by_cases": has_bycases,
        "uses_ext": has_ext,
        "uses_constructor": has_constructor,
        "uses_subset_antisymm": has_antisymm,
        "uses_simp_set": has_simp_set,
        "uses_aesop": has_aesop,
        "uses_rw": has_rw,
        "likely_reusable_probe": probe,
        "likely_missing_lemma": likely_missing,
        "notes": notes,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default=CASES)
    p.add_argument("--context-lines", type=int, default=120)
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-md", default=OUT_MD)
    args = p.parse_args(argv)

    cases = json.load(open(args.cases))["selected"]
    N = args.context_lines
    results = []
    for c in cases:
        name = c["full_name"]
        short = name.split(".")[-1]
        fp = c["file_path"]
        abspath = locate(fp)
        rec = {"full_name": name, "file_path": fp, "located_at": abspath,
               "statement": None, "official_proof": None, "proof_line": None,
               "neighbor_decls": [], "imports": [], "nearby_lemmas": [],
               "classification": None}
        if not abspath:
            rec["error"] = "source file not located in any candidate root"
            results.append(rec)
            continue
        lines = open(abspath, encoding="utf-8").read().splitlines()
        rec["imports"] = [l for l in lines[:80] if l.startswith("import ")][:40]
        found = find_decl(lines, short)
        if not found:
            rec["error"] = f"declaration `{short}` not found in {fp}"
            results.append(rec)
            continue
        i, hend, pend = found
        rec["proof_line"] = i + 1
        rec["statement"] = "\n".join(lines[i:hend + 1])
        rec["official_proof"] = "\n".join(lines[i:pend]).strip()
        lo, hi = max(0, i - N), min(len(lines), pend + N)
        ctx = lines[lo:hi]
        rec["context_window_lines"] = [lo + 1, hi]
        # neighbor decls in window
        neigh = []
        for ln in ctx:
            m = DECL_RE.match(ln)
            if m and m.group(2).split(".")[-1] != short:
                neigh.append(m.group(2))
        rec["neighbor_decls"] = neigh[:60]
        # nearby lemmas mentioning Set keywords (decl lines only)
        nearby = []
        for idx in range(lo, hi):
            ln = lines[idx]
            if DECL_RE.match(ln) and any(k in ln for k in KEYWORDS):
                nearby.append({"line": idx + 1, "decl": ln.strip()[:120]})
        rec["nearby_lemmas"] = nearby[:40]
        rec["classification"] = classify_proof(rec["statement"], rec["official_proof"])
        results.append(rec)

    out = {"context_lines": N, "num_cases": len(results), "cases": results}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), ensure_ascii=False, indent=2)

    # markdown
    L = ["# SF2 Set Cluster — Source Context & Official-Proof Analysis", ""]
    styles = {}
    for r in results:
        cl = r.get("classification") or {}
        styles[cl.get("proof_style", "n/a")] = styles.get(cl.get("proof_style", "n/a"), 0) + 1
    L.append(f"- cases: {len(results)} | context lines: ±{N}")
    L.append(f"- proof-style histogram: `{styles}`")
    L.append("")
    for r in results:
        L.append(f"## `{r['full_name']}`")
        L.append(f"- file: `{r['file_path']}` (line {r.get('proof_line')})")
        if r.get("error"):
            L.append(f"- **ERROR**: {r['error']}")
            L.append("")
            continue
        cl = r["classification"]
        L.append(f"- proof_style: **{cl['proof_style']}** | "
                 f"ext={cl['uses_ext']} by_cases={cl['uses_by_cases']} "
                 f"rw={cl['uses_rw']} simp_set={cl['uses_simp_set']} aesop={cl['uses_aesop']}")
        L.append(f"- likely_reusable_probe: `{cl['likely_reusable_probe']}`")
        if cl["notes"]:
            L.append(f"- notes: {'; '.join(cl['notes'])}")
        L.append("")
        L.append("```lean")
        L.append(r["official_proof"])
        L.append("```")
        L.append("")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[sf2:source] cases={len(results)} located="
          f"{sum(1 for r in results if r.get('located_at'))} "
          f"styles={styles} -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
