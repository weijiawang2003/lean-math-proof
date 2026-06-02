#!/usr/bin/env python3
"""FLI1 Part 7 — typecheck candidate lemma statements with `lake env lean`.

Writes a temp Lean file (`import <module>` + `open <ns>` + `lemma … := by sorry`) and elaborates
it against the compiled Mathlib oleans in the traced cache. TYPECHECKS iff the only diagnostic is
the `sorry` warning. Errors are classified. Temp files only; Mathlib source is never modified.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import tempfile
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _mathlib_root():
    cands = glob.glob(os.path.expanduser("~/.cache/lean_dojo/*/mathlib4"))
    cands = [c for c in cands if os.path.exists(os.path.join(c, "lakefile.lean"))]
    if not cands:
        raise RuntimeError("no traced mathlib4 root found")
    return sorted(cands)[0]


def classify_lean_output(rc, text):
    t = text.lower()
    if rc == 0 and ("declaration uses 'sorry'" in t or "sorry" in t) and "error" not in t:
        return "TYPECHECKS"
    if rc == 0 and "error" not in t:
        return "TYPECHECKS"  # (no sorry warning emitted but elaborated clean)
    if "unknown identifier" in t or "unknown constant" in t:
        return "UNKNOWN_CONSTANT"
    if "unknown module" in t or "unknown package" in t or "no such file" in t \
            or ("import" in t and "not found" in t):
        return "MISSING_IMPORT"
    if "failed to synthesize" in t or "typeclass" in t or "universe" in t:
        return "UNIVERSE_OR_TYPECLASS_ERROR"
    if "unexpected token" in t or "unexpected identifier" in t or "expected ofnat" in t \
            or "binderident" in t or "unexpected" in t:
        return "BINDER_ERROR"
    if "type mismatch" in t or "function expected" in t or "type expected" in t \
            or "error" in t:
        return "TYPE_ERROR"
    return "NEEDS_REVIEW"


def run_lean(mroot, body, timeout=120):
    with tempfile.NamedTemporaryFile("w", suffix=".lean", dir="/tmp", delete=False) as tf:
        tf.write(body)
        path = tf.name
    try:
        proc = subprocess.run(["lake", "env", "lean", path], cwd=mroot,
                              capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout + "\n" + proc.stderr)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def build_body(cand, proof="sorry"):
    imports = cand.get("required_imports") or []
    opens = cand.get("open_namespaces") or []
    lines = [f"import {m}" for m in imports if m]
    if not lines:
        lines = ["import Mathlib"]
    for ns in opens:
        if ns:
            lines.append(f"open {ns}")
    stmt = cand["lemma_statement_lean"]
    stmt = re.sub(r":=\s*by\s+sorry\s*$", "", stmt).rstrip()
    stmt = re.sub(r":=\s*sorry\s*$", "", stmt).rstrip()
    lines.append(f"{stmt} := by {proof}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    mroot = _mathlib_root()
    cands = [json.loads(l) for l in open(_p(args.candidates)) if l.strip()]
    out = []
    for c in cands:
        body = build_body(c, "sorry")
        rc, text = run_lean(mroot, body, args.timeout)
        cls = "TIMEOUT" if rc == 124 else classify_lean_output(rc, text)
        rec = dict(c)
        rec.update({"typecheck": cls,
                    "typecheck_diagnostic": "\n".join(
                        ln for ln in text.splitlines()
                        if "error" in ln.lower() or "sorry" in ln.lower())[:400]})
        out.append(rec)
        print(f"[fli1-typecheck] {c['candidate_id']} {c['source_seed_ids'][0]} -> {cls}", flush=True)

    with open(_p(args.out_jsonl), "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hist = Counter(r["typecheck"] for r in out)
    summary = {"generated_by": "scripts/fli1_typecheck_candidate_lemmas.py",
               "mathlib_root": mroot, "num_candidates": len(out),
               "typecheck_histogram": dict(hist),
               "typechecks": hist.get("TYPECHECKS", 0),
               "target_met_10": hist.get("TYPECHECKS", 0) >= 10,
               "typecheck_targets": sorted({r["downstream_targets"][0] for r in out
                                            if r["typecheck"] == "TYPECHECKS"})}
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI1 typecheck summary", "",
          f"- candidates: {summary['num_candidates']} | **TYPECHECKS: {summary['typechecks']}** "
          f"(target ≥10: {summary['target_met_10']})",
          f"- histogram: {summary['typecheck_histogram']}", "",
          "| id | seed | typecheck | diagnostic |", "|---|---|---|---|"]
    for r in out:
        md.append(f"| {r['candidate_id']} | {r['source_seed_ids'][0]} | {r['typecheck']} | "
                  f"`{(r['typecheck_diagnostic'] or '')[:80].replace(chr(10),' ')}` |")
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli1-typecheck] TYPECHECKS={summary['typechecks']}/{len(out)} hist={dict(hist)}")


if __name__ == "__main__":
    main()
