#!/usr/bin/env python3
"""FLI1 Part 8 — try to prove TYPECHECKS candidate lemmas with safe tactics via `lake env lean`.

Only attempts TYPECHECKS candidates with confidence high/medium and risk low/medium. Tries an
ordered, pattern-aware list of SAFE tactics (no simp_all, no bare broad aesop as the credited
proof, no long search). First accepting proof wins. A proved candidate is NOT a project success
until it rescues downstream (Part 9).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fli1_typecheck_candidate_lemmas as TC  # reuse run_lean / build_body / classify

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _proof_attempts(c):
    """Ordered safe proof tactics (each a single `by`-body). Pattern + closest-lemma aware."""
    pat = c["pattern"]
    L = c.get("closest_existing_lemma")
    goal = c.get("lemma_goal", "")
    attempts = ["simp"]
    if L:
        # the spec allows deploying / applying the specific close existing lemma
        attempts += [f"simp [{L}]", f"exact {L}", f"apply {L}", f"exact {L} h",
                     f"simpa using {L}"]
    # gcongr: safe, bounded congruence-monotonicity for ⊆ / ≤ / < bridge goals
    if any(r in goal for r in ("⊆", "≤", "<", "⊂")):
        attempts.append("gcongr")
    if pat in ("IFF_SPLIT", "MEMBERSHIP_BRIDGE", "SINGLETON_CHARACTERIZATION"):
        attempts += ["constructor <;> intro h <;> simp at *"]
        if L:
            attempts.append(f"constructor <;> intro h <;> simp [{L}] at *")
    if pat in ("EXTENSIONALITY_NEEDED", "SUBSET_BRIDGE"):
        attempts += ["ext x <;> simp", "intro x hx <;> simp at *"]
        if L:
            attempts.append(f"ext x <;> simp [{L}]")
    if pat == "MAP_FILTER_BIND_BRIDGE" and L:
        attempts.append(f"simp only [{L}]")
    # simp-then-aesop (aesop only after a specific simp, never bare broad aesop)
    attempts.append("simp <;> aesop")
    if L:
        attempts.append(f"simp [{L}] <;> aesop")
    if c["namespace"] == "Nat" or "Nat" in c.get("lemma_goal", ""):
        attempts.append("omega")
    # dedup preserve order
    seen, out = set(), []
    for a in attempts:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    mroot = TC._mathlib_root()
    cands = [json.loads(l) for l in open(_p(args.candidates)) if l.strip()]
    out = []
    for c in cands:
        rec = dict(c)
        eligible = (c.get("typecheck") == "TYPECHECKS"
                    and c.get("confidence") in ("high", "medium")
                    and c.get("risk") in ("low", "medium"))
        if not eligible:
            rec.update({"prove": "SKIPPED", "proof_tactic": None, "proof_script": None,
                        "proof_runtime_sec": None, "proof_trivial": None})
            out.append(rec)
            continue
        import time
        proved = None
        attempts_log = []
        for tac in _proof_attempts(c):
            body = TC.build_body(c, tac)
            t0 = time.time()
            rcode, text = TC.run_lean(mroot, body, args.timeout)
            dt = round(time.time() - t0, 2)
            tl = text.lower()
            ok = (rcode == 0 and "error" not in tl and "sorry" not in tl)
            attempts_log.append({"tactic": tac, "ok": ok, "runtime": dt})
            if ok:
                proved = tac
                break
            if rcode == 124:
                attempts_log[-1]["timeout"] = True
        if proved is not None:
            rec.update({"prove": "PROVED", "proof_tactic": proved,
                        "proof_script": f"by {proved}",
                        "proof_runtime_sec": sum(a["runtime"] for a in attempts_log),
                        "proof_trivial": proved in ("simp", "rfl", "omega"),
                        "proof_is_existing": bool(c.get("retrieval_gap")),
                        "proof_attempts": attempts_log})
        else:
            any_to = any(a.get("timeout") for a in attempts_log)
            rec.update({"prove": "TIMEOUT" if any_to else "FAILED", "proof_tactic": None,
                        "proof_script": None, "proof_attempts": attempts_log,
                        "proof_runtime_sec": sum(a["runtime"] for a in attempts_log)})
        out.append(rec)
        print(f"[fli1-prove] {c['candidate_id']} {c['source_seed_ids'][0]} -> {rec['prove']}"
              + (f" via `{proved}`" if proved else ""), flush=True)

    with open(_p(args.out_jsonl), "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hist = Counter(r["prove"] for r in out)
    proved = [r for r in out if r["prove"] == "PROVED"]
    summary = {"generated_by": "scripts/fli1_prove_candidate_lemmas.py",
               "num_candidates": len(out), "prove_histogram": dict(hist),
               "proved": len(proved),
               "proved_nontrivial": sum(1 for r in proved if not r.get("proof_trivial")),
               "proved_existing": sum(1 for r in proved if r.get("proof_is_existing")),
               "proved_targets": sorted({r["downstream_targets"][0] for r in proved}),
               "proof_tactic_histogram": dict(Counter(r["proof_tactic"] for r in proved))}
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI1 candidate lemma proof summary", "",
          f"- candidates: {summary['num_candidates']} | **PROVED: {summary['proved']}** "
          f"(nontrivial {summary['proved_nontrivial']}, already-existing {summary['proved_existing']})",
          f"- histogram: {summary['prove_histogram']}",
          f"- proof tactics: {summary['proof_tactic_histogram']}", "",
          "| id | seed | prove | tactic | trivial |", "|---|---|---|---|---|"]
    for r in out:
        md.append(f"| {r['candidate_id']} | {r['source_seed_ids'][0]} | {r['prove']} | "
                  f"`{r.get('proof_tactic') or ''}` | {r.get('proof_trivial')} |")
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli1-prove] PROVED={summary['proved']}/{len(out)} hist={dict(hist)}")


if __name__ == "__main__":
    main()
