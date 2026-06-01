#!/usr/bin/env python3
"""SX4 — normalized trace sequence detector (heuristic).

Given a literal-production results JSON (with a `trace_path`), detect whether the
production run ALREADY contains a sequence equivalent to "A-advanced state -> B-close"
for a set of known patterns. This is the trace-side evidence for PRODUCTION_SUBSUMED
in sx4_sequence_attribution.py.

Supported patterns (A_advances_then_B_closes):
  simp [Set.ite]  -> aesop
  simp [Set.ite]  -> simp_all
  ext             -> aesop          (ext / ext x / ext1 ...)
  constructor     -> aesop          (constructor / intro / rintro ...)

Confidence per theorem/pattern:
  exact         A advanced (TacticState, no error) at step i AND B closed (ProofFinished)
                at step j>i — a concrete A->B continuation in the trace
  likely        A advanced and B closed for the theorem, ordering not strictly established
  weak          only one side present (A advanced, or B closed, but not both)
  insufficient  no trace rows for the theorem / no trace file

Heuristic only — DO NOT make release decisions solely on `weak`/`likely` detection.
The authoritative credit decision is sx4_sequence_attribution.py (which also requires
baseline_finished and depth-1 control failure).
"""
from __future__ import annotations

import argparse
import json
import os

# (label, A-prefix matcher, B exact closers)
PATTERNS = [
    {"id": "simp_set_ite__aesop", "A_prefixes": ["simp [Set.ite]"], "B": ["aesop"]},
    {"id": "simp_set_ite__simp_all", "A_prefixes": ["simp [Set.ite]"], "B": ["simp_all"]},
    {"id": "ext__aesop", "A_prefixes": ["ext", "ext ", "ext1"], "B": ["aesop"]},
    {"id": "constructor_intro__aesop",
     "A_prefixes": ["constructor", "intro", "rintro"], "B": ["aesop"]},
]


def _matches_A(tac, prefixes):
    if not tac:
        return False
    return any(tac == pre or tac.startswith(pre + " ") or tac == pre.strip() for pre in prefixes)


def _load_trace(path):
    rows = {}
    if not path or not os.path.isfile(path):
        return rows
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        rows.setdefault(rec.get("full_name"), []).append(rec)
    return rows


def _detect(rows, pat):
    a_adv_steps, b_close_steps = [], []
    for r in rows:
        tac = r.get("tactic")
        if _matches_A(tac, pat["A_prefixes"]) and not r.get("proof_finished") \
                and r.get("result_kind") == "TacticState":
            a_adv_steps.append(r.get("step"))
        if tac in pat["B"] and r.get("proof_finished"):
            b_close_steps.append(r.get("step"))
    if a_adv_steps and b_close_steps:
        ordered = min(s for s in a_adv_steps if s is not None) < max(s for s in b_close_steps if s is not None)
        return ("exact" if ordered else "likely", a_adv_steps, b_close_steps)
    if a_adv_steps or b_close_steps:
        return ("weak", a_adv_steps, b_close_steps)
    return ("insufficient", a_adv_steps, b_close_steps)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-results", required=True)
    p.add_argument("--out-json", required=True)
    args = p.parse_args(argv)

    base = json.load(open(args.baseline_results))
    trace_path = base.get("trace_path")
    trace = _load_trace(trace_path)
    finished = {r["full_name"]: bool(r.get("finished")) for r in base.get("per_theorem", [])}

    detections, hist = [], {}
    for fn in sorted(set(trace) | set(finished)):
        rows = trace.get(fn, [])
        per_pat = {}
        best = "insufficient"
        order = {"exact": 3, "likely": 2, "weak": 1, "insufficient": 0}
        for pat in PATTERNS:
            conf, a_steps, b_steps = _detect(rows, pat)
            per_pat[pat["id"]] = {"confidence": conf, "A_advance_steps": a_steps,
                                  "B_close_steps": b_steps}
            if order[conf] > order[best]:
                best = conf
        hist[best] = hist.get(best, 0) + 1
        detections.append({"theorem": fn, "baseline_finished": finished.get(fn),
                           "best_confidence": best, "patterns": per_pat})

    out = {
        "baseline_results": args.baseline_results,
        "baseline_trace": trace_path,
        "patterns_supported": [p["id"] for p in PATTERNS],
        "num_theorems": len(detections),
        "best_confidence_histogram": hist,
        "exact_subsumption_theorems": sorted(d["theorem"] for d in detections
                                             if d["best_confidence"] == "exact"),
        "caveat": "Heuristic. Do NOT make release decisions solely on weak/likely detection; "
                  "authoritative credit = scripts/sx4_sequence_attribution.py.",
        "detections": detections,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)
    print(f"[sx4-trace] theorems={len(detections)} hist={hist} "
          f"exact_subsumption={len(out['exact_subsumption_theorems'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
