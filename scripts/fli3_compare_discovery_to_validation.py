#!/usr/bin/env python3
"""FLI3 Part 9 — compare FLI2 discovery to FLI3 literal validation."""
from __future__ import annotations
import argparse, json, os
from collections import Counter
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _p(*a): return os.path.join(_REPO, *a)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fli2-attribution", required=True); ap.add_argument("--fli2-rules", required=True)
    ap.add_argument("--fli3-attribution", required=True); ap.add_argument("--fli3-results", required=True)
    ap.add_argument("--out-json", required=True); ap.add_argument("--out-md", required=True)
    a = ap.parse_args()
    f2 = [json.loads(l) for l in open(_p(a.fli2_attribution)) if l.strip()]
    f2_true = sorted({r["theorem"] for r in f2 if r["classification"] == "TRUE_RETRIEVAL_GAP_RESCUE"})
    f3 = json.load(open(_p(a.fli3_attribution)))
    f3_true = set(f3["true_delta_theorems"])
    reproduced = [t for t in f2_true if t in f3_true]
    not_repro = [t for t in f2_true if t not in f3_true]
    # holdout wins = true deltas not in the FLI2 rescue set
    f3recs = f3["records"]
    holdout_true = sorted({r["theorem"] for r in f3recs
                           if r["classification"] == "TRUE_FLI3_DELTA" and r["set"] == "family_holdout"})
    out = {"generated_by": "scripts/fli3_compare_discovery_to_validation.py",
           "fli2_true_rescue_count": len(f2_true), "fli2_true_rescues": f2_true,
           "fli3_true_delta_count": f3["true_fli3_delta"], "fli3_true_deltas": sorted(f3_true),
           "fli2_rescues_reproduced_under_literal_rc2": reproduced,
           "fli2_rescues_not_reproduced": not_repro,
           "reproduction_rate": round(len(reproduced) / len(f2_true), 3) if f2_true else None,
           "family_holdout_generalization_wins": holdout_true,
           "num_family_holdout_wins": len(holdout_true),
           "families_surviving": sorted(set(f3["true_delta_by_family"])),
           "control_duplicates": f3.get("control_duplicates", 0),
           "unknown_name_gaps": f3.get("unknown_name", 0),
           "narrative": (
               f"FLI2 discovered {len(f2_true)} robust rescues; under literal-RC2 additive "
               f"validation {len(reproduced)} reproduce as TRUE_FLI3_DELTA "
               f"({'all' if len(reproduced)==len(f2_true) else f'{len(reproduced)}/{len(f2_true)}'}). "
               f"Family holdout produced {len(holdout_true)} additional generalization wins, so "
               f"failure-derived discovery {'DOES' if holdout_true else 'does NOT yet'} generalize "
               f"beyond the exact discovered theorems. Failure-derived discovery "
               f"{'CAN' if reproduced else 'cannot yet'} feed RC-style validation.")}
    json.dump(out, open(_p(a.out_json), "w"), indent=2)
    md = ["# FLI3 discovery → validation comparison", "",
          f"- FLI2 true rescues: {len(f2_true)} → FLI3 TRUE_FLI3_DELTA: {f3['true_fli3_delta']}",
          f"- **reproduced under literal RC2: {len(reproduced)}/{len(f2_true)} "
          f"(rate {out['reproduction_rate']})**",
          f"- not reproduced: {not_repro}",
          f"- **family-holdout generalization wins: {len(holdout_true)}** {holdout_true}",
          f"- families surviving: {out['families_surviving']}",
          f"- control-duplicates: {out['control_duplicates']} | unknown-name gaps: {out['unknown_name_gaps']}",
          "", "## Narrative", "", out["narrative"]]
    open(_p(a.out_md), "w").write("\n".join(md) + "\n")
    print(f"[fli3-compare] reproduced={len(reproduced)}/{len(f2_true)} holdout_wins={len(holdout_true)}")
if __name__ == "__main__": main()
