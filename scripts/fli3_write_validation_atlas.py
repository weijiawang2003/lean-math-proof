#!/usr/bin/env python3
"""FLI3 Part 10 — validation atlas (md + json)."""
from __future__ import annotations
import argparse, json, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _p(*a): return os.path.join(_REPO, *a)
def _L(p): return json.load(open(_p(p)))
def main():
    ap = argparse.ArgumentParser()
    for x in ("sets","rc2-results","candidate-results","attribution","safety","comparison"):
        ap.add_argument("--"+x, required=True)
    ap.add_argument("--out-md", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    sets=_L(a.sets); rc2=_L(a.rc2_results); cand=_L(a.candidate_results)
    attr=_L(a.attribution); safe=_L(a.safety); comp=_L(a.comparison)
    true_delta=[r for r in attr["records"] if r["classification"]=="TRUE_FLI3_DELTA"]
    atlas={"generated_by":"scripts/fli3_write_validation_atlas.py",
        "validation_sets":{ "num_items":sets["num_items"]},
        "literal_rc2":rc2["status_histogram"],
        "candidate_wins":cand["candidate_wins"],"robust_wins":cand["robust_wins"],
        "true_fli3_delta":attr["true_fli3_delta"],
        "true_delta_by_family":attr["true_delta_by_family"],
        "rescue_replay_reproduced":attr["rescue_replay_reproduced"],
        "family_holdout_wins":attr["family_holdout_wins"],
        "safety_verdict":safe["verdict"],"offgate_emissions":safe["offgate_emissions"],
        "regressions":safe["regressions"],
        "reproduction_rate":comp["reproduction_rate"],
        "best_examples":[{"theorem":r["theorem"],"family":r["candidate_family"],"lemma":r["lemma"],
            "tactic":r["tactic"],"set":r["set"],"explanation":r.get("explanation")} for r in true_delta]}
    json.dump(atlas, open(_p(a.out_json),"w"), indent=2)
    md=["# FLI3 Validation Atlas","",
        "## 1. Overview",
        f"Validated {sets['num_items']} items; literal RC2 {rc2['status_histogram']}; "
        f"**TRUE_FLI3_DELTA {attr['true_fli3_delta']}** (rescue_replay {attr['rescue_replay_reproduced']}/6, "
        f"family_holdout {attr['family_holdout_wins']}).","",
        "## 2. Why FLI3 follows FLI2",
        "FLI2 discovered 6 robust at-position rescues; FLI3 tests whether they survive RC-style "
        "literal-RC2 additive validation and whether the families generalize.","",
        "## 3. Candidate families","FINSET_CARD_BRIDGE, FINSET_MEM_DEF_UNFOLD, LIST_DEF_UNFOLD.","",
        "## 4. Validation sets",f"{sets['num_items']} items across rescue_replay/family_holdout/"
        "offgate_negative/canonical_floor/regression_guard.","",
        "## 5. Literal RC2 baseline",f"{rc2['status_histogram']}","",
        "## 6. Candidate evaluation",
        f"candidate wins {cand['candidate_wins']} (robust {cand['robust_wins']}); "
        f"offgate emissions {cand['offgate_emissions']}; regressions {cand['regressions']}.","",
        "## 7. Attribution",f"{attr['classification_histogram']}","",
        "## 8. Safety / offgate / determinism",
        f"verdict {safe['verdict']}; offgate {safe['offgate_emissions']}; regressions {safe['regressions']}; "
        f"vacuous {safe['vacuous_wins']}.","",
        "## 9. Discovery → validation comparison",comp["narrative"],"",
        "## 10. Best validated examples",""]
    for r in true_delta:
        md+= [f"### `{r['theorem']}` ({r['set']}, {r['candidate_family']})",
              f"- deploy `{r['lemma']}` via `{r['tactic']}`; {r.get('explanation')}",""]
    md+=["## 11. Rejected or fragile examples","",
        f"- control-duplicates: {attr.get('control_duplicates',0)}; flakes: {attr.get('flakes',0)}; "
        f"unknown-name/import gaps: {attr.get('unknown_name',0)}.","",
        "## 12. Recommended FLI4","",
        "- Push validated families through full RC4-style floor benchmark + schema wrapper run; "
        "tighten gates for family generalization; address import-gap cases."]
    open(_p(a.out_md),"w").write("\n".join(md)+"\n")
    print(f"[fli3-atlas] true_delta={attr['true_fli3_delta']} verdict={safe['verdict']}")
if __name__ == "__main__": main()
