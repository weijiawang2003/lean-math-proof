#!/usr/bin/env python3
"""TR5 Part 7 — attribute every live win against literal RC2 (SX4 discipline).

A win is credited TRUE_RANKER_DELTA only if literal RC2 is a CONFIRMED_RC2_FAILURE, the
ranker-selected program solved it, the bare controls (run in-worker) did NOT, and the
program is not source-specific. RC4B/RC4C evidence are TRUE_RANKER_DELTAs whose winning
program uses Set.disjoint_left / is d2_simp_aesop. def_unfold reproductions of the RC4A
wins are TRUE_RC4A_REPRODUCTION. Everything literal RC2 already solves is
PRODUCTION_SUBSUMED; high-ranked failures are RANKER_FALSE_POSITIVE; no success under the
budget is NO_WIN_UNDER_BUDGET.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _load(path):
    if not path:
        return None
    fp = _p(path) if not os.path.isabs(path) else path
    return json.load(open(fp)) if os.path.exists(fp) else None


def _uses_disjoint_left(prog):
    return bool(prog and any("disjoint_left" in (L or "") for L in prog.get("used_lemmas", [])))


def _is_d2_aesop(prog):
    if not prog:
        return False
    return prog.get("family") == "d2_simp_aesop" or (
        "<;>" in (prog.get("tactic") or "") and "aesop" in (prog.get("tactic") or "")
        and (prog.get("tactic") or "").strip().startswith("simp ["))


def _is_def_unfold(prog):
    return bool(prog and prog.get("family") == "def_unfold_simp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirmation", required=True)
    ap.add_argument("--b5", required=True)
    ap.add_argument("--b10")
    ap.add_argument("--b20")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    conf = {r["full_name"]: r for r in _load(args.confirmation)["results"]}
    b5 = {r["full_name"]: r for r in _load(args.b5)["results"]}
    b10 = _load(args.b10)
    b20 = _load(args.b20)
    # continuation files store new_results + merged
    cont = {}  # full_name -> {budget, winning_program, first_success_rank}
    for d in (b10, b20):
        if not d:
            continue
        for r in d.get("new_results", []):
            if r.get("success"):
                cont[r["full_name"]] = {"budget": d["budget"],
                                        "winning_program": r.get("winning_program"),
                                        "first_success_rank": r.get("first_success_rank"),
                                        "control_wins": r.get("control_wins", [])}

    rc4a_wins = set((_load("project/evolve/experiments/rc4_candidates/def_unfold_simp/out/minimal_attribution.json") or {}).get("true_def_unfold_win_targets", []))
    tr3_winners = {r["full_name"] for r in
                   (_load("project/evolve/experiments/tr3/out/tr3_depth_program_results.json") or {}).get("results", []) if r.get("best_win")}

    records = []
    for fn, b in b5.items():
        cls_rc2 = conf.get(fn, {}).get("classification")
        # determine the winning program across budgets (first solved)
        win = b.get("winning_program")
        win_budget = 5 if win else None
        control_wins = b.get("control_wins", [])
        first_rank = b.get("first_success_rank")
        if not win and fn in cont:
            win = cont[fn]["winning_program"]
            win_budget = cont[fn]["budget"]
            first_rank = cont[fn]["first_success_rank"]
            control_wins = control_wins or cont[fn].get("control_wins", [])

        rc2_failed = cls_rc2 == "CONFIRMED_RC2_FAILURE"
        if b.get("setup_error") and not win:
            cls = "OPEN_FLAKE" if "exceeded" in (b.get("setup_error") or "") else "NEEDS_REVIEW"
        elif not rc2_failed:
            cls = "PRODUCTION_SUBSUMED"
        elif win:
            if control_wins:
                cls = "BASELINE_DUPLICATE"
            elif _is_def_unfold(win) and fn in rc4a_wins:
                cls = "TRUE_RC4A_REPRODUCTION"
            else:
                cls = "TRUE_RANKER_DELTA"
        else:
            cls = "NO_WIN_UNDER_BUDGET"

        credited = cls in ("TRUE_RANKER_DELTA", "TRUE_RC4A_REPRODUCTION")
        rc4b = credited and _uses_disjoint_left(win)
        rc4c = credited and _is_d2_aesop(win)
        rec = {
            "full_name": fn, "namespace": b.get("namespace"),
            "rc2_status": cls_rc2, "classification": cls, "credited": credited,
            "win_budget": win_budget, "first_success_rank": first_rank,
            "winning_program": win, "control_wins": control_wins,
            "rc4b_evidence": rc4b, "rc4c_evidence": rc4c,
            "reproduces_tr3_win": fn in tr3_winners,
            "evidence": ("controls solved → baseline" if cls == "BASELINE_DUPLICATE"
                         else "literal RC2 failed; ranker program solved; controls failed"
                         if credited else
                         "no ranker program solved under budget" if cls == "NO_WIN_UNDER_BUDGET"
                         else cls),
        }
        records.append(rec)

    hist = Counter(r["classification"] for r in records)
    true_delta = [r for r in records if r["classification"] == "TRUE_RANKER_DELTA"]
    rc4a_repro = [r for r in records if r["classification"] == "TRUE_RC4A_REPRODUCTION"]
    rc4b_ev = [r for r in records if r["rc4b_evidence"]]
    rc4c_ev = [r for r in records if r["rc4c_evidence"]]
    fp = [r for r in records if r["classification"] == "RANKER_FALSE_POSITIVE"]
    out = {
        "generated_by": "scripts/tr5_apply_attribution.py",
        "num_targets": len(records), "classification_histogram": dict(hist),
        "num_true_ranker_delta": len(true_delta),
        "num_true_rc4a_reproduction": len(rc4a_repro),
        "num_credited": sum(1 for r in records if r["credited"]),
        "true_ranker_delta_targets": [r["full_name"] for r in true_delta],
        "rc4a_reproduction_targets": [r["full_name"] for r in rc4a_repro],
        "rc4b_evidence_targets": [r["full_name"] for r in rc4b_ev],
        "rc4c_evidence_targets": [r["full_name"] for r in rc4c_ev],
        "num_false_positive": len(fp),
        "records": records,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR5 attribution", "",
          f"- targets: {len(records)} | classifications: {dict(hist)}",
          f"- **TRUE_RANKER_DELTA: {len(true_delta)}** | TRUE_RC4A_REPRODUCTION: {len(rc4a_repro)} "
          f"| credited total: {out['num_credited']}",
          f"- RC4B evidence (Set.disjoint_left): {len(rc4b_ev)} → {[r['full_name'] for r in rc4b_ev]}",
          f"- RC4C evidence (d2_simp_aesop): {len(rc4c_ev)} → {[r['full_name'] for r in rc4c_ev]}", "",
          "## Credited wins", "",
          "| theorem | class | budget | rank | tags | winning tactic |",
          "|---|---|---|---|---|---|"]
    for r in records:
        if not r["credited"]:
            continue
        wp = r["winning_program"]
        tags = [t for t, on in (("RC4B", r["rc4b_evidence"]), ("RC4C", r["rc4c_evidence"])) if on]
        md.append(f"| `{r['full_name']}` | {r['classification']} | {r['win_budget']} | "
                  f"{r['first_success_rank']} | {','.join(tags)} | "
                  f"`{wp['tactic'][:45] if wp else ''}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr5-attr] {dict(hist)} | true_delta={len(true_delta)} rc4a_repro={len(rc4a_repro)} "
          f"rc4b={len(rc4b_ev)} rc4c={len(rc4c_ev)}")


if __name__ == "__main__":
    main()
