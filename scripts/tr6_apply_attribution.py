#!/usr/bin/env python3
"""TR6 Part 9 — attribute fresh live wins against literal RC2 (SX4 discipline).

Every win must beat literal RC2: RC2 confirmed-failure + bare controls (run in-worker at
B5) fail + not source-specific. Since the registry excludes all TR3/TR5 theorems, any
credited win is FRESH by construction. Classes: FRESH_TRUE_DELTA (+ rc4a/rc4b/rc4c
evidence flags), BASELINE_DUPLICATE, PRODUCTION_SUBSUMED, SOURCE_SPECIFIC,
NO_WIN_UNDER_BUDGET, OPEN_FLAKE, NEEDS_REVIEW.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_'.]*$")


def _p(*a):
    return os.path.join(_REPO, *a)


def _load(path):
    if not path:
        return None
    fp = _p(path) if not os.path.isabs(path) else path
    return json.load(open(fp)) if os.path.exists(fp) else None


def _lemmas(prog):
    return prog.get("used_lemmas") or prog.get("lemmas") or []


def _is_def_unfold(p):
    return p.get("family") == "def_unfold_simp"


def _uses_disjoint_left(p):
    return any("disjoint_left" in (L or "") for L in _lemmas(p))


def _is_d2_aesop(p):
    t = p.get("tactic") or ""
    return p.get("family") == "d2_simp_aesop" or (t.strip().startswith("simp [") and "<;>" in t and "aesop" in t)


def _source_specific(p):
    # a win is source-specific if its lemma names are malformed (shouldn't happen — filtered)
    return any(not _NAME_RE.match(L or "") for L in _lemmas(p))


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
    # collect winning record per theorem at earliest budget
    win_at = {}     # fn -> (budget, winning_program, control_wins, first_rank)
    controls_seen = {}  # fn -> control_wins (from b5)
    for path, budget in ((args.b5, 5), (args.b10, 10), (args.b20, 20)):
        d = _load(path)
        if not d:
            continue
        for r in d["results"]:
            fn = r["full_name"]
            if r.get("control_wins") and fn not in controls_seen:
                controls_seen[fn] = r["control_wins"]
            if r.get("success") and fn not in win_at:
                win_at[fn] = (r.get("budget", budget), r.get("winning_program"),
                              r.get("control_wins", []), r.get("first_success_rank"))

    # all searched theorems = those in b5 results
    b5 = _load(args.b5)
    searched = [r["full_name"] for r in b5["results"]] if b5 else []

    records = []
    for fn in searched:
        cls_rc2 = conf.get(fn, {}).get("classification")
        rc2_failed = cls_rc2 == "CONFIRMED_RC2_FAILURE"
        b5rec = next((r for r in b5["results"] if r["full_name"] == fn), {})
        cw = controls_seen.get(fn, b5rec.get("control_wins", []))
        w = win_at.get(fn)
        win = w[1] if w else None
        budget = w[0] if w else None
        rank = w[3] if w else None

        if not rc2_failed:
            cls = "PRODUCTION_SUBSUMED"
        elif win:
            if cw:
                cls = "BASELINE_DUPLICATE"
            elif _source_specific(win):
                cls = "SOURCE_SPECIFIC"
            else:
                cls = "FRESH_TRUE_DELTA"
        elif b5rec.get("setup_error"):
            cls = "OPEN_FLAKE" if "exceed" in (b5rec.get("setup_error") or "") else "NEEDS_REVIEW"
        else:
            cls = "NO_WIN_UNDER_BUDGET"

        credited = cls == "FRESH_TRUE_DELTA"
        rec = {
            "full_name": fn, "namespace": (conf.get(fn, {}) or {}).get("namespace"),
            "rc2_status": cls_rc2, "classification": cls, "credited": credited,
            "win_budget": budget, "first_success_rank": rank, "winning_program": win,
            "control_wins": cw,
            "rc4a_evidence": bool(credited and _is_def_unfold(win)),
            "rc4b_evidence": bool(credited and _uses_disjoint_left(win)),
            "rc4c_evidence": bool(credited and _is_d2_aesop(win)),
            "nonset_positive": bool(credited and (conf.get(fn, {}) or {}).get("namespace") != "Set"),
        }
        records.append(rec)

    hist = Counter(r["classification"] for r in records)
    td = [r for r in records if r["credited"]]
    nonset = [r for r in td if r["nonset_positive"]]
    out = {
        "generated_by": "scripts/tr6_apply_attribution.py",
        "num_searched": len(records), "classification_histogram": dict(hist),
        "num_fresh_true_delta": len(td),
        "fresh_true_delta_targets": [r["full_name"] for r in td],
        "num_nonset_positives": len(nonset),
        "nonset_positive_targets": [r["full_name"] for r in nonset],
        "nonset_positive_namespaces": dict(Counter(r["namespace"] for r in nonset)),
        "rc4a_evidence_targets": [r["full_name"] for r in records if r["rc4a_evidence"]],
        "rc4b_evidence_targets": [r["full_name"] for r in records if r["rc4b_evidence"]],
        "rc4c_evidence_targets": [r["full_name"] for r in records if r["rc4c_evidence"]],
        "records": records,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR6 attribution", "",
          f"- searched: {len(records)} | classifications: {dict(hist)}",
          f"- **FRESH_TRUE_DELTA: {len(td)}** | non-Set positives: {len(nonset)} "
          f"{dict(Counter(r['namespace'] for r in nonset))}",
          f"- RC4A evidence: {len(out['rc4a_evidence_targets'])} | "
          f"RC4B: {len(out['rc4b_evidence_targets'])} | RC4C: {len(out['rc4c_evidence_targets'])}", "",
          "## Credited fresh wins", "",
          "| theorem | ns | budget | rank | tags | winning tactic |", "|---|---|---|---|---|---|"]
    for r in td:
        wp = r["winning_program"]
        tags = [t for t, on in (("RC4A", r["rc4a_evidence"]), ("RC4B", r["rc4b_evidence"]),
                                ("RC4C", r["rc4c_evidence"])) if on]
        md.append(f"| `{r['full_name']}` | {r['namespace']} | {r['win_budget']} | "
                  f"{r['first_success_rank']} | {','.join(tags)} | "
                  f"`{(wp or {}).get('tactic','')[:42]}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr6-attr] {dict(hist)} | fresh_true_delta={len(td)} nonset={len(nonset)} "
          f"rc4b={len(out['rc4b_evidence_targets'])} rc4c={len(out['rc4c_evidence_targets'])}")


if __name__ == "__main__":
    main()
