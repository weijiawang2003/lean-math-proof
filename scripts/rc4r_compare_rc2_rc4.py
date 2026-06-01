#!/usr/bin/env python3
"""RC4R Part 6 — RC2-vs-RC4 benchmark comparison + delta attribution.

Per theorem classifies RC4_NEW_WIN / RC4_REGRESSION / BOTH_SOLVED / BOTH_FAILED / FLAKE /
PATH_ERROR. Each RC4_NEW_WIN is attributed to the component whose action is RC4's winning
tactic (RC4A / RC4B / RC4C_residue), falling back to the RC4D minimal-attribution component for
known wins, else UNKNOWN_COMPONENT. Rolls up totals, raw/net delta, by set / namespace /
component, and known-vs-fresh.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RC4_ACTIONS = {
    "simp [Finset.disjUnion]": "RC4A", "simp [Monotone, MonotoneOn]": "RC4A",
    "simp [Antitone, AntitoneOn]": "RC4A", "simp [StrictMono, StrictMonoOn]": "RC4A",
    "simp [StrictAnti, StrictAntiOn]": "RC4A",
    "simp [Set.disjoint_left]": "RC4B", "simp [Set.disjoint_left] <;> aesop": "RC4B",
    "simp [Multiset.disjoint_left]": "RC4B", "simp [Multiset.disjoint_left] <;> aesop": "RC4B",
    "simp [Multiset.disjoint_right]": "RC4C_residue",
    "simp [Multiset.disjoint_right] <;> aesop": "RC4C_residue",
    "simp [Set.subset_pair_iff_eq]": "RC4C_residue",
    "simp [Set.subset_pair_iff_eq] <;> aesop": "RC4C_residue",
    "simp [List.forall_iff_forall_mem]": "RC4C_residue",
    "simp [List.forall_iff_forall_mem] <;> aesop": "RC4C_residue",
}


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rc2", required=True)
    ap.add_argument("--rc4", required=True)
    ap.add_argument("--rc4d-attribution",
                    default="project/evolve/experiments/rc4_candidates/composition_rc4d/out/minimal_attribution.json")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rc2 = {r["full_name"]: r for r in json.load(open(_p(args.rc2)))["results"]}
    rc4 = {r["full_name"]: r for r in json.load(open(_p(args.rc4)))["results"]}
    attr_comp = {}
    if os.path.exists(_p(args.rc4d_attribution)):
        for r in json.load(open(_p(args.rc4d_attribution)))["records"]:
            if r.get("credited"):
                attr_comp[r["full_name"]] = r["winning_component"]

    rows = []
    for fn in sorted(set(rc2) | set(rc4)):
        a, b = rc2.get(fn, {}), rc4.get(fn, {})
        sa, sb = a.get("status"), b.get("status")
        sets = b.get("sets") or a.get("sets") or []
        ns = b.get("namespace") or a.get("namespace")
        fresh = "fresh_out_of_sample_frontier" in sets
        if sa in ("path_error",) or sb in ("path_error",):
            cls = "PATH_ERROR"
        elif sa in ("open_flake", "trace_insufficient") or sb in ("open_flake", "trace_insufficient"):
            cls = "FLAKE"
        elif sa == "solved" and sb == "solved":
            cls = "BOTH_SOLVED"
        elif sa != "solved" and sb == "solved":
            cls = "RC4_NEW_WIN"
        elif sa == "solved" and sb != "solved":
            cls = "RC4_REGRESSION"
        else:
            cls = "BOTH_FAILED"
        comp = None
        if cls == "RC4_NEW_WIN":
            comp = RC4_ACTIONS.get(b.get("winning_tactic")) or attr_comp.get(fn) or "UNKNOWN_COMPONENT"
        rows.append({"full_name": fn, "namespace": ns, "sets": sets, "fresh": fresh,
                     "rc2_status": sa, "rc4_status": sb, "classification": cls,
                     "rc4_winning_tactic": b.get("winning_tactic"), "component": comp})

    hist = Counter(r["classification"] for r in rows)
    new_wins = [r for r in rows if r["classification"] == "RC4_NEW_WIN"]
    regr = [r for r in rows if r["classification"] == "RC4_REGRESSION"]
    rc2_solved = sum(1 for r in rows if r["rc2_status"] == "solved")
    rc4_solved = sum(1 for r in rows if r["rc4_status"] == "solved")
    by_comp = Counter(r["component"] for r in new_wins)
    by_ns = Counter(r["namespace"] for r in new_wins)
    by_set = Counter(s for r in new_wins for s in r["sets"])
    known_wins = [r["full_name"] for r in new_wins if not r["fresh"]]
    fresh_wins = [r["full_name"] for r in new_wins if r["fresh"]]

    out = {
        "generated_by": "scripts/rc4r_compare_rc2_rc4.py",
        "classification_histogram": dict(hist),
        "rc2_total_solved": rc2_solved, "rc4_total_solved": rc4_solved,
        "raw_delta": rc4_solved - rc2_solved,
        "new_wins": len(new_wins), "regressions": len(regr),
        "net_delta": len(new_wins) - len(regr),
        "new_wins_by_component": dict(by_comp),
        "new_wins_by_namespace": dict(by_ns),
        "new_wins_by_set": dict(by_set),
        "known_wins": known_wins, "fresh_wins": fresh_wins,
        "num_known_wins": len(known_wins), "num_fresh_wins": len(fresh_wins),
        "regression_targets": [r["full_name"] for r in regr],
        "new_win_targets": [r["full_name"] for r in new_wins],
        "rows": rows,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC4 vs RC2 comparison", "",
          f"- RC2 solved: **{rc2_solved}** | RC4 solved: **{rc4_solved}** | raw delta: **{out['raw_delta']}**",
          f"- new wins: **{len(new_wins)}** | regressions: **{len(regr)}** | net delta: **{out['net_delta']}**",
          f"- new wins by component: {dict(by_comp)}",
          f"- new wins by namespace: {dict(by_ns)}",
          f"- known-win reproductions: {len(known_wins)} | fresh new wins: {len(fresh_wins)} {fresh_wins}",
          f"- regressions: {out['regression_targets']}",
          f"- classification: {dict(hist)}", "",
          "## RC4 new wins", "", "| theorem | ns | set | component | rc4_tactic |", "|---|---|---|---|---|"]
    for r in new_wins:
        md.append(f"| `{r['full_name']}` | {r['namespace']} | {'fresh' if r['fresh'] else 'known'} | "
                  f"{r['component']} | `{r['rc4_winning_tactic'] or ''}` |")
    if regr:
        md += ["", "## Regressions", ""] + [f"- `{r['full_name']}` ({r['namespace']})" for r in regr]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4r-compare] RC2={rc2_solved} RC4={rc4_solved} raw_delta={out['raw_delta']} "
          f"new={len(new_wins)} regr={len(regr)} net={out['net_delta']}")
    print(f"[rc4r-compare] by_comp={dict(by_comp)} known={len(known_wins)} fresh={len(fresh_wins)} {fresh_wins}")


if __name__ == "__main__":
    main()
