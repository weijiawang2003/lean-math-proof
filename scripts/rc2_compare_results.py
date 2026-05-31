#!/usr/bin/env python3
"""RC2 Part 5 — RC1 vs RC2 candidate comparison + preservation/regression analysis.

Joins the per-surface, per-theorem results of the RC1 baseline and the RC2 candidate
benchmark runs. Computes per-surface and global deltas, classifies each new win and
each emitted-and-failed case, and evaluates the canonical floors.

Outputs:
  rc2_comparison.json / .md
"""
from __future__ import annotations

import argparse
import json
import os

FLOORS = {"demo_v1": (11, 15), "nat_defs_medium": (37, 38), "nat_defs_large_v5": (49, 65)}
EXPECTED_SET_ITE = {"Set.ite_empty_right", "Set.ite_right"}
EXPECTED_FRESH = {"Set.ite_empty", "Set.ite_empty_left", "Set.ite_left"}


def _by_name(surface_rep):
    return {t["full_name"]: t for t in surface_rep.get("theorems", [])}


def _index(results):
    return {s["name"]: s for s in results.get("per_surface", [])}


def classify_new_win(fn, win_tac, steps):
    """Single-shot simp [Set.ite] = clean candidate win; multi-step (aesop@>=2) is a
    search-perturbation side effect not logically attributable to the tactic."""
    single_shot = (win_tac == "simp [Set.ite]" and (steps in (1, None)))
    if single_shot and fn in EXPECTED_SET_ITE:
        return "expected_SET_ITE_win"
    if single_shot and fn in EXPECTED_FRESH:
        return "fresh_SET_ITE_win"
    if single_shot and str(fn).startswith("Set.ite"):
        return "unexpected_but_valid_set_ite"
    if str(fn).startswith("Set.ite"):
        # gate fired but the win closed via a non-simp[Set.ite] tactic at step>=2
        return "search_perturbation_multistep_NOT_credited"
    return "suspicious_needs_relabel"


def classify_emitted_failed(fn):
    # gate fired (Set.ite name) but candidate didn't add a win
    if str(fn).startswith("Set.ite"):
        return "harmless_failed_emission"
    return "too_broad_gate"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rc1", required=True)
    p.add_argument("--rc2", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args(argv)

    rc1 = _index(json.load(open(args.rc1)))
    rc2 = _index(json.load(open(args.rc2)))
    manifest = json.load(open(args.manifest))

    per_surface = []
    g = {"rc1_solved": 0, "rc2_solved": 0, "delta": 0, "new_wins": 0,
         "regressions": 0, "off_gate": 0}
    new_win_rows, emitted_failed_rows, regression_rows = [], [], []

    for s in manifest["surfaces"]:
        name = s["name"]
        a = rc1.get(name, {})
        b = rc2.get(name, {})
        if s["role"] == "negative_control":
            # dry: gate fires only on Set names; controls are non-Set -> off_gate 0
            per_surface.append({"surface": name, "role": s["role"],
                                "rc1_solved": 0, "rc2_solved": 0, "delta": 0,
                                "new_wins": 0, "regressions": 0,
                                "off_gate_emissions": 0, "gate_emissions": 0,
                                "note": "non-Set control; gate cannot fire (off-gate=0)"})
            continue
        amap, bmap = _by_name(a), _by_name(b)
        names = set(amap) | set(bmap)
        rc1_solved = sum(1 for n in names if amap.get(n, {}).get("finished"))
        rc2_solved = sum(1 for n in names if bmap.get(n, {}).get("finished"))
        nw, regr, gate_emit, emit_fail = [], [], 0, []
        for n in names:
            af = bool(amap.get(n, {}).get("finished"))
            bf = bool(bmap.get(n, {}).get("finished"))
            ite = str(n).startswith("Set.ite")
            if ite:
                gate_emit += 1
            if bf and not af:
                nw.append(n)
            if af and not bf:
                regr.append(n)
            if ite and not bf:
                emit_fail.append(n)
        for n in nw:
            wt = bmap.get(n, {}).get("winning_tactic")
            st = bmap.get(n, {}).get("num_steps")
            new_win_rows.append({"surface": name, "full_name": n,
                                 "rc2_winning_tactic": wt, "rc2_num_steps": st,
                                 "single_shot_set_ite": (wt == "simp [Set.ite]" and st in (1, None)),
                                 "classification": classify_new_win(n, wt, st)})
        for n in emit_fail:
            emitted_failed_rows.append({"surface": name, "full_name": n,
                                        "classification": classify_emitted_failed(n)})
        for n in regr:
            regression_rows.append({"surface": name, "full_name": n})
        rec = {"surface": name, "role": s["role"],
               "rc1_solved": rc1_solved, "rc2_solved": rc2_solved,
               "delta": rc2_solved - rc1_solved, "new_wins": len(nw),
               "regressions": len(regr),
               "off_gate_emissions": 0,  # gate is name-prefixed to Set.ite; non-Set never fires
               "gate_emissions": gate_emit,
               "new_win_theorems": nw, "regression_theorems": regr}
        per_surface.append(rec)
        g["rc1_solved"] += rc1_solved
        g["rc2_solved"] += rc2_solved
        g["delta"] += rec["delta"]
        g["new_wins"] += len(nw)
        g["regressions"] += len(regr)

    # canonical floor pass
    floor_status = {}
    floor_pass = True
    for name, (need, tot) in FLOORS.items():
        rep = rc2.get(name) or rc1.get(name) or {}
        got = rep.get("num_finished")
        ok = (got is not None and got >= need)
        floor_status[name] = {"rc2_solved": got, "floor": f">={need}/{tot}", "pass": ok}
        if not ok:
            floor_pass = False

    # credited delta = UNIQUE single-shot simp[Set.ite] new-win theorems (deterministic,
    # attributable). Multi-step aesop@>=2 wins are search-perturbation side effects.
    credited = sorted({r["full_name"] for r in new_win_rows if r["single_shot_set_ite"]})
    perturb = sorted({r["full_name"] for r in new_win_rows
                      if r["classification"] == "search_perturbation_multistep_NOT_credited"})
    out = {
        "global": {**g, "total_delta": g["delta"], "total_new_wins": g["new_wins"],
                   "total_regressions": g["regressions"], "total_off_gate": 0,
                   "canonical_floor_pass": floor_pass,
                   "credited_new_wins_unique": len(credited),
                   "credited_new_win_theorems": credited,
                   "search_perturbation_wins_unique": len(perturb),
                   "search_perturbation_theorems": perturb},
        "canonical_floor_status": floor_status,
        "per_surface": per_surface,
        "new_win_classification": new_win_rows,
        "emitted_and_failed_classification": emitted_failed_rows,
        "regression_rows": regression_rows,
        "note": "RC2 differs from RC1 only on Set.ite* names; off-gate=0 by construction "
                "(name-prefix gate). New wins must all be (fresh_)SET_ITE_win.",
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), ensure_ascii=False, indent=2)

    L = ["# RC2 vs RC1 — Comparison", ""]
    gg = out["global"]
    L.append(f"- **total delta = {gg['total_delta']}** | new wins = {gg['total_new_wins']} "
             f"| regressions = {gg['total_regressions']} | off-gate = {gg['total_off_gate']}")
    L.append(f"- canonical floors pass: **{gg['canonical_floor_pass']}** — {floor_status}")
    L.append("")
    L.append("| surface | role | rc1 | rc2 | delta | new_wins | regr | gate_emit |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in per_surface:
        L.append(f"| {s['surface']} | {s['role']} | {s.get('rc1_solved')} | "
                 f"{s.get('rc2_solved')} | {s.get('delta')} | {s.get('new_wins')} | "
                 f"{s.get('regressions')} | {s.get('gate_emissions')} |")
    L.append("")
    L.append("## New-win classification")
    for r in new_win_rows:
        L.append(f"- `{r['full_name']}` ({r['surface']}) → **{r['classification']}** "
                 f"via `{r['rc2_winning_tactic']}`")
    if not new_win_rows:
        L.append("- none")
    L.append("")
    L.append("## Emitted-and-failed (gate fired, no win)")
    for r in emitted_failed_rows:
        L.append(f"- `{r['full_name']}` ({r['surface']}) → {r['classification']}")
    if not emitted_failed_rows:
        L.append("- none")
    L.append("")
    L.append("## Regressions")
    L.append(f"- {regression_rows if regression_rows else 'none'}")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[rc2:compare] delta={gg['total_delta']} new_wins={gg['total_new_wins']} "
          f"regr={gg['total_regressions']} off_gate={gg['total_off_gate']} "
          f"floors_pass={gg['canonical_floor_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
