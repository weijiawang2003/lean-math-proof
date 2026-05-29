"""MX2 Stage 4 — aggregate the Set-aesop live eval into probe metadata.

Reads the per-(variant,set) trace dirs from scripts/mx2_run_eval.sh and computes,
per set and variant (A prod / B broad-Set-aesop / C narrow-Set.Finite-aesop):
wins, new-wins-beyond-production, regressions, aesop emissions, aesop closes,
and the theorem names. The key metrics are B/C new wins beyond A and any
regressions or off-target aesop fires on the negative / mixed controls.

Output: project/data/mx2_set_aesop_probe_meta.json
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETS = list(json.loads(
    (ROOT / "project/evolve/routing/mx2_theorem_sets.json").read_text()).keys())
OUT = ROOT / "project/data/mx2_set_aesop_probe_meta.json"


def load(tag, s):
    recs = []
    for tf in glob.glob(f"project/evolve/eval_runs/mx2_{tag}_{s}/eval-*/traces.jsonl"):
        for line in open(tf):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs


def wins(recs):
    return {r["full_name"] for r in recs
            if r.get("proof_finished") and r.get("full_name")}


def aesop_stats(recs):
    """(#aesop emissions, #theorems closed by an aesop tactic)."""
    emit = 0
    closed = set()
    for r in recs:
        t = (r.get("tactic") or "")
        if "aesop" in t:
            emit += 1
            if r.get("proof_finished"):
                closed.add(r.get("full_name"))
    return emit, closed


def seen(recs):
    return {r.get("full_name") for r in recs if r.get("full_name")}


def main() -> None:
    per_set = []
    grand = defaultdict(int)
    new_win_records = []
    for s in SETS:
        A = load("A_prod", s)
        B = load("B_broad", s)
        C = load("C_narrow", s)
        if not (A or B or C):
            per_set.append({"set": s, "status": "no traces"})
            continue
        aw, bw = wins(A), wins(B)
        b_emit, b_closed = aesop_stats(B)
        b_new = sorted(bw - aw)
        b_regr = sorted(aw - bw)
        # Narrow config (C) gates aesop to Set.Finite./Set.toFinset. It is only
        # RUN on sets where that gate can fire; elsewhere it is provably inert
        # (== production), so we treat absent C traces as C == A (not a loss).
        c_ran = bool(C)
        if c_ran:
            cw = wins(C)
            c_emit, c_closed = aesop_stats(C)
            c_new = sorted(cw - aw)
            c_regr = sorted(aw - cw)
        else:
            cw = aw  # gate-inert: identical to production
            c_emit, c_closed = 0, set()
            c_new, c_regr = [], []
        for fn in b_new:
            new_win_records.append({
                "set": s, "theorem": fn, "config": "broad",
                "closed_by_aesop": fn in b_closed})
        row = {
            "set": s, "n": len(seen(A or B or C)),
            "narrow_ran": c_ran,
            "production_wins": len(aw),
            "broad_wins": len(bw), "narrow_wins": len(cw),
            "broad_new_beyond_prod": len(b_new), "broad_new_theorems": b_new,
            "narrow_new_beyond_prod": len(c_new), "narrow_new_theorems": c_new,
            "broad_regressions": len(b_regr), "broad_regr_theorems": b_regr,
            "narrow_regressions": len(c_regr), "narrow_regr_theorems": c_regr,
            "broad_aesop_emissions": b_emit, "broad_aesop_closes": len(b_closed),
            "narrow_aesop_emissions": c_emit,
            "narrow_aesop_closes": len(c_closed),
        }
        per_set.append(row)
        for k in ("production_wins", "broad_wins", "narrow_wins",
                  "broad_new_beyond_prod", "narrow_new_beyond_prod",
                  "broad_regressions", "narrow_regressions",
                  "broad_aesop_emissions", "broad_aesop_closes"):
            grand[k] += row[k]

    out = {
        "description": "MX2 Stage 4 — live Set-aesop fallback probe.",
        "variants": {"A": "production wrapper (no Set aesop)",
                     "B": "broad Set-gated aesop (mx2_set_aesop_safe)",
                     "C": "narrow Set.Finite/toFinset aesop (mx2_set_finite_aesop_safe)"},
        "per_set": per_set,
        "totals": dict(grand),
        "broad_new_beyond_prod_total": grand.get("broad_new_beyond_prod", 0),
        "narrow_new_beyond_prod_total": grand.get("narrow_new_beyond_prod", 0),
        "broad_regressions_total": grand.get("broad_regressions", 0),
        "narrow_regressions_total": grand.get("narrow_regressions", 0),
        "new_win_records": new_win_records,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for r in per_set:
        if r.get("status") == "no traces":
            print(f"  {r['set']:34s} NO TRACES")
            continue
        print(f"  {r['set']:34s} n={r['n']:2d} A={r['production_wins']:2d} "
              f"B={r['broad_wins']:2d}(new {r['broad_new_beyond_prod']},"
              f"regr {r['broad_regressions']}) "
              f"C={r['narrow_wins']:2d}(new {r['narrow_new_beyond_prod']},"
              f"regr {r['narrow_regressions']}) "
              f"Baesop={r['broad_aesop_emissions']}/{r['broad_aesop_closes']}")
    print(f"TOTAL broad_new={out['broad_new_beyond_prod_total']} "
          f"narrow_new={out['narrow_new_beyond_prod_total']} "
          f"broad_regr={out['broad_regressions_total']} "
          f"narrow_regr={out['narrow_regressions_total']}")


if __name__ == "__main__":
    main()
