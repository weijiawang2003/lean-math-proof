"""MX1 Stage 3 — aggregate live mining traces into the probe metadata.

Reads the per-(variant,set) trace dirs produced by scripts/mx1_run_live_mining.sh
and computes, per theorem set and variant, the wins / new-wins-beyond-production
/ symbolic-emission stats. Variant C (AX4 predictor) is computed OFFLINE from
B's Multiset traces: the predictor only suppresses NULL-scored Multiset
emissions, so its wins ⊆ B's symbolic wins gated by the classifier firing.

Wrappers:
  A raw   : mx1_A_raw_<set>      (NS24 routed-raw baseline)
  B prod  : mx1_B_prod_<set>     (WX3 production wrapper)
  E sym   : mx1_E_sym_<set>      (MX1 extended Set/Finset/Multiset symbolic)
  D seq   : mx1_D_seq_<set>      (SX1 depth-2 sequence trace generator)

Key metric: new_wins_beyond_production = E wins − B wins (per set, and the
theorem names). Also reports symbolic firings, symbolic-origin wins, regressions
(B win lost under E — impossible since E ⊇ B actions, verified), and the
per-theorem winning symbolic action for relabeling.

Output: project/data/mx1_live_mining_probe_meta.json
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETS = list(json.loads(
    (ROOT / "project/evolve/routing/mx1_theorem_sets.json").read_text()).keys())
OUT = ROOT / "project/data/mx1_live_mining_probe_meta.json"
SYM = "wrapper_symbolic_action"


def run_glob(tag, s):
    return f"project/evolve/eval_runs/mx1_{tag}_{s}/eval-*/traces.jsonl"


def load(tag, s):
    recs = []
    for tf in glob.glob(run_glob(tag, s)):
        for line in open(tf):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs


def wins(recs, symbolic_only=False):
    w = {}
    for r in recs:
        fn = r.get("full_name")
        if not fn or not r.get("proof_finished"):
            continue
        if symbolic_only and r.get("tactic_origin") != SYM:
            continue
        w[fn] = r  # keep a winning record (last wins)
    return w


def seen_theorems(recs):
    return {r.get("full_name") for r in recs if r.get("full_name")}


def winning_symbolic_action(recs, fn):
    """The symbolic tactic/action that closed fn, if any."""
    for r in recs:
        if r.get("full_name") == fn and r.get("proof_finished") \
                and r.get("tactic_origin") == SYM:
            return r.get("tactic"), r.get("tactic_template_source") or \
                r.get("tactic_family_source")
    return None, None


def main() -> None:
    per_set = []
    grand = defaultdict(int)
    new_win_records = []
    for s in SETS:
        A = load("A_raw", s)
        B = load("B_prod", s)
        E = load("E_sym", s)
        D = load("D_seq", s)
        if not (A or B or E):
            per_set.append({"set": s, "status": "no traces"})
            continue
        n = len(seen_theorems(E or B or A))
        aw = set(wins(A))
        bw = set(wins(B))
        ew = set(wins(E))
        e_sym = set(wins(E, symbolic_only=True))
        new_beyond_prod = sorted(ew - bw)
        regressions = sorted(bw - ew)  # production win lost under E
        # symbolic firings on E
        e_fire = sum(1 for r in E if r.get("tactic_origin") == SYM)
        # per-theorem new-win symbolic attribution
        for fn in new_beyond_prod:
            tac, fam = winning_symbolic_action(E, fn)
            new_win_records.append({
                "set": s, "theorem": fn, "namespace": fn.split(".", 1)[0],
                "won_by_raw": fn in aw, "won_by_production": fn in bw,
                "winning_symbolic_tactic": tac,
                "winning_symbolic_family": fam,
                "symbolic_win": fn in e_sym,
            })
        row = {
            "set": s, "n_theorems_seen": n,
            "raw_wins": len(aw), "production_wins": len(bw),
            "extended_symbolic_wins": len(ew),
            "extended_symbolic_origin_wins": len(e_sym),
            "new_wins_beyond_production": len(new_beyond_prod),
            "new_win_theorems": new_beyond_prod,
            "regressions_vs_production": len(regressions),
            "regression_theorems": regressions,
            "symbolic_firings_E": e_fire,
            "sequence_traces_D": sum(1 for r in D
                                     if r.get("tactic_origin") == SYM),
        }
        per_set.append(row)
        for k in ("raw_wins", "production_wins", "extended_symbolic_wins",
                  "extended_symbolic_origin_wins", "new_wins_beyond_production",
                  "regressions_vs_production", "symbolic_firings_E"):
            grand[k] += row[k]

    out = {
        "description": "MX1 Stage 3 — LIVE LeanDojo mining probe over fresh "
                       "symbolic frontier (Set/Finset/Multiset/List).",
        "live": True,
        "variants": {
            "A": "NS24 routed-raw baseline",
            "B": "WX3 production wrapper (Multiset symbolic + NS9)",
            "C": "AX4 predictor (offline from B Multiset traces; see relabel)",
            "D": "SX1 depth-2 sequence trace generator (Multiset/List)",
            "E": "MX1 extended symbolic wrapper (B + Set/Finset ext/cases)",
        },
        "per_set": per_set,
        "totals": dict(grand),
        "new_wins_beyond_production_total":
            grand.get("new_wins_beyond_production", 0),
        "regressions_total": grand.get("regressions_vs_production", 0),
        "new_win_records": new_win_records,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for r in per_set:
        if r.get("status") == "no traces":
            print(f"  {r['set']:34s} NO TRACES")
            continue
        print(f"  {r['set']:34s} n={r['n_theorems_seen']:3d} "
              f"A={r['raw_wins']:3d} B={r['production_wins']:3d} "
              f"E={r['extended_symbolic_wins']:3d} "
              f"new={r['new_wins_beyond_production']:2d} "
              f"sym_fire={r['symbolic_firings_E']:3d} "
              f"regr={r['regressions_vs_production']}")
    print(f"TOTAL new_wins_beyond_production="
          f"{out['new_wins_beyond_production_total']} "
          f"regressions={out['regressions_total']}")


if __name__ == "__main__":
    main()
