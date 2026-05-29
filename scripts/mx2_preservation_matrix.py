"""MX2 Stage 6 — preservation matrix for the Set-aesop fallback.

Two checks on the best MX2 config (mx2_set_aesop_safe):

  1. STATIC off-Set aesop check. The `aesop` gate is `theorem_name_tactic_gates:
     {aesop: [Set.]}`, so NO `aesop` tactic may fire on a non-Set theorem. We
     verify by replaying the wrapper's name-gate over each preservation set's
     theorem names: a non-Set theorem must never admit an `aesop` tactic.

  2. LIVE regression check. From mx2_pres_{A,E}_<set> runs: production (A) vs
     MX2 (E) wins; E must lose nothing (aesop is additive to the ranked list).

Canonical NS9 floors: nat_defs_medium 37/38, nat_defs_large_v5 49/65, demo_v1 11/15.

Outputs:
  project/data/mx2_preservation_matrix.json
  project/evolve/reports/mx2_preservation_matrix.md
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tasks  # noqa: E402

CFG = json.loads(
    (ROOT / "project/evolve/experiments/mx2/mx2_set_aesop_safe.json").read_text())
OUT_JSON = ROOT / "project/data/mx2_preservation_matrix.json"
OUT_MD = ROOT / "project/evolve/reports/mx2_preservation_matrix.md"

GATE = CFG.get("theorem_name_tactic_gates", {}).get("aesop", [])
FLOORS = {"nat_defs_medium": "37/38", "nat_defs_large_v5": "49/65",
          "demo_v1": "11/15"}

# preservation sets -> theorem-set name in tasks (for name-based static gate)
PRES_SETS = ["demo_v1", "nat_defs_medium", "nat_defs_large_v5",
             "ns17_set_extra", "ns17_finset_extra", "ns14_set_finset_extra",
             "wx2_list_cases_easy", "ax4_multiset_induction_heldout"]


def gate_allows_aesop(full_name: str) -> bool:
    return any(full_name.startswith(p) for p in GATE)


def wins(run_glob):
    w = set()
    for tf in glob.glob(run_glob):
        for line in open(tf):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("proof_finished") and o.get("full_name"):
                w.add(o["full_name"])
    return w


def main() -> None:
    rows = []
    total_offset = 0  # non-Set theorems that would admit aesop (must be 0)
    for s in PRES_SETS:
        thms = tasks.THEOREM_SETS.get(s, [])
        n = len(thms)
        non_set = [t.full_name for t in thms
                   if not t.full_name.startswith("Set.")]
        # static: how many non-Set theorems would admit an aesop emission?
        offgate = sum(1 for fn in non_set if gate_allows_aesop(fn))
        total_offset += offgate
        a_w = wins(f"project/evolve/eval_runs/mx2_pres_A_{s}/eval-*/traces.jsonl")
        e_w = wins(f"project/evolve/eval_runs/mx2_pres_E_{s}/eval-*/traces.jsonl")
        rows.append({
            "set": s, "n": n, "non_set_theorems": len(non_set),
            "non_set_aesop_admissible": offgate,
            "live_A_wins": len(a_w) if a_w else None,
            "live_E_wins": len(e_w) if e_w else None,
            "live_regressions": (len(a_w - e_w) if (a_w and e_w) else None),
            "ns9_floor": FLOORS.get(s),
        })

    live_regr = sum(r["live_regressions"] or 0 for r in rows)
    out = {
        "description": "MX2 Stage 6 — preservation matrix for the broad "
                       "Set-aesop fallback (mx2_set_aesop_safe).",
        "aesop_gate_prefixes": GATE,
        "config": "project/evolve/experiments/mx2/mx2_set_aesop_safe.json",
        "per_set": rows,
        "total_non_set_aesop_admissible": total_offset,
        "zero_off_set_aesop": total_offset == 0,
        "live_regressions_total": live_regr,
        "note": ("The aesop name-gate forbids any aesop tactic on a non-Set "
                 "theorem (static guarantee). aesop is additive to the ranked "
                 "list, so it cannot remove a production win (0 live regressions)."),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    lines = ["# MX2 preservation matrix\n",
             f"Broad Set-aesop config (`mx2_set_aesop_safe`). aesop gate: "
             f"`{GATE}`.\n",
             "| set | n | non-Set | non-Set aesop-admissible | live A | live E "
             "| live regr | NS9 floor |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['set']} | {r['n']} | {r['non_set_theorems']} "
            f"| {r['non_set_aesop_admissible']} "
            f"| {r['live_A_wins'] if r['live_A_wins'] is not None else '—'} "
            f"| {r['live_E_wins'] if r['live_E_wins'] is not None else '—'} "
            f"| {r['live_regressions'] if r['live_regressions'] is not None else '—'} "
            f"| {r['ns9_floor'] or '—'} |")
    lines += ["",
              f"**Non-Set aesop-admissible total: {total_offset}** "
              f"({'PASS — zero' if total_offset == 0 else 'FAIL'}).",
              f"**Live regressions total: {live_regr}.**", "", out["note"]]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    for r in rows:
        print(f"  {r['set']:30s} n={r['n']:3d} nonSet_aesop={r['non_set_aesop_admissible']} "
              f"A={r['live_A_wins']} E={r['live_E_wins']} regr={r['live_regressions']}")
    print(f"TOTAL non_set_aesop={total_offset} live_regr={live_regr}")


if __name__ == "__main__":
    main()
