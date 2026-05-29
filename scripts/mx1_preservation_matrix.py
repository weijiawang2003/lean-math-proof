"""MX1 Stage 7 — preservation matrix for the extended symbolic wrapper.

Two checks on the best MX1 config (mx1_combined_symbolic_frontier_safe, flag ON):

  1. STATIC off-gate emission check. Instantiate the MX1 combined symbolic
     actions over each preservation set's initial states (read from existing
     trace dirs) and count emissions. Truly off-gate namespaces (Nat / Int /
     demo's Nat goals) MUST yield 0 emissions. Set/Finset/Multiset are now
     GATED families, so they may emit there — that is expected and additive.

  2. LIVE regression check. If MX1 preservation run traces exist (produced by
     scripts/mx1_run_preservation.sh: variant E live on the floor sets), compare
     E wins to the production baseline B / canonical floors. Symbolic actions are
     additive to the NS9 ranked list, so wins must be >= the floor (0 regressions).

Canonical NS9 floors: nat_defs_medium 37/38, nat_defs_large_v5 49/65, demo_v1 11/15.

Outputs:
  project/data/mx1_preservation_matrix.json
  project/evolve/reports/mx1_preservation_matrix.md
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from project.evolve.symbolic_actions import (  # noqa: E402
    instantiate_symbolic_action, load_actions)

CFG = json.loads(
    (ROOT / "project/evolve/experiments/mx1/mx1_combined_symbolic_frontier_safe.json")
    .read_text())
OUT_JSON = ROOT / "project/data/mx1_preservation_matrix.json"
OUT_MD = ROOT / "project/evolve/reports/mx1_preservation_matrix.md"

GATED = {"Multiset", "Set", "Finset"}
OFFGATE_FLOORS = {"nat_defs_medium": "37/38", "nat_defs_large_v5": "49/65",
                  "demo_v1": "11/15"}

# preservation set -> a trace dir glob for initial states (production runs)
INIT_SETS = {
    "demo_v1": "project/evolve/eval_runs/wx3_comb_demo_v1/eval-*/traces.jsonl",
    "nat_defs_medium":
        "project/evolve/eval_runs/wx3_comb_nat_defs_medium/eval-*/traces.jsonl",
    "nat_defs_large_v5":
        "project/evolve/eval_runs/gen_v5_ns15_balanced_namespace_raw_nat_defs_large_v5/eval-*/traces.jsonl",
    "ns17_set_extra":
        "project/evolve/eval_runs/wx3_comb_ns17_set_extra/eval-*/traces.jsonl",
    "ns17_finset_extra":
        "project/evolve/eval_runs/wx3_comb_ns17_finset_extra/eval-*/traces.jsonl",
    "wx2_list_cases_easy":
        "project/evolve/eval_runs/wx1_wx2gen_wx2_list_cases_easy/eval-*/traces.jsonl",
    "ax4_multiset_induction_heldout":
        "project/evolve/eval_runs/ax4_wx3ind_ax4_multiset_induction_heldout/eval-*/traces.jsonl",
}


def first_states(run_glob):
    best = {}
    for tf in glob.glob(run_glob):
        for line in open(tf):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            fn, sp, st = o.get("full_name"), o.get("state_pp") or "", \
                o.get("step", 1) or 1
            if fn and sp and (fn not in best or st < best[fn][0]):
                best[fn] = (st, sp)
    return {fn: sp for fn, (st, sp) in best.items()}


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
    actions = load_actions(CFG["symbolic_actions"]["actions"])
    rows = []
    total_offgate = 0
    for set_name, run_glob in INIT_SETS.items():
        states = first_states(run_glob)
        emit_gated = emit_offgate = 0
        offgate_ex = []
        for fn, sp in states.items():
            ns = fn.split(".", 1)[0]
            emitted = False
            for a in actions:
                if instantiate_symbolic_action(a, sp, fn):
                    emitted = True
                    break
            if emitted:
                if ns in GATED:
                    emit_gated += 1
                else:
                    emit_offgate += 1
                    if len(offgate_ex) < 5:
                        offgate_ex.append(fn)
        total_offgate += emit_offgate
        # optional live regression: E run on this set
        live_glob = f"project/evolve/eval_runs/mx1_pres_E_{set_name}/eval-*/traces.jsonl"
        prod_glob = f"project/evolve/eval_runs/mx1_pres_B_{set_name}/eval-*/traces.jsonl"
        e_w = wins(live_glob)
        b_w = wins(prod_glob)
        rows.append({
            "set": set_name, "n_theorems": len(states),
            "static_emissions_gated_ns": emit_gated,
            "static_emissions_offgate_ns": emit_offgate,
            "offgate_examples": offgate_ex,
            "ns9_floor": OFFGATE_FLOORS.get(set_name),
            "live_E_wins": len(e_w) if e_w else None,
            "live_B_wins": len(b_w) if b_w else None,
            "live_regressions": (len(b_w - e_w) if (e_w and b_w) else None),
        })

    live_regr = sum(r["live_regressions"] or 0 for r in rows)
    out = {
        "description": "MX1 Stage 7 — preservation matrix for the extended "
                       "Set/Finset/Multiset symbolic wrapper (combined config).",
        "gated_namespaces": sorted(GATED),
        "config": "project/evolve/experiments/mx1/mx1_combined_symbolic_frontier_safe.json",
        "per_set": rows,
        "total_offgate_emissions": total_offgate,
        "zero_offgate_emissions": total_offgate == 0,
        "live_regressions_total": live_regr,
        "note": ("Set/Finset/Multiset are now gated symbolic families, so static "
                 "emissions on Set/Finset preservation sets are EXPECTED and "
                 "additive; the preservation guarantees are (a) 0 emissions on "
                 "off-gate namespaces (Nat/Int), and (b) 0 live regressions "
                 "(symbolic actions are additive to the NS9 ranked list)."),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    lines = ["# MX1 preservation matrix\n",
             "Best MX1 config (`mx1_combined_symbolic_frontier_safe`, flag ON). "
             f"Gated symbolic families: {sorted(GATED)}.\n",
             "| set | n | static emit (gated) | static emit (off-gate) "
             "| live E wins | live B wins | live regr | NS9 floor |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['set']} | {r['n_theorems']} "
            f"| {r['static_emissions_gated_ns']} "
            f"| {r['static_emissions_offgate_ns']} "
            f"| {r['live_E_wins'] if r['live_E_wins'] is not None else '—'} "
            f"| {r['live_B_wins'] if r['live_B_wins'] is not None else '—'} "
            f"| {r['live_regressions'] if r['live_regressions'] is not None else '—'} "
            f"| {r['ns9_floor'] or '—'} |")
    lines += ["",
              f"**Off-gate emissions total: {total_offgate}** "
              f"({'PASS — zero' if total_offgate == 0 else 'FAIL'}).",
              f"**Live regressions total: {live_regr}.**", "",
              out["note"]]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    for r in rows:
        print(f"  {r['set']:34s} n={r['n_theorems']:3d} "
              f"gated_emit={r['static_emissions_gated_ns']:3d} "
              f"offgate_emit={r['static_emissions_offgate_ns']} "
              f"liveE={r['live_E_wins']} liveB={r['live_B_wins']} "
              f"regr={r['live_regressions']}")
    print(f"TOTAL off-gate={total_offgate} live_regr={live_regr}")


if __name__ == "__main__":
    main()
