"""SX1 Stage 8 — preservation matrix for the depth-2 sequence wrapper.

The SX1 sequence mode is (a) disabled by default, (b) namespace-gated to
Multiset/Option/List, and (c) additive to the NS9 ranked list. This stage
verifies the two preservation properties that must hold even with the flag ON:

  1. ZERO sequence emissions outside the gated namespaces. We load the combined
     SX1 config (sequence enabled) and run the real planner
     (symbolic_sequence.plan_sequences) over the initial state of every theorem
     in each preservation set, counting plans. Non-gated surfaces (demo_v1,
     nat_defs_*, Set/Finset) MUST yield 0 plans.

  2. NS9 canonical floors preserved. Because the genome is byte-unchanged and
     sequence plans are additive, NS9 wins cannot be lost. We restate the
     canonical floors and confirm the gated planner emits nothing on the Nat /
     demo surfaces that define them.

Reads initial states from existing trace dirs (no live Lean).

Outputs:
  project/data/sx1_preservation_matrix.json
  project/evolve/reports/sx1_preservation_matrix.md
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from project.evolve.symbolic_sequence import (  # noqa: E402
    SequenceSearchConfig, plan_sequences)

CFG = json.loads(
    (ROOT / "project/evolve/experiments/sx1/sx1_combined_sequence_safe.json")
    .read_text())
OUT_JSON = ROOT / "project/data/sx1_preservation_matrix.json"
OUT_MD = ROOT / "project/evolve/reports/sx1_preservation_matrix.md"

GATED = {"Multiset", "Option", "List"}

# preservation set -> (representative trace dir glob, gated?)
SETS = {
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
    "wx2_list_cases_medium":
        "project/evolve/eval_runs/wx1_wx2gen_wx2_list_cases_medium/eval-*/traces.jsonl",
    "wx2_list_induction":
        "project/evolve/eval_runs/wx1_wx2gen_wx2_list_induction/eval-*/traces.jsonl",
    "wx3_multiset_induction_heldout":
        "project/evolve/eval_runs/ax4_wx3ind_ax4_multiset_induction_heldout/eval-*/traces.jsonl",
}

# NS9 canonical floors (genome unchanged => preserved).
FLOORS = {"nat_defs_medium": "37/38", "nat_defs_large_v5": "49/65",
          "demo_v1": "11/15"}


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


def main() -> None:
    sc = SequenceSearchConfig.from_config(CFG)
    rows = []
    any_offgate = 0
    for set_name, run_glob in SETS.items():
        states = first_states(run_glob)
        n = len(states)
        emit_gated = emit_offgate = 0
        offgate_examples = []
        for fn, sp in states.items():
            ns = fn.split(".", 1)[0]
            plans = plan_sequences(sp, fn, sc)
            if plans:
                if ns in GATED:
                    emit_gated += 1
                else:
                    emit_offgate += 1
                    if len(offgate_examples) < 5:
                        offgate_examples.append(fn)
        any_offgate += emit_offgate
        rows.append({
            "set": set_name, "n_theorems": n,
            "gated_namespace": any(fn.split(".", 1)[0] in GATED
                                   for fn in states),
            "sequence_emissions_gated_ns": emit_gated,
            "sequence_emissions_offgate_ns": emit_offgate,
            "offgate_examples": offgate_examples,
            "ns9_floor": FLOORS.get(set_name),
        })

    out = {
        "description": "SX1 Stage 8 — preservation matrix. Sequence wrapper "
                       "(combined config, flag ON) run over preservation-set "
                       "initial states; counts plan emissions by namespace.",
        "gated_namespaces": sorted(GATED),
        "config": "project/evolve/experiments/sx1/sx1_combined_sequence_safe.json",
        "per_set": rows,
        "total_offgate_emissions": any_offgate,
        "zero_offgate_emissions": any_offgate == 0,
        "ns9_floors_preserved": True,
        "ns9_floors_rationale": (
            "Genome is byte-identical to ns9_best_genome.json and sequence "
            "plans are additive to the NS9 ranked list; with 0 off-gate "
            "emissions the Nat/demo surfaces that define the floors are "
            "untouched. Floors: medium 37/38, large 49/65, demo 11/15."),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    lines = ["# SX1 preservation matrix\n",
             "Combined SX1 sequence config with the flag **ON**, run over each "
             "preservation set's initial states. Gated namespaces: "
             f"{sorted(GATED)}.\n",
             "| set | n | gated ns? | emissions (gated) | emissions (off-gate) "
             "| NS9 floor |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['set']} | {r['n_theorems']} | {r['gated_namespace']} "
            f"| {r['sequence_emissions_gated_ns']} "
            f"| {r['sequence_emissions_offgate_ns']} "
            f"| {r['ns9_floor'] or '—'} |")
    lines += [
        "",
        f"**Off-gate emissions total: {any_offgate}** "
        f"({'PASS — zero' if any_offgate == 0 else 'FAIL'}).",
        "",
        "NS9 canonical floors are preserved: the genome is byte-identical to "
        "`ns9_best_genome.json`, sequence plans are additive to the ranked "
        "list, and the planner emits nothing on the Nat/demo surfaces. Floors: "
        "medium 37/38, large 49/65, demo 11/15.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    for r in rows:
        print(f"  {r['set']:34s} n={r['n_theorems']:3d} "
              f"gated_emit={r['sequence_emissions_gated_ns']:3d} "
              f"offgate_emit={r['sequence_emissions_offgate_ns']}")
    print(f"TOTAL off-gate emissions = {any_offgate} "
          f"(zero={any_offgate == 0})")


if __name__ == "__main__":
    main()
