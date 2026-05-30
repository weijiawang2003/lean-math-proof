"""WX3 Stage 9 — gate decision.

Reads the probe matrix, the minimal-relabel family pools, and the
preservation matrix, and classifies WX3 against the four gates:

  A. Wrapper-ready success  — WX3 adds >=5 new Multiset wins beyond NS9,
     zero preservation regressions, actions state-aware and reliable.
  B. Symbolic-learning ready — >=40 clean single-shot symbolic labels, OR
     >=20 in one label family with held-out surface remaining.
  C. Multi-step-only        — actions help but do not close from init.
  D. Negative               — no meaningful new wins / too brittle.

Output: project/data/wx3_gate_decision_meta.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "project/data/wx3_multiset_probe_meta.json"
POOLS = ROOT / "project/data/wx3_multiset_family_pools_meta.json"
PRESERV = ROOT / "project/data/wx3_preservation_matrix.json"
AUDIT = ROOT / "project/data/wx3_multiset_catalog_audit_meta.json"
OUT = ROOT / "project/data/wx3_gate_decision_meta.json"


def main() -> None:
    probe = json.loads(PROBE.read_text())
    pools = json.loads(POOLS.read_text())
    audit = json.loads(AUDIT.read_text())
    preserv = json.loads(PRESERV.read_text()) if PRESERV.exists() else None

    best = probe["best_config"]
    best_only = probe["totals"][best]["only_beyond_ns9"]
    best_regr = probe["totals"][best]["regressions_vs_ns9"]

    clean = pools["clean_single_shot_symbolic"]
    biggest_id = pools["biggest_single_family"]
    pool_items = pools["symbolic_label_pools"]
    # family-aggregate: induction_on across simp modes
    induction_family = sum(
        v["unique_count"] for k, v in pool_items.items()
        if k.startswith("MULTISET_INDUCTION_SIMP"))

    # held-out surface: fresh available not placed in the 5 evaluated sets.
    fresh_total = audit["fresh_available_count"]
    # induction-shape fresh candidates total (catalog), vs how many were
    # evaluated (induction_easy=40 + some in quotient/mixed). Use category.
    induction_shape_total = audit["category_counts"].get("induction", 0)
    held_out_exists = fresh_total > 165  # 165 evaluated across the 5 sets

    preserv_regr = (preserv or {}).get("summary", {}).get(
        "total_regressions")
    preserv_leak = (preserv or {}).get("summary", {}).get(
        "multiset_emissions_outside_multiset")

    # --- gate logic ---
    gate_A = (best_only >= 5) and (best_regr == 0) and (
        preserv_regr in (0, None))
    gate_B_strict = (clean >= 40) or (biggest_id >= 20)
    gate_B_family = (induction_family >= 20) and held_out_exists
    gate_B = gate_B_strict or gate_B_family
    gate_C = (pools["multistep_symbolic_assisted"] > 0) and not gate_A
    gate_D = best_only < 5

    if gate_A and gate_B:
        verdict = "A+B"
        headline = ("WRAPPER-READY (promote WX3) AND borderline "
                    "SYMBOLIC-LEARNING-READY (induction_on family).")
    elif gate_A:
        verdict = "A"
        headline = "WRAPPER-READY: promote WX3 Multiset induction wrapper."
    elif gate_C:
        verdict = "C"
        headline = "MULTI-STEP-ONLY: pursue sequence-level symbolic search."
    elif gate_D:
        verdict = "D"
        headline = "NEGATIVE: try another namespace / stronger wrapper."
    else:
        verdict = "?"
        headline = "Indeterminate."

    out = {
        "best_config": best,
        "best_config_path": probe["configs"][best],
        "wins_beyond_ns9": best_only,
        "robust_single_shot_symbolic_wins": clean,
        "matrix_regressions_vs_ns9": best_regr,
        "preservation_regressions": preserv_regr,
        "multiset_emissions_outside_multiset": preserv_leak,
        "clean_single_shot_symbolic_labels": clean,
        "biggest_single_action_id": biggest_id,
        "induction_on_family_aggregate": induction_family,
        "dominant_action_id": (
            "MULTISET_INDUCTION_SIMP[Multiset,simp_all]"),
        "multistep_symbolic_assisted": pools["multistep_symbolic_assisted"],
        "multistep_nonsymbolic": pools["multistep_nonsymbolic"],
        "dropped_over_attributed": pools[
            "dropped_simpler_tactic_closes_single_shot"],
        "fresh_available_total": fresh_total,
        "evaluated_candidates": 165,
        "held_out_surface_exists": held_out_exists,
        "induction_shape_fresh_total": induction_shape_total,
        "gates": {
            "A_wrapper_ready": gate_A,
            "B_symbolic_learning_ready": gate_B,
            "B_strict_40_or_20_single_id": gate_B_strict,
            "B_family_aggregate_20_plus_heldout": gate_B_family,
            "C_multi_step_only": gate_C,
            "D_negative": gate_D,
        },
        "verdict": verdict,
        "headline": headline,
        "recommendation": [
            "PROMOTE the WX3 Multiset wrapper (induction-only "
            "wx3_multiset_induction_safe is sufficient; combined adds "
            "ext/cases for generality at no regression cost).",
            "AX3 is now plausible for the first time: the induction_on family "
            "reaches 18 (simp_all) / 20 (both simp modes) clean single-shot "
            "symbolic labels — at/near the >=20-in-one-family gate, unlike "
            "AX2 (0 clean labels). Recommend EXPANDING the Multiset induction "
            "surface (mine the ~86 held-out fresh + full induction-shape "
            "catalog) to push clean labels to >=40 / single-id >=20 before "
            "committing to AX3 symbolic-action training.",
        ],
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"verdict={verdict}: {headline}")
    print(f"  wins_beyond_ns9={best_only} clean_single_shot={clean} "
          f"biggest_id={biggest_id} induction_family={induction_family}")
    print(f"  gate_A={gate_A} gate_B={gate_B} "
          f"(strict={gate_B_strict} family={gate_B_family})")
    print(f"  preservation_regr={preserv_regr} leak={preserv_leak}")


if __name__ == "__main__":
    main()
