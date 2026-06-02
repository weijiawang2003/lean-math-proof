#!/usr/bin/env python3
"""RC4B/RC4C joint comparison (the standalone Part 16 artifact).

Reads ONLY the already-committed RC4B (disjoint_left_bridge) and RC4C (d2_simp_aesop)
validation outputs and produces a side-by-side comparison + composition-readiness decision.

This script does NOT run any live Lean / LeanDojo evaluation and does NOT modify any
candidate artifact, wrapper, or report. It is a pure read-and-summarize step over the
committed validation record:
  - minimal_attribution.json   (TRUE_DELTA classification)
  - candidate_results.json     (single-shot probe metrics over literal RC2)
  - offgate_preservation.json  (off-gate / regression safety)
  - determinism_check.json     (deterministic repeat)
  - schema_wrapper_smoke.json  (production-search wrapper smoke)
  - <rc4b|rc4c>_evidence_summary.json (TR5/TR6 evidence, fresh vs reproduction split)
  - rc4<b|c>_candidate_wrapper.json   (added gated actions = wrapper simplicity)

Decision options: BOTH_READY_FOR_COMPOSITION / RC4B_ONLY_READY / RC4C_ONLY_READY /
BOTH_PROMISING_NEED_REFINEMENT / BOTH_REJECTED. Composition itself is NOT performed here.
"""
from __future__ import annotations

import argparse
import json
import os

# gates present in the literal RC2 release wrapper — adding these does NOT count as
# an RC4-candidate action (they are pre-existing). Everything else in the candidate
# wrapper's theorem_name_tactic_gates is an action the candidate introduced.
RC2_BASELINE_GATES = {"aesop", "simp [Set.ite]"}


def _load(path):
    with open(path) as f:
        return json.load(f)


def _evidence_count(v):
    """evidence_summary fresh_wins/reproduction_wins are either a list or {'count': n}."""
    if isinstance(v, dict):
        return v.get("count", 0)
    if isinstance(v, list):
        return len(v)
    return v or 0


def _added_gate_actions(wrapper):
    gates = wrapper.get("theorem_name_tactic_gates", {}) or {}
    return [g for g in gates if g not in RC2_BASELINE_GATES]


def _extract(root, prefix):
    """prefix is 'rc4b' or 'rc4c' — locate the per-candidate file names."""
    out = os.path.join(root, "out")
    attr = _load(os.path.join(out, "minimal_attribution.json"))
    cand = _load(os.path.join(out, "candidate_results.json"))
    offg = _load(os.path.join(out, "offgate_preservation.json"))
    det = _load(os.path.join(out, "determinism_check.json"))
    smoke = _load(os.path.join(out, "schema_wrapper_smoke.json"))
    ev = _load(os.path.join(out, f"{prefix}_evidence_summary.json"))
    wrapper = _load(os.path.join(root, f"{prefix}_candidate_wrapper.json"))

    m = cand.get("metrics", {})
    # TRUE_DELTA: RC4B has a single class; RC4C splits pure / overlap-with-RC4B.
    hist = attr.get("classification_histogram", {})
    if prefix == "rc4b":
        true_delta = attr.get("num_true_bridge_wins", 0)
        true_delta_pure = true_delta
        overlap_rc4b = 0
        composition_credited = true_delta
        ns_new = m.get("new_wins_by_namespace", {})
        ns_new_residue = ns_new
    else:
        true_delta_pure = attr.get("num_pure_rc4c_true_wins", 0)
        overlap_rc4b = attr.get("num_overlap_rc4b_true_wins", 0)
        composition_credited = attr.get("num_composition_credited",
                                        true_delta_pure + overlap_rc4b)
        true_delta = true_delta_pure + overlap_rc4b
        ns_new = m.get("new_wins_by_namespace_all", {})
        ns_new_residue = m.get("new_wins_by_namespace_nonoverlap", {})

    num_known = ev.get("num_known_wins", 0)
    fresh = _evidence_count(ev.get("fresh_wins"))
    repro = _evidence_count(ev.get("reproduction_wins"))

    off_gate = m.get("off_gate_emissions", offg.get("off_gate_emissions",
                     offg.get("off_gate_emissions_all", 0)))
    off_gate_nonoverlap = offg.get("off_gate_emissions_nonoverlap", off_gate)
    regressions = m.get("regressions", offg.get("regressions", 0))

    added_actions = _added_gate_actions(wrapper)

    rec = {
        "candidate": prefix.upper(),
        "root": root,
        "classification_histogram": hist,
        "true_delta_total": true_delta,
        "true_delta_pure": true_delta_pure,
        "true_delta_overlap_rc4b": overlap_rc4b,
        "composition_credited": composition_credited,
        "mechanism": ev.get("mechanism"),
        "num_known_wins": num_known,
        "fresh_holdout_wins": fresh,
        "reproduction_wins": repro,
        "fresh_holdout_rate": round(fresh / num_known, 3) if num_known else None,
        "candidate_probe": {
            "num_theorems": m.get("num_theorems"),
            "rc2_solved": m.get("rc2_solved"),
            "raw_delta": m.get("raw_delta", m.get("raw_delta_all")),
            "raw_delta_nonoverlap": m.get("raw_delta_nonoverlap"),
            "gate_emissions": m.get("gate_emissions"),
            "emitted_and_solved": m.get("emitted_and_solved"),
            "emitted_and_failed": m.get("emitted_and_failed"),
        },
        "namespace_coverage_all": ns_new,
        "namespace_coverage_residue": ns_new_residue,
        "off_gate_emissions": off_gate,
        "off_gate_emissions_nonoverlap": off_gate_nonoverlap,
        "regressions": regressions,
        "offgate_verdict": offg.get("verdict"),
        "deterministic": bool(det.get("deterministic")),
        "determinism_hash": det.get("clean_run1_hash"),
        "determinism_hash_match": det.get("clean_run1_hash") == det.get("clean_run2_hash"),
        "schema_smoke": {
            "known_wins_total": smoke.get("known_wins_total"),
            "known_wins_solved_by_wrapper": smoke.get("known_wins_solved_by_wrapper"),
            "no_regression": bool(smoke.get("no_regression")),
        },
        "wrapper_added_gated_actions": added_actions,
        "wrapper_added_action_count": len(added_actions),
    }
    # production-search wrapper smoke reproduces the wins under the deployed form?
    rec["schema_smoke"]["reproduces_wins"] = bool(
        (smoke.get("known_wins_solved_by_wrapper") or 0) > 0)
    return rec


def _readiness(rec):
    """READY criteria (Parts 8 & 15): >=1 TRUE_DELTA, evidence replays under the single-shot
    probe, 0 regressions, 0 off-gate, deterministic, wrapper-smoke shows no regression."""
    checks = {
        "has_true_delta": rec["true_delta_total"] >= 1,
        "evidence_replays_probe": (rec["candidate_probe"]["emitted_and_solved"] or 0) >= 1,
        "zero_regressions": rec["regressions"] == 0,
        "zero_offgate": rec["off_gate_emissions"] == 0,
        "deterministic": rec["deterministic"] and rec["determinism_hash_match"],
        "wrapper_smoke_no_regression": rec["schema_smoke"]["no_regression"],
    }
    ready = all(checks.values())
    caveats = []
    if not rec["schema_smoke"]["reproduces_wins"]:
        caveats.append(
            "production-search wrapper smoke reproduces 0 known wins under the deployed "
            "(fused `simp [L] <;> aesop`) form — deploy as the bare `simp [L]` enabling "
            "action (RC4B-style) so the search's own aesop closes; documented in RC4D.")
    return checks, ready, caveats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rc4b-root", required=True)
    ap.add_argument("--rc4c-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rc4b = _extract(args.rc4b_root, "rc4b")
    rc4c = _extract(args.rc4c_root, "rc4c")
    b_checks, b_ready, b_caveats = _readiness(rc4b)
    c_checks, c_ready, c_caveats = _readiness(rc4c)

    if b_ready and c_ready:
        decision = "BOTH_READY_FOR_COMPOSITION"
    elif b_ready and not c_ready:
        decision = "RC4B_ONLY_READY"
    elif c_ready and not b_ready:
        decision = "RC4C_ONLY_READY"
    elif rc4b["true_delta_total"] >= 1 and rc4c["true_delta_total"] >= 1:
        decision = "BOTH_PROMISING_NEED_REFINEMENT"
    else:
        decision = "BOTH_REJECTED"

    # RC4C net contribution beyond RC4B (the part composition actually adds on top of RC4B).
    rc4c_residue_actions = [a for a in rc4c["wrapper_added_gated_actions"]
                            if a not in set(rc4b["wrapper_added_gated_actions"])]
    overlap_note = (
        f"RC4C overlaps RC4B on {rc4c['true_delta_overlap_rc4b']} wins "
        f"(`disjoint_left` is literally an RC4B action); RC4C's net contribution beyond RC4B "
        f"is {rc4c['true_delta_pure']} pure depth-2 wins via the residue actions "
        f"{rc4c_residue_actions}.")

    out = {
        "generated_by": "scripts/rc4bc_compare_candidates.py",
        "note": ("Pure read-only comparison over the committed RC4B/RC4C validation record "
                 "(commit 8f9d08e). No live evaluation, no artifact mutation. Composition "
                 "is a separate later task and is NOT performed here."),
        "rc4b": rc4b,
        "rc4c": rc4c,
        "readiness": {
            "rc4b": {"checks": b_checks, "ready": b_ready, "caveats": b_caveats,
                     "committed_decision": "RC4B_CANDIDATE_CONFIRMED"},
            "rc4c": {"checks": c_checks, "ready": c_ready, "caveats": c_caveats,
                     "committed_decision": "RC4C_CONFIRMED_WITH_RC4B_OVERLAP"},
        },
        "comparison": {
            "true_delta": {"RC4B": rc4b["true_delta_total"],
                           "RC4C_total": rc4c["true_delta_total"],
                           "RC4C_pure_nonoverlap": rc4c["true_delta_pure"],
                           "RC4C_overlap_rc4b": rc4c["true_delta_overlap_rc4b"]},
            "fresh_holdout_rate": {"RC4B": rc4b["fresh_holdout_rate"],
                                   "RC4C": rc4c["fresh_holdout_rate"]},
            "fresh_holdout_wins": {"RC4B": rc4b["fresh_holdout_wins"],
                                   "RC4C": rc4c["fresh_holdout_wins"]},
            "offgate_risk": {"RC4B": rc4b["off_gate_emissions"],
                             "RC4C": rc4c["off_gate_emissions"]},
            "regression_risk": {"RC4B": rc4b["regressions"], "RC4C": rc4c["regressions"]},
            "namespace_coverage_all": {"RC4B": rc4b["namespace_coverage_all"],
                                       "RC4C": rc4c["namespace_coverage_all"]},
            "wrapper_added_action_count": {"RC4B": rc4b["wrapper_added_action_count"],
                                           "RC4C": rc4c["wrapper_added_action_count"]},
            "deterministic": {"RC4B": rc4b["deterministic"], "RC4C": rc4c["deterministic"]},
        },
        "rc4c_residue_actions_beyond_rc4b": rc4c_residue_actions,
        "overlap_note": overlap_note,
        "decision": decision,
        "decision_rationale": (
            "Both candidates clear every READY gate (>=1 TRUE_DELTA, probe evidence replay, "
            "0 regressions, 0 off-gate, deterministic, wrapper-smoke no-regression). RC4C "
            "carries a deployment caveat (deploy bare `simp [L]`, not the fused combinator) "
            "and overlaps RC4B; its additive value for an RC4 composition is the residue "
            "actions only." if decision == "BOTH_READY_FOR_COMPOSITION"
            else "See per-candidate readiness checks."),
        "composition_guidance": (
            "If composed (separate task): order the gate as [RC4A, RC4B, RC4C_residue] so the "
            "namespace-parametric disjoint_left bridge (RC4B) is attributed before the depth-2 "
            "residue; de-duplicate the RC4B/RC4C overlap to avoid double-counting "
            f"({rc4c['true_delta_overlap_rc4b']} Multiset/Set disjoint wins). This matches the "
            "already-committed RC4D composition."),
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    def yn(b):
        return "✅" if b else "❌"

    md = [
        "# RC4B / RC4C joint candidate comparison", "",
        f"**Decision: `{decision}`**", "",
        "> Read-only comparison over the committed RC4B/RC4C validation record (commit "
        "`8f9d08e`). No live evaluation, no artifact mutation. Composition is a separate task.",
        "",
        "## Side-by-side", "",
        "| dimension | RC4B (disjoint_left bridge) | RC4C (d2 simp/aesop) |",
        "|---|---|---|",
        f"| committed decision | `RC4B_CANDIDATE_CONFIRMED` | `RC4C_CONFIRMED_WITH_RC4B_OVERLAP` |",
        f"| TRUE_DELTA (total) | {rc4b['true_delta_total']} | {rc4c['true_delta_total']} "
        f"(pure {rc4c['true_delta_pure']} + overlap-RC4B {rc4c['true_delta_overlap_rc4b']}) |",
        f"| mechanism | `{rc4b['mechanism']}` | `{rc4c['mechanism']}` |",
        f"| known wins | {rc4b['num_known_wins']} | {rc4c['num_known_wins']} |",
        f"| fresh-holdout wins | {rc4b['fresh_holdout_wins']} "
        f"(rate {rc4b['fresh_holdout_rate']}) | {rc4c['fresh_holdout_wins']} "
        f"(rate {rc4c['fresh_holdout_rate']}) |",
        f"| reproduction wins | {rc4b['reproduction_wins']} | {rc4c['reproduction_wins']} |",
        f"| namespace coverage (all) | {rc4b['namespace_coverage_all']} | "
        f"{rc4c['namespace_coverage_all']} |",
        f"| off-gate emissions | {rc4b['off_gate_emissions']} | {rc4c['off_gate_emissions']} |",
        f"| regressions | {rc4b['regressions']} | {rc4c['regressions']} |",
        f"| offgate verdict | {rc4b['offgate_verdict']} | {rc4c['offgate_verdict']} |",
        f"| deterministic | {yn(rc4b['deterministic'])} `{rc4b['determinism_hash']}` | "
        f"{yn(rc4c['deterministic'])} `{rc4c['determinism_hash']}` |",
        f"| schema-smoke (known reproduced / no-regr) | "
        f"{rc4b['schema_smoke']['known_wins_solved_by_wrapper']}/"
        f"{rc4b['schema_smoke']['known_wins_total']}, {yn(rc4b['schema_smoke']['no_regression'])} | "
        f"{rc4c['schema_smoke']['known_wins_solved_by_wrapper']}/"
        f"{rc4c['schema_smoke']['known_wins_total']}, {yn(rc4c['schema_smoke']['no_regression'])} |",
        f"| added gated actions (wrapper simplicity) | {rc4b['wrapper_added_action_count']} | "
        f"{rc4c['wrapper_added_action_count']} |",
        "",
        "## Readiness checks", "",
        "| check | RC4B | RC4C |", "|---|---|---|",
    ]
    for k in b_checks:
        md.append(f"| {k} | {yn(b_checks[k])} | {yn(c_checks[k])} |")
    md += [f"| **READY** | {yn(b_ready)} | {yn(c_ready)} |", ""]
    if b_caveats or c_caveats:
        md.append("### Caveats")
        for c in b_caveats:
            md.append(f"- **RC4B:** {c}")
        for c in c_caveats:
            md.append(f"- **RC4C:** {c}")
        md.append("")
    md += [
        "## Overlap & residue", "",
        f"- {overlap_note}",
        f"- RC4C residue actions beyond RC4B: `{rc4c_residue_actions}`",
        "",
        "## Composition guidance (not performed here)", "",
        f"- {out['composition_guidance']}",
        "",
        "## Decision", "",
        f"**`{decision}`** — {out['decision_rationale']}",
        "",
        "_Both candidates remain off-by-default; this comparison promotes nothing and "
        "composes nothing. The committed RC4D composition already realizes the guidance above._",
    ]
    with open(args.out_md, "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"[rc4bc] decision={decision}")
    print(f"[rc4bc] RC4B ready={b_ready} true_delta={rc4b['true_delta_total']} "
          f"fresh={rc4b['fresh_holdout_wins']}/{rc4b['num_known_wins']} "
          f"offgate={rc4b['off_gate_emissions']} regr={rc4b['regressions']}")
    print(f"[rc4bc] RC4C ready={c_ready} true_delta={rc4c['true_delta_total']} "
          f"(pure {rc4c['true_delta_pure']}/overlap {rc4c['true_delta_overlap_rc4b']}) "
          f"fresh={rc4c['fresh_holdout_wins']}/{rc4c['num_known_wins']} "
          f"offgate={rc4c['off_gate_emissions']} regr={rc4c['regressions']}")
    print(f"[rc4bc] residue beyond RC4B: {rc4c_residue_actions}")


if __name__ == "__main__":
    main()
