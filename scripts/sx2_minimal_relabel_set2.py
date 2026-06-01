#!/usr/bin/env python3
"""SX2 Part 6 — NS23-style minimal-sufficient relabeling / attribution for SET2.

Promotion filter. For every theorem SET2 *solved*, decide whether the win is real
or an artifact, using ONLY data already captured live by sx2_run_set2_eval.py
(the per-theorem baseline battery outcomes) plus the mined template metadata. No
new Lean runs are needed: the eval already ran simp / simp_all / aesop /
classical<;>aesop on every theorem, which is exactly the minimal-sufficient
battery NS23 requires.

Attribution classes (per SET2 solve):
  TRUE_SET2_WIN        RC1(proxy) failed, ALL baselines failed, the SET2 tactic is
                       non-baseline and closed it, gate conditions valid.
  BASELINE_DUPLICATE   a baseline (simp/simp_all/aesop/classical<;>aesop) also closed
                       it, OR the emitting SET2 tactic IS itself a baseline string.
  SOURCE_SPECIFIC      the emitting gate's template uses theorem-specific named
                       lemmas / local hypotheses (per set_sequence_templates.json).
  PARSER_ARTIFACT      RC1's original failure was a single-line parser limitation,
                       not a capability gap (the SET2 tactic is the same shape RC1
                       would parse-fail on).
  NEEDS_DEEPER_SEQUENCE  emitted but did not solve (partial).

Outputs:
  set2_minimal_relabel_results.json
  set2_minimal_relabel_results.md
"""
from __future__ import annotations

import argparse
import json
import os

BASELINES = {"simp", "simp_all", "aesop", "classical <;> aesop"}


def _load(path):
    return json.load(open(path)) if path and os.path.exists(path) else None


def _template_index(templates_obj):
    """gate/family -> {uses_named_lemmas, uses_local_hyp, theorem_agnostic}."""
    idx = {}
    if not templates_obj:
        return idx
    for t in templates_obj.get("templates", []):
        idx[t["template_family"]] = t
    return idx


def classify(rec, tmpl_idx, mined_support):
    """Return (attribution_class, reason) for a single eval result row."""
    if not rec.get("set2_solved"):
        if rec.get("set2_emitted"):
            return "NEEDS_DEEPER_SEQUENCE", "SET2 emitted but did not close the goal"
        return "NEEDS_DEEPER_SEQUENCE", "no SET2 gate fired"
    tactic = rec.get("set2_tactic") or ""
    gate = rec.get("set2_gate")
    # which baselines solved this theorem (from the live battery)
    base_solved = [b["probe"] for b in rec.get("baseline_outcomes", [])
                   if b.get("solved")]
    if tactic.strip() in BASELINES:
        return "BASELINE_DUPLICATE", f"emitting SET2 tactic `{tactic}` is itself a baseline"
    if base_solved:
        return "BASELINE_DUPLICATE", f"baseline(s) also closed it: {base_solved}"
    # template-level theorem-specificity (gate family -> mined template)
    fam = tmpl_idx.get(gate)
    if fam and (fam.get("uses_named_lemmas") or fam.get("uses_local_hyp")):
        return "SOURCE_SPECIFIC", ("emitting family uses theorem-specific "
                                   f"lemmas/hyps (named={fam.get('uses_named_lemmas')}, "
                                   f"local_hyp={fam.get('uses_local_hyp')})")
    # speculative gate (0 mined support): generic near-baseline structural tactic
    # (ext<;>simp, constructor<;>intro<;>simp_all, apply antisymm...). The 4-tactic
    # baseline proxy is NARROWER than RC1's real top-11+ battery, so a generic
    # ext/constructor reduction is very likely within RC1's true reach. Do NOT credit
    # as a novel SET2 capability -> BASELINE_DUPLICATE (baseline-class).
    if mined_support.get(gate, 0) < 2:
        return "BASELINE_DUPLICATE", (
            f"gate {gate} is speculative (mined_support="
            f"{mined_support.get(gate, 0)}<2); `{tactic}` is a generic near-baseline "
            "structural reduction likely within RC1's real top-N battery (4-tactic "
            "proxy is narrower than RC1) — not a novel SET2 lever")
    return "TRUE_SET2_WIN", (f"RC1(proxy) failed, all baselines failed, non-baseline "
                             f"mined gate {gate} (`{tactic}`) closed it")


def relabel_set(results, tmpl_idx, mined_support, label):
    rows = []
    for rec in results:
        cls, reason = classify(rec, tmpl_idx, mined_support)
        rows.append({
            "set": label,
            "full_name": rec.get("full_name"),
            "rc1_solved": rec.get("rc1_solved"),
            "set2_emitted": rec.get("set2_emitted"),
            "set2_gate": rec.get("set2_gate"),
            "set2_tactic": rec.get("set2_tactic"),
            "set2_solved": rec.get("set2_solved"),
            "off_gate": rec.get("off_gate"),
            "baseline_solved_by": rec.get("baseline_solved_by"),
            "attribution": cls,
            "reason": reason,
        })
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--selected-results",
                   default="project/evolve/experiments/sx2/out/set2_selected_eval_results.json")
    p.add_argument("--holdout-results",
                   default="project/evolve/experiments/sx2/out/set2_holdout_eval_results.json")
    p.add_argument("--templates",
                   default="project/evolve/experiments/sx2/out/set_sequence_templates.json")
    p.add_argument("--gate-policy",
                   default="project/evolve/experiments/sx2/set2_gate_policy.json")
    p.add_argument("--out-json",
                   default="project/evolve/experiments/sx2/out/set2_minimal_relabel_results.json")
    p.add_argument("--out-md",
                   default="project/evolve/experiments/sx2/out/set2_minimal_relabel_results.md")
    args = p.parse_args(argv)

    tmpl_idx = _template_index(_load(args.templates))
    pol = _load(args.gate_policy) or {}
    mined_support = {g["gate_id"]: g.get("mined_support", {}).get("num_theorems", 0)
                     for g in pol.get("gates", [])}
    rows = []
    for path, label in ((args.selected_results, "selected"),
                        (args.holdout_results, "holdout")):
        obj = _load(path)
        if obj:
            rows += relabel_set(obj.get("results", []), tmpl_idx, mined_support, label)

    hist = {}
    for r in rows:
        hist[r["attribution"]] = hist.get(r["attribution"], 0) + 1
    true_wins = [r for r in rows if r["attribution"] == "TRUE_SET2_WIN"]
    true_by_gate = {}
    for r in true_wins:
        true_by_gate[r["set2_gate"]] = true_by_gate.get(r["set2_gate"], 0) + 1

    out = {
        "attribution_histogram": hist,
        "true_set2_win_count": len(true_wins),
        "true_set2_wins_by_gate": true_by_gate,
        "true_set2_wins": [r["full_name"] for r in true_wins],
        "off_gate_emissions": sum(1 for r in rows if r.get("off_gate")),
        "policy": "A SET2 solve is a TRUE_SET2_WIN only if RC1(proxy) and ALL "
                  "baselines failed, the gate is non-baseline and theorem-agnostic. "
                  "Mirrors NS23 minimal-sufficient attribution. No promotion claim "
                  "without a TRUE_SET2_WIN here.",
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), ensure_ascii=False, indent=2)

    L = ["# SX2 — SET2 Minimal-Sufficient Relabel (Attribution)", ""]
    L.append(f"- attribution histogram: `{hist}`")
    L.append(f"- **TRUE_SET2_WIN = {len(true_wins)}** by gate: {true_by_gate}")
    L.append(f"- off-gate emissions: {out['off_gate_emissions']}")
    L.append(f"- {out['policy']}")
    L.append("")
    L.append("| set | theorem | rc1 | gate | set2_tactic | solved | attribution | reason |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['set']} | `{r['full_name']}` | {r['rc1_solved']} | "
                 f"{r['set2_gate']} | `{(r['set2_tactic'] or '')}` | {r['set2_solved']} "
                 f"| **{r['attribution']}** | {r['reason'][:70]} |")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[sx2:relabel] rows={len(rows)} hist={hist} TRUE_SET2_WIN={len(true_wins)} "
          f"by_gate={true_by_gate} off_gate={out['off_gate_emissions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
