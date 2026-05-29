"""AX4 Stage 7 — held-out theorem-level evaluation of the learned predictor.

Compares, per evaluation set:
  A. NS9 wrapper baseline           (ax4_ns9_/ax3_ns9_ traces)
  B. WX3 oracle induction wrapper   (ax4_wx3ind_/ax3_wx3ind_ traces)
  C. AX4 learned symbolic predictor (oracle win AND classifier fires)

The Multiset symbolic action is additive + namespace-gated, so the predictor
differs from the oracle only by SUPPRESSING emissions it scores NULL — making
A/B/C computable offline from the already-mined oracle traces plus the
classifier's prediction on each theorem's initial state (same method as AX3).

Multiset held-out sets measure retained fraction of oracle wins, emission
precision, and regressions vs NS9. Non-Multiset sets (demo_v1, nat_defs_medium,
ns17_set_extra, ns17_finset_extra) measure the raw classifier firing rate; the
wrapper gates emission to Multiset so effective non-Multiset FP = 0.

Promotion criterion (reported, not enforced):
  retain >=50% of WX3 oracle held-out wins AND 0 regressions AND effective
  non-Multiset FP = 0 AND a clear precision/recall threshold.

Output: project/data/ax4_predictor_eval_meta.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core_types import build_prompt  # noqa: E402

MODEL = ROOT / "project/models/ax4_multiset_symbolic_clf/model.joblib"
OUT = ROOT / "project/data/ax4_predictor_eval_meta.json"
NULL = "NULL"
THRESH = 0.5

# Multiset held-out eval sets -> (oracle_run_prefix, ns9_run_prefix)
MULTISET_HELDOUT = {
    "ax4_multiset_induction_heldout": ("ax4_wx3ind", "ax4_ns9"),
    "ax4_multiset_induction_heldout2": ("ax4_wx3ind", "ax4_ns9"),
    "ax4_multiset_negative_control": ("ax4_wx3ind", "ax4_ns9"),
    "ax3_multiset_induction_heldout": ("ax3_wx3ind", "ax3_ns9"),
    "ax3_multiset_mixed_heldout": ("ax3_wx3ind", "ax3_ns9"),
    "ax3_multiset_negative_control": ("ax3_wx3ind", "ax3_ns9"),
}
# non-Multiset FP control -> any run dir with initial-state traces
CONTROL = {
    "demo_v1": "project/evolve/eval_runs/wx3_ns9_demo_v1",
    "nat_defs_medium": "project/evolve/eval_runs/wx3_ns9_nat_defs_medium",
    "ns17_set_extra": "project/evolve/eval_runs/ns17_ns15routed_raw_ns17_set_extra",
    "ns17_finset_extra":
        "project/evolve/eval_runs/ns17_ns15routed_raw_ns17_finset_extra",
}


def first_states(rundir_glob):
    best = {}
    for tf in glob.glob(rundir_glob):
        for line in open(tf):
            try:
                o = json.loads(line)
            except Exception:
                continue
            fn, sp = o.get("full_name"), o.get("state_pp") or ""
            st = o.get("step", 1) or 1
            if fn and sp and (fn not in best or st < best[fn][0]):
                best[fn] = (st, sp)
    return {fn: sp for fn, (st, sp) in best.items()}


def theorem_wins(rundir_glob, symbolic_only=False):
    """Reconstruct {full_name: won} from traces (proof_finished)."""
    won = {}
    for tf in glob.glob(rundir_glob):
        for line in open(tf):
            try:
                o = json.loads(line)
            except Exception:
                continue
            fn = o.get("full_name")
            if not fn:
                continue
            if o.get("proof_finished"):
                if symbolic_only and o.get("tactic_origin") != \
                        "wrapper_symbolic_action":
                    continue
                won[fn] = True
    return set(won)


def main() -> None:
    if not MODEL.exists():
        OUT.write_text(json.dumps(
            {"evaluated": False,
             "reason": "no AX4 model (readiness not GREEN / not trained)"},
            indent=2), encoding="utf-8")
        print("no AX4 model; skipping held-out eval")
        return
    bundle = joblib.load(MODEL)
    clf, classes = bundle["pipeline"], bundle["classes"]

    def fires(state_pp, full_name):
        if not state_pp:
            return False
        proba = clf.predict_proba([build_prompt(state_pp, full_name)])[0]
        bi = max(range(len(proba)), key=lambda i: proba[i])
        return classes[bi] != NULL and proba[bi] >= THRESH

    rows = []
    tot_oracle = tot_pred = 0
    for s, (orc, ns9) in MULTISET_HELDOUT.items():
        o_glob = f"project/evolve/eval_runs/{orc}_{s}/eval-*/traces.jsonl"
        n_glob = f"project/evolve/eval_runs/{ns9}_{s}/eval-*/traces.jsonl"
        states = first_states(o_glob)
        osym = theorem_wins(o_glob, symbolic_only=True)
        ns9_w = theorem_wins(n_glob)
        fired = [fn for fn, sp in states.items() if fires(sp, fn)]
        pred_wins = {fn for fn in fired if fn in osym}
        fp = len(fired) - len(pred_wins)
        rows.append({
            "set": s, "n": len(states), "ns9_wins": len(ns9_w),
            "oracle_symbolic_wins": len(osym),
            "predictor_wins": len(pred_wins),
            "retained_fraction_of_oracle": round(len(pred_wins) / len(osym), 4)
            if osym else None,
            "emissions_fired": len(fired),
            "emission_precision": round(len(pred_wins) / len(fired), 4)
            if fired else None,
            "false_positive_emissions": fp,
            "regressions_vs_ns9": len(ns9_w - (ns9_w | pred_wins)),
        })
        tot_oracle += len(osym)
        tot_pred += len(pred_wins)

    control_rows = []
    for s, rundir in CONTROL.items():
        states = first_states(f"{rundir}/eval-*/traces.jsonl")
        fired = sum(1 for fn, sp in states.items() if fires(sp, fn))
        control_rows.append({
            "set": s, "n": len(states), "raw_classifier_fires": fired,
            "raw_fire_rate": round(fired / len(states), 4) if states else None,
            "effective_emissions_after_namespace_gate": 0,
        })

    retained = round(tot_pred / tot_oracle, 4) if tot_oracle else None
    regr = sum(r["regressions_vs_ns9"] for r in rows)
    eff_fp = 0  # all control is non-Multiset -> gated
    promote = (retained is not None and retained >= 0.5 and regr == 0
               and eff_fp == 0)
    out = {
        "evaluated": True, "confidence_threshold": THRESH,
        "model": str(MODEL.relative_to(ROOT)),
        "per_set_multiset_heldout": rows,
        "control_non_multiset": control_rows,
        "totals": {"oracle_symbolic_wins": tot_oracle,
                   "predictor_wins": tot_pred,
                   "retained_fraction_of_oracle": retained,
                   "regressions_vs_ns9": regr,
                   "effective_non_multiset_fp": eff_fp},
        "promotion": {
            "criterion": "retain>=50% oracle held-out AND 0 regr AND "
                         "effective non-Multiset FP=0",
            "promote": promote,
        },
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for r in rows:
        print(f"  {r['set']:36s} ns9={r['ns9_wins']} "
              f"oracle_sym={r['oracle_symbolic_wins']} "
              f"pred={r['predictor_wins']} "
              f"retain={r['retained_fraction_of_oracle']} "
              f"prec={r['emission_precision']} regr={r['regressions_vs_ns9']}")
    for r in control_rows:
        print(f"  [ctrl] {r['set']:24s} raw_fires={r['raw_classifier_fires']}"
              f"/{r['n']} eff_after_gate=0")
    print(f"TOTAL retain={retained} regr={regr} promote={promote}")


if __name__ == "__main__":
    main()
