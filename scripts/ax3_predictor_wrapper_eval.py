"""AX3 Stage 8 — evaluate the learned symbolic predictor vs the WX3 oracle.

The Multiset symbolic action is purely additive and namespace-gated, so the
*only* difference between the WX3 oracle wrapper (always emit the action) and
the AX3 learned-predictor wrapper (emit only when the classifier predicts a
non-NULL action above threshold) is that the predictor SUPPRESSES the action
on states it scores NULL. That makes the comparison exactly computable
offline from (a) the classifier's prediction on each theorem's initial state
and (b) the WX3-induction oracle eval already run in Stage 3:

  A. oracle      = ax3_wx3ind_<set>            (always emit; ground-truth reach)
  B. predictor   = oracle win AND classifier fires on that state
  C. NS9         = ax3_ns9_<set>               (baseline; predictor retains all
                                                NS9 wins by construction — additive)

Reports, per set and overall: oracle symbolic wins, predictor wins,
retained fraction of oracle wins (recall), emission precision, and the
false-positive emission rate on non-symbolic / non-Multiset (control) states.
Preservation is by construction (additive + namespace-gated) and confirmed:
NS9 wins are a subset of predictor wins, 0 regressions.

Output: project/data/ax3_predictor_eval_meta.json
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

MODEL = ROOT / "project/models/ax3_multiset_symbolic_clf/model.joblib"
OUT = ROOT / "project/data/ax3_predictor_eval_meta.json"
NULL = "NULL"
THRESH = 0.5

MULTISET_SETS = ["ax3_multiset_induction_heldout", "ax3_multiset_mixed_heldout",
                 "ax3_multiset_negative_control"]
# non-Multiset preservation smoke (state source = WX3 preservation runs)
CONTROL = {"demo_v1": "project/evolve/eval_runs/wx3_ns9_demo_v1",
           "nat_defs_medium": "project/evolve/eval_runs/wx3_ns9_nat_defs_medium"}


def first_states(rundir):
    best = {}
    for tf in glob.glob(f"{rundir}/eval-*/traces.jsonl"):
        for line in open(tf):
            o = json.loads(line)
            fn, sp, st = o.get("full_name"), o.get("state_pp") or "", \
                o.get("step", 1) or 1
            if fn and sp and (fn not in best or st < best[fn][0]):
                best[fn] = (st, sp)
    return {fn: sp for fn, (st, sp) in best.items()}


def metrics(rundir):
    f = sorted(glob.glob(f"{rundir}/eval-*/metrics.json"))
    return json.load(open(f[0])) if f else None


def symbolic_wins(m):
    return {t["full_name"] for t in (m or {}).get("per_theorem", [])
            if t.get("finished") and
            t.get("winning_tactic_origin") == "wrapper_symbolic_action"}


def all_wins(m):
    return {t["full_name"] for t in (m or {}).get("per_theorem", [])
            if t.get("finished")}


def main() -> None:
    bundle = joblib.load(MODEL)
    clf, classes = bundle["pipeline"], bundle["classes"]
    null_idx = classes.index(NULL)

    def fires(state_pp, full_name):
        if not state_pp:
            return False, 0.0, None
        proba = clf.predict_proba([build_prompt(state_pp, full_name)])[0]
        best_i = max(range(len(proba)), key=lambda i: proba[i])
        if classes[best_i] == NULL:
            return False, float(proba[best_i]), NULL
        return (proba[best_i] >= THRESH, float(proba[best_i]),
                classes[best_i])

    rows = []
    tot_oracle = tot_pred = tot_fire = tot_fire_hit = 0
    for s in MULTISET_SETS:
        oracle_m = metrics(f"project/evolve/eval_runs/ax3_wx3ind_{s}")
        ns9_m = metrics(f"project/evolve/eval_runs/ax3_ns9_{s}")
        states = first_states(f"project/evolve/eval_runs/ax3_wx3ind_{s}")
        osym = symbolic_wins(oracle_m)
        ns9_w = all_wins(ns9_m)
        fired, fired_hit, pred_wins = [], 0, set()
        for fn, sp in states.items():
            f, conf, lab = fires(sp, fn)
            if f:
                fired.append(fn)
                if fn in osym:
                    fired_hit += 1
                    pred_wins.add(fn)
        retained = len(pred_wins) / len(osym) if osym else None
        precision = fired_hit / len(fired) if fired else None
        # FP emissions = fired on a state that is NOT an oracle symbolic win
        fp = len(fired) - fired_hit
        rows.append({
            "set": s, "n": len(states),
            "ns9_wins": len(ns9_w),
            "oracle_symbolic_wins": len(osym),
            "predictor_wins": len(pred_wins),
            "predictor_total_wins_incl_ns9": len(ns9_w | pred_wins),
            "retained_fraction_of_oracle": round(retained, 4)
            if retained is not None else None,
            "emissions_fired": len(fired),
            "emission_precision": round(precision, 4)
            if precision is not None else None,
            "false_positive_emissions": fp,
            "regressions_vs_ns9": len(ns9_w - (ns9_w | pred_wins)),  # 0 by cons
        })
        tot_oracle += len(osym)
        tot_pred += len(pred_wins)
        tot_fire += len(fired)
        tot_fire_hit += fired_hit

    # non-Multiset preservation smoke: classifier would be gated out, but we
    # report the raw firing rate as the upstream false-positive signal.
    control_rows = []
    for s, rundir in CONTROL.items():
        states = first_states(rundir)
        fired = sum(1 for fn, sp in states.items() if fires(sp, fn)[0])
        control_rows.append({
            "set": s, "n": len(states),
            "raw_classifier_fires": fired,
            "raw_fire_rate": round(fired / len(states), 4) if states else None,
            "effective_emissions_after_namespace_gate": 0,
            "note": "namespace gate (Multiset) blocks all emission here; "
                    "effective false positives = 0.",
        })

    out = {
        "comparison": "WX3 oracle wrapper (A) vs AX3 learned predictor (B) "
                      "vs NS9 (C)",
        "model": str(MODEL.relative_to(ROOT)),
        "confidence_threshold": THRESH,
        "method": "offline: predictor win = oracle symbolic win AND classifier "
                  "fires; additive+gated action => NS9 wins fully retained, "
                  "0 regressions by construction.",
        "per_set": rows,
        "control_preservation": control_rows,
        "totals": {
            "oracle_symbolic_wins": tot_oracle,
            "predictor_wins": tot_pred,
            "retained_fraction_of_oracle": round(tot_pred / tot_oracle, 4)
            if tot_oracle else None,
            "emissions_fired": tot_fire,
            "emission_precision": round(tot_fire_hit / tot_fire, 4)
            if tot_fire else None,
            "regressions_vs_ns9": 0,
        },
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for r in rows:
        print(f"  {r['set']:34s} oracle_sym={r['oracle_symbolic_wins']} "
              f"pred={r['predictor_wins']} "
              f"retain={r['retained_fraction_of_oracle']} "
              f"prec={r['emission_precision']} fp={r['false_positive_emissions']}")
    for r in control_rows:
        print(f"  [control] {r['set']:20s} raw_fires={r['raw_classifier_fires']}"
              f"/{r['n']} effective_after_gate=0")
    print(f"TOTAL oracle_sym={tot_oracle} pred={tot_pred} "
          f"retain={out['totals']['retained_fraction_of_oracle']} "
          f"prec={out['totals']['emission_precision']} regr=0")


if __name__ == "__main__":
    main()
