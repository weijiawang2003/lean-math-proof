"""AX2 Stage 6 — symbolic-action training-readiness decision.

Reads the merged symbolic-label dataset and the minimal-relabel family
pools, then computes whether there is enough clean data to train a
symbolic-action predictor (AX3) or whether more mining is needed (WX3).

Only examples whose minimal relabel says the symbolic action is actually
needed (symbolic_action_needed == True) count toward training totals; AX1
prototype examples are trusted (they passed the WX minimal relabel).

Readiness thresholds:
  Green : >=80 total, >=1 label with >=30, held-out symbolic-only available
  Yellow: 40-79 total, one dominant label with >=20  (tiny smoke only)
  Red   : <40 total  (do not train; mine more / extend wrapper)

Output: project/data/ax2_readiness_meta.json
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "project/data/ax2_symbolic_label_dataset_meta.json"
POOLS = ROOT / "project/data/ax2_symbolic_family_pools_meta.json"
OUT = ROOT / "project/data/ax2_readiness_meta.json"

GREEN_TOTAL, GREEN_LABEL = 80, 30
YELLOW_TOTAL_LO, YELLOW_LABEL = 40, 20


def entropy(counts):
    n = sum(counts)
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counts if c)


def main() -> None:
    if not DS.exists():
        sys.exit(f"missing {DS}; run Stage 4 first")
    ds = json.load(open(DS))
    examples = ds["examples"]

    # An example is training-eligible iff its symbolic action is needed.
    # AX1 examples carry symbolic_action_needed True; AX2 examples get it from
    # Stage 5. If Stage 5 hasn't run, fall back to symbolic_label presence.
    minimal_applied = ds.get("minimal_relabel_applied", False)

    def eligible(e):
        if e.get("symbolic_action_needed") is True:
            return True
        if not minimal_applied and e.get("symbolic_label"):
            return True
        return False

    def label_of(e):
        return e.get("final_training_label") or e.get("symbolic_label")

    elig = [e for e in examples if eligible(e)
            and label_of(e) and not str(label_of(e)).startswith("NON_SYMBOLIC")]
    total = len(elig)
    unique_thms = len({e["theorem"] for e in elig})
    by_label = Counter(label_of(e) for e in elig)
    by_ns = Counter(e["namespace"] for e in elig)
    by_arc = Counter(e["arc"] for e in elig)

    label_counts = list(by_label.values())
    max_label = max(label_counts) if label_counts else 0
    dom_label = by_label.most_common(1)[0][0] if by_label else None
    ent = entropy(label_counts)
    max_ent = math.log2(len(by_label)) if len(by_label) > 1 else 0.0
    balance = (ent / max_ent) if max_ent else (1.0 if len(by_label) == 1 else 0.0)

    # held-out feasibility: enough per-label to spare a test slice, and at
    # least one AX2 (freshly mined) symbolic-only example to hold out.
    ax2_count = by_arc.get("AX2", 0)
    held_out_possible = total >= 10 and max_label >= 4

    if total >= GREEN_TOTAL and max_label >= GREEN_LABEL and held_out_possible:
        readiness = "GREEN"
        rationale = (f"{total} examples, top label {max_label}>= {GREEN_LABEL}, "
                     "held-out feasible -> train AX3 symbolic-action predictor.")
    elif total >= YELLOW_TOTAL_LO and max_label >= YELLOW_LABEL:
        readiness = "YELLOW"
        rationale = (f"{total} examples, dominant label {max_label}>= "
                     f"{YELLOW_LABEL} -> tiny smoke classifier only.")
    else:
        readiness = "RED"
        rationale = (f"{total} examples (<{YELLOW_TOTAL_LO}) / top label "
                     f"{max_label} -> do not train; mine more (WX3) or extend "
                     "the wrapper.")

    # train/test split feasibility (stratified 80/20, need >=2 per kept label)
    keepable_labels = {lab: c for lab, c in by_label.items() if c >= 2}
    split_feasible = total >= 20 and len(keepable_labels) >= 2

    pools = json.load(open(POOLS)) if POOLS.exists() else {}

    out = {
        "minimal_relabel_applied": minimal_applied,
        "total_symbolic_examples": total,
        "unique_theorems": unique_thms,
        "examples_by_label": dict(by_label.most_common()),
        "examples_by_namespace": dict(by_ns.most_common()),
        "examples_by_arc": dict(by_arc.most_common()),
        "ax2_newly_mined_examples": ax2_count,
        "num_labels": len(by_label),
        "max_label_count": max_label,
        "dominant_label": dom_label,
        "label_entropy_bits": round(ent, 4),
        "label_balance_ratio": round(balance, 4),
        "train_test_split_feasible": split_feasible,
        "keepable_labels_ge2": dict(sorted(keepable_labels.items(),
                                           key=lambda kv: -kv[1])),
        "held_out_possible": held_out_possible,
        "thresholds": {
            "green": {"total": GREEN_TOTAL, "one_label": GREEN_LABEL,
                      "held_out": True},
            "yellow": {"total_lo": YELLOW_TOTAL_LO, "dominant_label": YELLOW_LABEL},
        },
        "readiness": readiness,
        "rationale": rationale,
        "minimal_pool_gate_met": pools.get("any_label_gate_met"),
        "recommendation": ("AX3 symbolic-action classifier training"
                           if readiness == "GREEN"
                           else "tiny smoke classifier then WX3 more mining"
                           if readiness == "YELLOW"
                           else "WX3 more symbolic mining / wrapper extension"),
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"total={total} unique_thms={unique_thms} labels={len(by_label)} "
          f"max_label={max_label}")
    print(f"by_label: {dict(by_label)}")
    print(f"entropy={ent:.3f} balance={balance:.3f} "
          f"split_feasible={split_feasible} held_out={held_out_possible}")
    print(f"READINESS: {readiness} -- {rationale}")


if __name__ == "__main__":
    main()
