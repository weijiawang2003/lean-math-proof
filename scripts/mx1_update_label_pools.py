"""MX1 Stage 6 — update the symbolic-label pools with the new MX1 labels.

Merges the clean single-shot symbolic-label pools across arcs and checks whether
any family (or sequence family) reaches a training threshold:

  single-action gate : >=40 total clean labels, OR >=20 in one family with a
                       held-out positive surface.
  sequence gate      : >=20 sequence-needed examples in one family.

Prior pools (canonical, from the committed arc reports / family-pool metas):
  - Multiset induction (WX3+AX3+AX4): the AX4 GREEN dataset — 46 clean labels
    (41 MULTISET_INDUCTION_SIMP[Multiset,simp_all], 5 [,simp]).
  - Option/List cases (WX1/WX2/AX1): wrapper-ready cases families
    (Option cases_simp 17, List cases_simp 9) — wrapper-ready, NOT SFT-ready.
  - SX1 sequences: 5 sequence-needed (biggest family 3) — far below the gate.
MX1 adds the new Finset/Set/Multiset clean labels from Stage 5.

Output: project/data/mx1_updated_symbolic_label_pools_meta.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "project/data/mx1_updated_symbolic_label_pools_meta.json"

# canonical prior single-shot symbolic-label pools (from committed reports)
PRIOR_SINGLE = {
    "MULTISET_INDUCTION_SIMP[Multiset,simp_all]": 41,
    "MULTISET_INDUCTION_SIMP[Multiset,simp]": 5,
    "CASES_SIMP[Option,simp]": 17,
    "CASES_SIMP[List,simp]": 6,
    "CASES_SIMP[List,simp_all]": 3,
    "CASES_SIMP[Option,simp_all]": 1,
}
PRIOR_SEQUENCE = {
    "SEQ[Multiset:induction=>aesop]": 3,
    "SEQ[List:cases=>cases]": 2,
}


def main() -> None:
    # MX1 new single-shot pools (Stage 5)
    mx1_pools = {}
    p = ROOT / "project/data/mx1_symbolic_family_pools_meta.json"
    if p.exists():
        d = json.loads(p.read_text())
        mx1_pools = {k: v.get("unique_count", 0)
                     for k, v in (d.get("pools") or {}).items()}

    # MX1 new sequence-needed labels (Stage 5 labels with classification)
    mx1_seq = defaultdict(int)
    lp = ROOT / "project/data/mx1_minimal_symbolic_frontier_labels.json"
    if lp.exists():
        for r in json.loads(lp.read_text()).get("labels", []):
            if r.get("classification") == "sequence_needed":
                fam = f"SEQ[{r['namespace']}]"
                mx1_seq[fam] += 1

    merged_single = defaultdict(int)
    for k, v in PRIOR_SINGLE.items():
        merged_single[k] += v
    new_by_family = defaultdict(int)
    for k, v in mx1_pools.items():
        merged_single[k] += v
        new_by_family[k] += v

    merged_seq = defaultdict(int)
    for k, v in PRIOR_SEQUENCE.items():
        merged_seq[k] += v
    for k, v in mx1_seq.items():
        merged_seq[k] += v

    total_single = sum(merged_single.values())
    total_new = sum(new_by_family.values())
    biggest_single = max(merged_single.values()) if merged_single else 0
    biggest_seq = max(merged_seq.values()) if merged_seq else 0

    # threshold checks
    multiset_simpall = merged_single.get(
        "MULTISET_INDUCTION_SIMP[Multiset,simp_all]", 0)
    single_gate = (total_single >= 40) or (
        biggest_single >= 20)  # held-out exists for Multiset
    # but the actionable question is a NEW trainable family from MX1:
    new_family_gate = any(v >= 20 for v in new_by_family.values())
    sequence_gate = biggest_seq >= 20

    out = {
        "description": "MX1 Stage 6 — merged symbolic-label pools across "
                       "AX/WX/SX/MX arcs + training-threshold check.",
        "single_shot_pools_merged": dict(sorted(
            merged_single.items(), key=lambda x: -x[1])),
        "mx1_new_single_shot_by_family": dict(new_by_family),
        "mx1_new_clean_labels_total": total_new,
        "sequence_pools_merged": dict(merged_seq),
        "mx1_new_sequence_by_family": dict(mx1_seq),
        "totals": {
            "single_shot_total": total_single,
            "biggest_single_family": biggest_single,
            "biggest_sequence_family": biggest_seq,
        },
        "thresholds": {
            "single_action_gate_met_globally": bool(single_gate),
            "single_action_gate_note":
                "Met historically by the Multiset induction family (already at "
                "GREEN under AX4); the open question is a NEW trainable family.",
            "new_mx1_family_reaches_gate": bool(new_family_gate),
            "sequence_gate_met": bool(sequence_gate),
        },
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"single-shot total={total_single} biggest={biggest_single} "
          f"| MX1 new clean={total_new} by_family={dict(new_by_family)}")
    print(f"sequence biggest={biggest_seq} | new_mx1_family_gate="
          f"{new_family_gate} sequence_gate={sequence_gate}")


if __name__ == "__main__":
    main()
