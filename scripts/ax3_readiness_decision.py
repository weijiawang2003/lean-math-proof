"""AX3 Stage 6 — symbolic-learning readiness decision.

Reads project/data/ax3_multiset_symbolic_dataset_meta.json and classifies:

  GREEN  — >=40 total clean symbolic labels AND >=20 in
           MULTISET_INDUCTION_SIMP[Multiset,simp_all] AND a held-out eval
           split exists untouched by training.
  YELLOW — 25-39 total clean labels with >=20 in the dominant label
           (smoke training only).
  RED    — <25 total clean labels or no held-out split (no training).

Output: project/data/ax3_readiness_meta.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "project/data/ax3_multiset_symbolic_dataset_meta.json"
OUT = ROOT / "project/data/ax3_readiness_meta.json"


def main() -> None:
    m = json.loads(META.read_text(encoding="utf-8"))
    total = m["clean_symbolic_labels_total"]
    simp_all = m["by_action_id"]["MULTISET_INDUCTION_SIMP[Multiset,simp_all]"]
    heldout_pos = m["heldout_positive_count"]
    train_pos = m["train_positive_count"]
    has_heldout = heldout_pos > 0

    green = total >= 40 and simp_all >= 20 and has_heldout
    yellow = (25 <= total < 40 or (total >= 40 and not green)) and \
        simp_all >= 20
    if green:
        verdict, train = "GREEN", True
    elif yellow or (total >= 25 and simp_all >= 20 and has_heldout):
        verdict, train = "YELLOW", True
    elif total >= 25 and has_heldout:
        verdict, train = "YELLOW", True
    else:
        verdict, train = "RED", False

    out = {
        "clean_symbolic_labels_total": total,
        "simp_all_count": simp_all,
        "train_positive_count": train_pos,
        "heldout_positive_count": heldout_pos,
        "has_heldout_split": has_heldout,
        "thresholds": {
            "green": ">=40 total AND >=20 simp_all AND held-out split",
            "yellow": "25-39 total AND >=20 dominant (smoke only)",
            "red": "<25 total or no held-out split",
        },
        "verdict": verdict,
        "train_recommended": train,
        "training_mode": ("full" if verdict == "GREEN"
                          else "smoke" if verdict == "YELLOW" else "none"),
        "label_distribution": m["label_distribution"],
        "split_x_label": m["split_x_label"],
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"verdict={verdict} train={train} mode={out['training_mode']}")
    print(f"  total_clean={total} simp_all={simp_all} "
          f"train_pos={train_pos} heldout_pos={heldout_pos}")


if __name__ == "__main__":
    main()
