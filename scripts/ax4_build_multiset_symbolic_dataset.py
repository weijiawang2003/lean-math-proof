"""AX4 Stage 5 — merge the expanded Multiset symbolic-action dataset + classify
readiness.

Merges clean single-shot symbolic labels from WX3 + AX3 + AX4 (newly mined),
deduplicates by theorem name, adds NULL negatives, assigns a clean train /
held-out split by source-set membership, and computes the AX4 Green/Yellow/Red
readiness verdict.

  positives — clean single-shot MULTISET_INDUCTION_SIMP[Multiset,*] wins
              (no name-free tactic closes them from init), across WX3/AX3/AX4.
  negatives — (a) Multiset states from AX3/AX4 mining sets that are NOT clean
              induction wins (NULL); (b) non-Multiset demo_v1 / nat_defs_medium
              states (NULL false-positive control).

Held-out split = labels/states whose SOURCE SET is a reserved held-out set
(AX4 held-out + AX3 held-out/mixed/negative). These are mined with the oracle
but excluded from training, giving an honest held-out positive surface.

AX4 readiness:
  GREEN  — >=40 total clean labels AND >=30 simp_all AND >=10 held-out
           positives AND negative controls available.
  YELLOW — >=25 total clean labels otherwise.
  RED     — <25 total or no held-out split.

Outputs:
  project/data/ax4_multiset_symbolic_dataset_meta.json   (committed)
  project/data/ax4_multiset_symbolic_dataset.jsonl       (gitignored; rows)
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WX3_LABELS = ROOT / "project/data/wx3_minimal_multiset_labels.json"
AX3_LABELS = ROOT / "project/data/ax3_minimal_multiset_symbolic_labels.json"
AX4_LABELS = ROOT / "project/data/ax4_minimal_multiset_symbolic_labels.json"
OUT_META = ROOT / "project/data/ax4_multiset_symbolic_dataset_meta.json"
OUT_JSONL = ROOT / "project/data/ax4_multiset_symbolic_dataset.jsonl"

NULL_LABEL = "NULL"
SIMP_ALL = "MULTISET_INDUCTION_SIMP[Multiset,simp_all]"
SIMP = "MULTISET_INDUCTION_SIMP[Multiset,simp]"

# reserved held-out source sets (mined with oracle, excluded from training)
HELDOUT_SETS = {
    "ax4_multiset_induction_heldout", "ax4_multiset_induction_heldout2",
    "ax4_multiset_negative_control",  # reserved expected-NULL set
    "ax3_multiset_induction_heldout", "ax3_multiset_mixed_heldout",
    "ax3_multiset_negative_control",
    "nat_defs_medium",  # non-Multiset control routed to held-out for FP
}
# AX4 mining sets whose wx3ind traces feed Multiset NULL negatives
AX4_MINE_SETS = [
    "ax4_multiset_induction_high_confidence", "ax4_multiset_cross_surface",
    "ax4_multiset_induction_heldout", "ax4_multiset_induction_medium_confidence",
    "ax4_multiset_induction_hard", "ax4_multiset_negative_control",
    "ax4_multiset_induction_heldout2",
]
AX3_MINE_SETS = ["ax3_multiset_induction_mine",
                 "ax3_multiset_induction_heldout",
                 "ax3_multiset_mixed_heldout", "ax3_multiset_negative_control"]


def step0_states(rundir_glob):
    """full_name -> initial state_pp (smallest-step trace row per theorem)."""
    best = {}
    for tf in sorted(glob.glob(rundir_glob)):
        try:
            for line in open(tf):
                o = json.loads(line)
                fn = o.get("full_name")
                sp = o.get("state_pp", "") or ""
                st = o.get("step", 1) or 1
                if not fn or not sp:
                    continue
                if fn not in best or st < best[fn][0]:
                    best[fn] = (st, sp)
        except Exception:
            pass
    return {fn: sp for fn, (st, sp) in best.items()}


def load_clean(path):
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8")).get(
        "relabel_results", [])
    return [r for r in rows
            if r.get("single_shot_symbolic") and r.get("minimal_action_id")]


def main() -> None:
    from core_types import build_prompt

    rows = []
    seen = set()

    def add(full_name, state_pp, label, arc, set_name, tactic=None, var=None):
        if not state_pp or full_name in seen:
            return False
        seen.add(full_name)
        split = "heldout_eval" if set_name in HELDOUT_SETS else "train_candidate"
        rows.append({
            "full_name": full_name, "namespace": full_name.split(".")[0],
            "set": set_name, "arc": arc, "split": split,
            "label": label, "instantiated_tactic": tactic, "variable": var,
            "state_pp": state_pp,
            "prompt": build_prompt(state_pp, full_name),
            "is_multiset": full_name.startswith("Multiset."),
        })
        return True

    # ---- WX3 positives (state_pp from WX3 eval traces) ----
    wx3_states = step0_states(
        "project/evolve/eval_runs/wx3_comb_wx3_multiset_*/eval-*/traces.jsonl")
    wx3_states.update(step0_states(
        "project/evolve/eval_runs/wx3_ind_wx3_multiset_*/eval-*/traces.jsonl"))
    wx3_pos = 0
    for r in load_clean(WX3_LABELS):
        if add(r["full_name"], wx3_states.get(r["full_name"], ""),
               r["minimal_action_id"], "WX3", r.get("set"),
               r.get("minimal_tactic"), r.get("minimal_var")):
            wx3_pos += 1

    # ---- AX3 positives (state_pp stored in relabel) ----
    ax3_pos = 0
    clean_names = set()
    for r in load_clean(AX3_LABELS):
        clean_names.add(r["full_name"])
        if add(r["full_name"], r.get("state_pp", ""), r["minimal_action_id"],
               "AX3", r.get("set"), r.get("minimal_tactic"),
               r.get("minimal_var")):
            ax3_pos += 1

    # ---- AX4 positives (state_pp stored in relabel) ----
    ax4_pos = 0
    for r in load_clean(AX4_LABELS):
        clean_names.add(r["full_name"])
        if add(r["full_name"], r.get("state_pp", ""), r["minimal_action_id"],
               "AX4", r.get("set"), r.get("minimal_tactic"),
               r.get("minimal_var")):
            ax4_pos += 1

    # ---- Multiset NULL negatives from AX3+AX4 wx3ind mining traces ----
    set_states = {}
    for s in AX4_MINE_SETS:
        for fn, sp in step0_states(
                f"project/evolve/eval_runs/ax4_wx3ind_{s}/eval-*/traces.jsonl"
        ).items():
            set_states.setdefault(fn, (sp, s))
    for s in AX3_MINE_SETS:
        for fn, sp in step0_states(
                f"project/evolve/eval_runs/ax3_wx3ind_{s}/eval-*/traces.jsonl"
        ).items():
            set_states.setdefault(fn, (sp, s))
    ms_null = 0
    for fn, (sp, s) in sorted(set_states.items()):
        if fn in seen or fn in clean_names:
            continue
        if add(fn, sp, NULL_LABEL, "mining", s):
            ms_null += 1

    # ---- non-Multiset NULL control (demo_v1 -> train, nat_defs_medium ->
    # held-out, mirroring AX3; control rows are arc-tagged for FP analysis) ----
    ctrl_null = 0
    for s, set_name in (("demo_v1", "demo_v1"),
                        ("nat_defs_medium", "nat_defs_medium")):
        for fn, sp in sorted(step0_states(
                f"project/evolve/eval_runs/wx3_ns9_{s}/eval-*/traces.jsonl"
        ).items()):
            if add(fn, sp, NULL_LABEL, "control", set_name):
                ctrl_null += 1

    OUT_JSONL.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")

    by_label = Counter(r["label"] for r in rows)
    by_split = Counter(r["split"] for r in rows)
    by_split_label = Counter((r["split"], r["label"]) for r in rows)
    pos_by_arc = Counter(r["arc"] for r in rows if r["label"] != NULL_LABEL)
    clean_total = sum(1 for r in rows if r["label"] != NULL_LABEL)
    simp_all = by_label.get(SIMP_ALL, 0)
    simp = by_label.get(SIMP, 0)
    heldout_pos = sum(1 for r in rows if r["split"] == "heldout_eval"
                      and r["label"] != NULL_LABEL)
    train_pos = sum(1 for r in rows if r["split"] == "train_candidate"
                    and r["label"] != NULL_LABEL)
    neg_controls = ms_null > 0 and ctrl_null > 0

    # ---- AX4 readiness ----
    green = (clean_total >= 40 and simp_all >= 30 and heldout_pos >= 10
             and neg_controls)
    if green:
        verdict, mode = "GREEN", "full"
    elif clean_total >= 25 and heldout_pos > 0:
        verdict, mode = "YELLOW", "smoke"
    else:
        verdict, mode = "RED", "none"

    meta = {
        "arc": "AX4",
        "jsonl": str(OUT_JSONL.relative_to(ROOT)),
        "total_rows": len(rows),
        "clean_symbolic_labels_total": clean_total,
        "positives_by_arc": dict(pos_by_arc),
        "wx3_positives": wx3_pos, "ax3_positives": ax3_pos,
        "ax4_positives": ax4_pos,
        "label_distribution": dict(by_label),
        "by_action_id": {SIMP_ALL: simp_all, SIMP: simp},
        "split_distribution": dict(by_split),
        "split_x_label": {f"{k[0]}|{k[1]}": v
                          for k, v in sorted(by_split_label.items())},
        "negatives": {"multiset_null": ms_null,
                      "non_multiset_control_null": ctrl_null,
                      "negative_controls_available": neg_controls},
        "heldout_positive_count": heldout_pos,
        "train_positive_count": train_pos,
        "readiness": {
            "thresholds": {
                "green": ">=40 total AND >=30 simp_all AND >=10 held-out "
                         "positives AND negative controls",
                "yellow": ">=25 total with held-out split",
                "red": "<25 total or no held-out",
            },
            "verdict": verdict,
            "training_mode": mode,
            "train_recommended": verdict in ("GREEN", "YELLOW"),
        },
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"wrote {OUT_JSONL.relative_to(ROOT)} ({len(rows)} rows)")
    print(f"wrote {OUT_META.relative_to(ROOT)}")
    print(f"clean labels total: {clean_total} "
          f"(WX3 {wx3_pos} + AX3 {ax3_pos} + AX4 {ax4_pos})")
    print(f"  simp_all={simp_all} simp={simp}")
    print(f"  train_pos={train_pos} heldout_pos={heldout_pos}")
    print(f"  negatives: multiset_null={ms_null} control={ctrl_null}")
    print(f"VERDICT: {verdict} (mode={mode})")


if __name__ == "__main__":
    main()
