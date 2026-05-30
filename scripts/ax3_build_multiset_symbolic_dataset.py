"""AX3 Stage 5 — build the Multiset symbolic-action dataset.

Merges clean single-shot symbolic labels from WX3 (20) and AX3 (newly mined)
into a labeled dataset for the first symbolic-action learner, and adds NULL
negatives so the classifier does not always emit induction:

  positives — clean single-shot `MULTISET_INDUCTION_SIMP[Multiset,*]` wins
              (no name-free tactic closes them from init).
  negatives — (a) Multiset states from the AX3 sets that are NOT clean
              induction_on wins (NULL), and (b) non-Multiset states from
              demo_v1 / nat_defs_medium (NULL false-positive control).

state_pp is recovered from eval traces (step 0) — AX3 relabel already stored
it for positives; WX3 positives and all negatives are pulled from the
corresponding eval-run traces. Each row carries split (train_candidate /
heldout_eval) and source arc (WX3 / AX3 / control).

Drops: over-attributed (aesop/simp), multi-step-assisted, flaky, and
simpler-raw-closable examples (i.e. anything not clean single-shot symbolic).

Outputs:
  project/data/ax3_multiset_symbolic_dataset_meta.json   (committed)
  project/data/ax3_multiset_symbolic_dataset.jsonl       (gitignored; rows)
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
AX3_PROBE = ROOT / "project/data/ax3_multiset_mining_probe_meta.json"
OUT_META = ROOT / "project/data/ax3_multiset_symbolic_dataset_meta.json"
OUT_JSONL = ROOT / "project/data/ax3_multiset_symbolic_dataset.jsonl"

NULL_LABEL = "NULL"
HELDOUT_SETS = {"ax3_multiset_induction_heldout",
                "ax3_multiset_mixed_heldout", "ax3_multiset_negative_control"}


def step0_states(rundir_glob):
    """full_name -> initial state_pp (first/min-step trace row per theorem).

    Traces are 1-indexed; the initial proof state is the state_pp of the
    earliest-step row for a theorem. We keep the row with the smallest step
    seen for each full_name.
    """
    best = {}  # full_name -> (step, state_pp)
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


def main() -> None:
    from core_types import build_prompt

    rows = []
    seen = set()

    def add(full_name, state_pp, label, split, arc, set_name, tactic=None,
            var=None):
        if not state_pp or full_name in seen:
            return False
        seen.add(full_name)
        rows.append({
            "full_name": full_name, "namespace": full_name.split(".")[0],
            "set": set_name, "arc": arc, "split": split,
            "label": label, "instantiated_tactic": tactic, "variable": var,
            "state_pp": state_pp,
            "prompt": build_prompt(state_pp, full_name),
            "is_multiset": full_name.startswith("Multiset."),
        })
        return True

    # ---- WX3 positives (arc WX3, split train) ----
    wx3 = json.load(open(WX3_LABELS)).get("relabel_results", [])
    # WX3 positives' state_pp: pull from WX3 comb/ind eval traces.
    wx3_states = step0_states(
        "project/evolve/eval_runs/wx3_comb_wx3_multiset_*/eval-*/traces.jsonl")
    wx3_states.update(step0_states(
        "project/evolve/eval_runs/wx3_ind_wx3_multiset_*/eval-*/traces.jsonl"))
    wx3_pos = 0
    for r in wx3:
        if r.get("single_shot_symbolic") and r.get("minimal_action_id"):
            sp = wx3_states.get(r["full_name"], "")
            if add(r["full_name"], sp, r["minimal_action_id"],
                   "train_candidate", "WX3", r.get("set"),
                   r.get("minimal_tactic"), r.get("minimal_var")):
                wx3_pos += 1

    # ---- AX3 positives (arc AX3; split by source set) ----
    ax3 = json.load(open(AX3_LABELS)).get("relabel_results", []) \
        if AX3_LABELS.exists() else []
    ax3_pos = 0
    ax3_clean_names = set()
    for r in ax3:
        if r.get("single_shot_symbolic") and r.get("minimal_action_id"):
            ax3_clean_names.add(r["full_name"])
            split = ("heldout_eval" if r.get("set") in HELDOUT_SETS
                     else "train_candidate")
            if add(r["full_name"], r.get("state_pp", ""),
                   r["minimal_action_id"], split, "AX3", r.get("set"),
                   r.get("minimal_tactic"), r.get("minimal_var")):
                ax3_pos += 1

    # ---- Multiset NULL negatives: AX3-set theorems not clean-positive ----
    ax3_set_states = {}
    for s in ("ax3_multiset_induction_mine", "ax3_multiset_induction_heldout",
              "ax3_multiset_mixed_heldout", "ax3_multiset_negative_control"):
        st = step0_states(
            f"project/evolve/eval_runs/ax3_wx3ind_{s}/eval-*/traces.jsonl")
        for fn, sp in st.items():
            ax3_set_states.setdefault(fn, (sp, s))
    ms_null = 0
    for fn, (sp, s) in sorted(ax3_set_states.items()):
        if fn in seen or fn in ax3_clean_names:
            continue
        split = "heldout_eval" if s in HELDOUT_SETS else "train_candidate"
        if add(fn, sp, NULL_LABEL, split, "AX3", s):
            ms_null += 1

    # ---- non-Multiset NULL control: demo_v1 / nat_defs_medium ----
    ctrl_null = 0
    for s, split in (("demo_v1", "train_candidate"),
                     ("nat_defs_medium", "heldout_eval")):
        st = step0_states(
            f"project/evolve/eval_runs/wx3_ns9_{s}/eval-*/traces.jsonl")
        for fn, sp in sorted(st.items()):
            if add(fn, sp, NULL_LABEL, split, "control", s):
                ctrl_null += 1

    # ---- write rows + meta ----
    OUT_JSONL.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")

    by_label = Counter(r["label"] for r in rows)
    by_split = Counter(r["split"] for r in rows)
    by_split_label = Counter((r["split"], r["label"]) for r in rows)
    pos_by_arc = Counter(r["arc"] for r in rows if r["label"] != NULL_LABEL)
    clean_total = sum(1 for r in rows if r["label"] != NULL_LABEL)
    simp_all = by_label.get("MULTISET_INDUCTION_SIMP[Multiset,simp_all]", 0)
    simp = by_label.get("MULTISET_INDUCTION_SIMP[Multiset,simp]", 0)

    meta = {
        "jsonl": str(OUT_JSONL.relative_to(ROOT)),
        "total_rows": len(rows),
        "clean_symbolic_labels_total": clean_total,
        "wx3_positives": wx3_pos, "ax3_positives": ax3_pos,
        "positives_by_arc": dict(pos_by_arc),
        "label_distribution": dict(by_label),
        "by_action_id": {
            "MULTISET_INDUCTION_SIMP[Multiset,simp_all]": simp_all,
            "MULTISET_INDUCTION_SIMP[Multiset,simp]": simp,
        },
        "split_distribution": dict(by_split),
        "split_x_label": {f"{k[0]}|{k[1]}": v
                          for k, v in sorted(by_split_label.items())},
        "negatives": {"multiset_null": ms_null,
                      "non_multiset_control_null": ctrl_null},
        "heldout_positive_count": sum(
            1 for r in rows if r["split"] == "heldout_eval"
            and r["label"] != NULL_LABEL),
        "train_positive_count": sum(
            1 for r in rows if r["split"] == "train_candidate"
            and r["label"] != NULL_LABEL),
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"wrote {OUT_JSONL.relative_to(ROOT)} ({len(rows)} rows)")
    print(f"wrote {OUT_META.relative_to(ROOT)}")
    print(f"clean symbolic labels total: {clean_total} "
          f"(WX3 {wx3_pos} + AX3 {ax3_pos})")
    print(f"  simp_all={simp_all} simp={simp}")
    print(f"label dist: {dict(by_label)}")
    print(f"split: {dict(by_split)}")
    print(f"train positives={meta['train_positive_count']} "
          f"heldout positives={meta['heldout_positive_count']}")


if __name__ == "__main__":
    main()
