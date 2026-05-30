"""SX1 Stage 7 — minimal relabeling of sequence(-assisted) wins.

For every depth-2 sequence-assisted close found in Stage 1/6, determine the
*minimal* proof the trace corpus supports, so we never credit a depth-2 plan
for something a single tactic already closes. We replay the theorem's mined
records and ask, from the THEOREM's initial state:

  * raw_tactic_over_attribution   — a plain battery tactic (simp / simp_all /
                                    aesop / rfl / omega / decide) closes in ONE
                                    step from the initial state. The depth-2
                                    sequence is unnecessary.
  * single_action_over_attribution — a single symbolic action closes in one step
                                    (the WX3/WX2 single-action oracle suffices).
  * genuinely_sequence_needed     — no single step closes from the initial state;
                                    the proof requires the symbolic first action
                                    to advance AND a follow-up to close.
  * flaky                         — the only observed closer is a non-
                                    deterministic generative tactic with no
                                    deterministic battery equivalent reproducible
                                    from the resulting state.

Outputs:
  project/data/sx1_minimal_sequence_labels.json
  project/data/sx1_sequence_family_pools_meta.json
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = json.loads(
    (ROOT / "project/data/sx1_multistep_symbolic_cases_meta.json").read_text())
OUT_LABELS = ROOT / "project/data/sx1_minimal_sequence_labels.json"
OUT_POOLS = ROOT / "project/data/sx1_sequence_family_pools_meta.json"
SYM = "wrapper_symbolic_action"
BATTERY = {"simp", "simp_all", "aesop", "rfl", "omega", "decide",
           "assumption", "decide!", "trivial"}
WRAPPER_TAGS = ["ax4_wx3ind", "ax3_wx3ind", "ax2_ax1sym"]


def load_records(fn):
    recs = []
    for wtag in WRAPPER_TAGS:
        for tf in glob.glob(
                f"project/evolve/eval_runs/{wtag}_*/eval-*/traces.jsonl"):
            for line in open(tf):
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if o.get("full_name") == fn:
                    recs.append(o)
    return recs


def initial_hash(recs):
    for r in recs:
        if r.get("step") == 1 and r.get("state_hash_before"):
            return r.get("state_hash_before")
    return None


def classify(case):
    fn = case["theorem"]
    recs = load_records(fn)
    ih = initial_hash(recs)
    # single-step closers from the initial state
    raw_closer = sym_closer = None
    for r in recs:
        if r.get("result_kind") != "ProofFinished":
            continue
        if r.get("state_hash_before") != ih:
            continue
        t = (r.get("tactic") or "").strip()
        o = r.get("tactic_origin")
        if o == SYM and sym_closer is None:
            sym_closer = t
        elif t in BATTERY and raw_closer is None:
            raw_closer = t
        elif t.split()[0] in BATTERY and raw_closer is None:
            raw_closer = t
    if raw_closer is not None:
        return "raw_tactic_over_attribution", raw_closer
    if sym_closer is not None:
        return "single_action_over_attribution", sym_closer
    # depth-2: the closer from the resulting state — deterministic battery?
    closer = (case.get("closing_tactic") or "").strip()
    origin = case.get("closing_tactic_origin")
    if closer.split()[0] in BATTERY or origin == SYM:
        return "genuinely_sequence_needed", closer
    return "flaky", closer


def main() -> None:
    labels = []
    for case in CASES.get("multistep_cases", []):
        cls, tac = classify(case)
        labels.append({
            "theorem": case["theorem"], "namespace": case["namespace"],
            "source_arc": case["source_arc"],
            "first_symbolic_tactic": case["first_tactic"],
            "closing_tactic": case["closing_tactic"],
            "closing_tactic_origin": case["closing_tactic_origin"],
            "classification": cls, "minimal_closer": tac,
        })

    by_cls = defaultdict(int)
    for r in labels:
        by_cls[r["classification"]] += 1

    # sequence family pools: only the genuinely_sequence_needed labels are
    # candidate training rows for a future AX5 sequence-label learner.
    pools = defaultdict(list)
    for r in labels:
        if r["classification"] != "genuinely_sequence_needed":
            continue
        fam = f"SEQ[{r['namespace']}:{r['first_symbolic_tactic'].split()[0]}" \
              f"=>{r['closing_tactic'].split()[0]}]"
        pools[fam].append({
            "theorem": r["theorem"],
            "first_symbolic_tactic": r["first_symbolic_tactic"],
            "followup_tactic": r["closing_tactic"],
        })

    labels_out = {
        "description": "SX1 Stage 7 — minimal relabeling of depth-2 "
                       "sequence-assisted wins (offline, trace-replay).",
        "n_cases": len(labels),
        "by_classification": dict(by_cls),
        "labels": labels,
    }
    OUT_LABELS.write_text(json.dumps(labels_out, indent=2, ensure_ascii=False),
                          encoding="utf-8")

    pools_out = {
        "description": "SX1 sequence family pools — genuinely-sequence-needed "
                       "depth-2 labels only (candidate AX5 training rows).",
        "gate_unique_required": 5,
        "n_genuine_sequence_labels":
            sum(len(v) for v in pools.values()),
        "biggest_pool": max((len(v) for v in pools.values()), default=0),
        "pools": {k: {"unique_count": len(v), "labels": v}
                  for k, v in pools.items()},
        "note": ("A future AX5 sequence-label learner needs >=5 unique labels "
                 "in a single family before training is worthwhile (cf. AX3/AX4 "
                 "count gate). The current corpus is far below that — SX1 lands "
                 "as dataset-generation, not a trainable sequence learner yet."),
    }
    OUT_POOLS.write_text(json.dumps(pools_out, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    print(f"wrote {OUT_LABELS.relative_to(ROOT)}")
    print(f"wrote {OUT_POOLS.relative_to(ROOT)}")
    print(f"classifications: {dict(by_cls)}")
    print(f"genuine sequence labels: {pools_out['n_genuine_sequence_labels']} "
          f"biggest pool: {pools_out['biggest_pool']}")
    for r in labels:
        print(f"  {r['theorem']:30s} {r['classification']:32s} "
              f"<- {r['minimal_closer']}")


if __name__ == "__main__":
    main()
