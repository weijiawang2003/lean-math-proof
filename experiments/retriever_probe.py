"""Retriever-quality probe (Experiment D).

For every theorem in project_state.json that has a verified proof, this script:
  1. Reconstructs the proof state Lean would have shown the policy
     (loads via LeanDojo only when a state is not cached in traces; otherwise
     falls back to scanning all_traces.jsonl by full_name + step==1).
  2. Extracts the lemma names actually used in the winning tactic.
  3. Asks PremiseRetriever to retrieve premises for that state.
  4. Records whether each ground-truth premise appears in the top-1/5/10/15.

Outputs a JSON report grouped by source file (Set/Basic, Finset/Basic, Nat/Defs,
others) with mean recall@k and per-theorem records for inspection.

This is pure analysis — no Lean execution, no GPU.  Runs in seconds-to-minutes.

Usage:
    python experiments/retriever_probe.py \
        --state project/project_state.json \
        --traces project/all_traces.jsonl \
        --premise-index project/premise_index.json \
        --out runs/retriever_probe.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import sys
# Make repo root importable when invoked from anywhere
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from premise_retriever import PremiseRetriever, extract_premises_from_tactic


# ── State reconstruction from traces (no Lean execution needed) ─────

def _build_state_index(traces_path: Path) -> dict[str, str]:
    """Map theorem full_name -> first-step proof state seen in traces.

    The first-step state for a theorem is what the policy sees when it makes
    its initial tactic decision, which is what we want to test the retriever
    on.  We pick the row with smallest step (typically 1) per theorem.
    """
    index: dict[str, tuple[int, str]] = {}
    if not traces_path.exists():
        return {}
    with open(traces_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            full_name = rec.get("full_name") or ""
            state = rec.get("state_pp") or ""
            step = int(rec.get("step", 1) or 1)
            if not full_name or not state:
                continue
            cur = index.get(full_name)
            if cur is None or step < cur[0]:
                index[full_name] = (step, state)
    return {k: v[1] for k, v in index.items()}


def _file_bucket(file_path: str) -> str:
    if "Set/Basic" in file_path:
        return "Set.Basic"
    if "Finset/Basic" in file_path:
        return "Finset.Basic"
    if "Nat/Defs" in file_path:
        return "Nat.Defs"
    if "Nat/Basic" in file_path:
        return "Nat.Basic"
    return "other"


def main():
    parser = argparse.ArgumentParser(description="Retriever-quality probe (D).")
    parser.add_argument("--state", default="project/project_state.json",
                        help="Path to project_state.json with proven theorems.")
    parser.add_argument("--traces", default="project/all_traces.jsonl",
                        help="Path to all_traces.jsonl for state reconstruction.")
    parser.add_argument("--premise-index", default="project/premise_index.json",
                        help="Path to a prebuilt premise index (built from traces).")
    parser.add_argument("--out", default="runs/retriever_probe.json",
                        help="Where to write the JSON report.")
    parser.add_argument("--k-retrieve", type=int, default=15,
                        help="Top-k retrieved premises to fetch per state.")
    args = parser.parse_args()

    state_path = Path(args.state)
    traces_path = Path(args.traces)
    index_path = Path(args.premise_index)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load proven theorems ----
    state_data = json.loads(state_path.read_text(encoding="utf-8"))
    theorems = state_data.get("theorems", {})
    proved = [
        (name, meta) for name, meta in theorems.items()
        if meta.get("proved") and meta.get("proof_tactics")
    ]
    print(f"[probe] {len(proved)} proven theorems in {state_path}")

    # ---- Build per-theorem first-state index from traces ----
    state_by_name = _build_state_index(traces_path)
    print(f"[probe] {len(state_by_name)} unique theorem names indexed from traces")

    # ---- Load retriever ----
    retriever = PremiseRetriever()
    if index_path.exists():
        retriever.load_index(str(index_path))
        # retriever.load_index() doesn't restore _premise_contexts — for the
        # probe we mostly need _all_premises + static catalog matching, which
        # load_index handles.  If we want full BM25 scoring, fall back to
        # rebuilding from traces.
    else:
        print(f"[probe] No index at {index_path}, rebuilding from traces")
        retriever.build_index_from_traces(str(traces_path))

    # ---- Per-theorem evaluation ----
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    no_state = 0
    no_premises = 0

    for full_name, meta in proved:
        file_path = meta.get("file_path") or ""
        tactics = meta.get("proof_tactics") or ""
        bucket = _file_bucket(file_path)

        ground_truth = extract_premises_from_tactic(tactics)
        if not ground_truth:
            # Tactic didn't reference any named premises (e.g., just `aesop` or
            # `tauto`).  These don't test the retriever — skip them but count.
            no_premises += 1
            continue

        state = state_by_name.get(full_name)
        if not state:
            no_state += 1
            continue

        retrieved = retriever.retrieve(state, k=args.k_retrieve)
        retrieved_set_topk = {
            1: set(retrieved[:1]),
            5: set(retrieved[:5]),
            10: set(retrieved[:10]),
            15: set(retrieved[:15]),
        }

        # Per-premise hits at each k
        per_premise_ranks: dict[str, int | None] = {}
        for gt in ground_truth:
            try:
                rank = retrieved.index(gt) + 1  # 1-indexed
            except ValueError:
                rank = None
            per_premise_ranks[gt] = rank

        # Recall@k = fraction of ground-truth premises present in top-k
        recall_at = {
            f"recall@{k}": (
                sum(1 for gt in ground_truth if gt in retrieved_set_topk[k])
                / len(ground_truth)
            )
            for k in (1, 5, 10, 15)
        }

        by_bucket[bucket].append({
            "full_name": full_name,
            "file_path": file_path,
            "winning_tactic": tactics,
            "ground_truth_premises": ground_truth,
            "retrieved_top15": retrieved,
            "per_premise_ranks": per_premise_ranks,
            **recall_at,
        })

    # ---- Aggregate per bucket ----
    summary: dict[str, dict] = {}
    overall_records: list[dict] = []
    for bucket, records in by_bucket.items():
        if not records:
            continue
        n = len(records)
        agg = {
            "n_theorems": n,
            "mean_recall@1": sum(r["recall@1"] for r in records) / n,
            "mean_recall@5": sum(r["recall@5"] for r in records) / n,
            "mean_recall@10": sum(r["recall@10"] for r in records) / n,
            "mean_recall@15": sum(r["recall@15"] for r in records) / n,
        }
        summary[bucket] = agg
        overall_records.extend(records)

    if overall_records:
        n = len(overall_records)
        summary["__OVERALL__"] = {
            "n_theorems": n,
            "mean_recall@1": sum(r["recall@1"] for r in overall_records) / n,
            "mean_recall@5": sum(r["recall@5"] for r in overall_records) / n,
            "mean_recall@10": sum(r["recall@10"] for r in overall_records) / n,
            "mean_recall@15": sum(r["recall@15"] for r in overall_records) / n,
        }

    report = {
        "config": {
            "state_path": str(state_path),
            "traces_path": str(traces_path),
            "premise_index": str(index_path),
            "k_retrieve": args.k_retrieve,
        },
        "counts": {
            "total_proved": len(proved),
            "evaluated": sum(len(v) for v in by_bucket.values()),
            "skipped_no_named_premises": no_premises,
            "skipped_no_state_in_traces": no_state,
        },
        "summary_by_bucket": summary,
        "per_theorem": {bucket: records for bucket, records in by_bucket.items()},
    }

    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[probe] Wrote {out_path}")

    # Console preview
    print()
    print("=" * 64)
    print("  RETRIEVER QUALITY PROBE — SUMMARY")
    print("=" * 64)
    print(f"  Total proved: {len(proved)}")
    print(f"  Evaluable (named premises + state available): "
          f"{sum(len(v) for v in by_bucket.values())}")
    print(f"  Skipped — no named premises in tactic:  {no_premises}")
    print(f"  Skipped — no first-state in traces:     {no_state}")
    print()
    print(f"  {'Bucket':<20} {'N':>4} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@15':>8}")
    print(f"  {'-' * 20} {'-' * 4} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for bucket, agg in summary.items():
        print(f"  {bucket:<20} {agg['n_theorems']:>4d} "
              f"{agg['mean_recall@1']:>7.1%} "
              f"{agg['mean_recall@5']:>7.1%} "
              f"{agg['mean_recall@10']:>7.1%} "
              f"{agg['mean_recall@15']:>7.1%}")
    print("=" * 64)


if __name__ == "__main__":
    main()
