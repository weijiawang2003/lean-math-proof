"""Build a seq2seq dataset from proof traces for generative tactic models.

Unlike build_sft_dataset.py (which outputs label indices for a classifier),
this outputs (prompt, tactic_text) pairs suitable for fine-tuning a
sequence-to-sequence model like CodeT5 or GPT-2.

Usage:
  python build_seq2seq_dataset.py --in traces.jsonl --out seq2seq_data.jsonl
  python build_seq2seq_dataset.py --in traces.jsonl --out seq2seq_data.jsonl --min-goal-drop 1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from core_types import build_prompt
from trace_io import iter_jsonl


def build_seq2seq_dataset(
    in_path: str,
    out_path: str,
    min_goal_drop: int = 1,
    dedup_state_action: bool = True,
    max_per_tactic: int = 0,
    include_metadata: bool = False,
) -> dict:
    """Build seq2seq (prompt, tactic) pairs from trace transitions.

    Unlike the classifier SFT builder, this does NOT filter by action space.
    Any tactic that makes progress is a valid training target.

    Returns stats dict.
    """
    n_in = 0
    n_kept = 0
    n_drop_no_progress = 0
    n_drop_dedup = 0
    n_drop_tactic_cap = 0

    seen: set[tuple[str, str]] = set()
    tactic_counts: Counter = Counter()

    with open(out_path, "w", encoding="utf-8") as fout:
        for obj in iter_jsonl(in_path) or []:
            n_in += 1
            tactic = obj.get("tactic", "")
            if not tactic:
                continue

            proof_finished = obj.get("proof_finished", False)
            nb = obj.get("num_goals_before")
            na = obj.get("num_goals_after")

            # Keep if: proof finished, or goals decreased
            if not proof_finished:
                if nb is None or na is None or (nb - na) < min_goal_drop:
                    n_drop_no_progress += 1
                    continue

            state_pp = obj.get("state_pp", "")
            if dedup_state_action:
                key = (state_pp, tactic)
                if key in seen:
                    n_drop_dedup += 1
                    continue
                seen.add(key)

            if max_per_tactic > 0 and tactic_counts[tactic] >= max_per_tactic:
                n_drop_tactic_cap += 1
                continue

            tactic_counts[tactic] += 1
            n_kept += 1

            prompt = build_prompt(
                state_pp=state_pp,
                full_name=obj.get("full_name", ""),
            )
            row = {"prompt": prompt, "tactic": tactic}

            if include_metadata:
                row["meta"] = {
                    "file_path": obj.get("file_path"),
                    "full_name": obj.get("full_name"),
                    "run_id": obj.get("run_id"),
                    "step": obj.get("step"),
                    "proof_finished": proof_finished,
                    "goals_before": nb,
                    "goals_after": na,
                }

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "transitions_read": n_in,
        "kept": n_kept,
        "drop_no_progress": n_drop_no_progress,
        "drop_dedup": n_drop_dedup,
        "drop_tactic_cap": n_drop_tactic_cap,
        "unique_tactics": len(tactic_counts),
        "top_tactics": tactic_counts.most_common(20),
    }

    print(f"Read {n_in} transitions, kept {n_kept}, wrote to {out_path}")
    print(f"  no_progress={n_drop_no_progress}, dedup={n_drop_dedup}, tactic_cap={n_drop_tactic_cap}")
    print(f"  Unique tactics: {len(tactic_counts)}")
    print(f"  Top 10 tactics:")
    for tac, count in tactic_counts.most_common(10):
        print(f"    {count:4d}  {tac}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Build seq2seq dataset from traces.")
    parser.add_argument("--in", dest="in_path", default="traces_from_search.jsonl")
    parser.add_argument("--out", dest="out_path", default="seq2seq_data.jsonl")
    parser.add_argument("--min-goal-drop", type=int, default=1)
    parser.add_argument("--dedup-state-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-per-tactic", type=int, default=0,
                        help="Cap examples per tactic string (0 = unlimited).")
    parser.add_argument("--include-metadata", action="store_true")
    args = parser.parse_args()

    build_seq2seq_dataset(
        args.in_path,
        args.out_path,
        min_goal_drop=args.min_goal_drop,
        dedup_state_action=args.dedup_state_action,
        max_per_tactic=args.max_per_tactic,
        include_metadata=args.include_metadata,
    )


if __name__ == "__main__":
    main()
