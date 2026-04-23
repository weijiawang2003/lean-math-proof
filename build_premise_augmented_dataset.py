"""Build a premise-augmented seq2seq dataset for tactic generation.

This is the key training innovation from the ReProver insight: instead of
giving the model just the proof state, we also tell it which lemma names
are likely relevant.  The model then learns to *use* those names in its
generated tactics (e.g. `rw [Nat.add_comm]` instead of just `rw`).

The augmented prompt format:

    Relevant premises: Set.ext_iff, or_comm, Set.mem_union
    Theorem: Set.union_comm

    Proof state:
    α : Type u
    s t : Set α
    ⊢ s ∪ t = t ∪ s

This is 100% compatible with the existing train_tactic_generator.py —
the only change is the prompt text.  The model checkpoint, tokenizer,
and training loop are all unchanged.

Two premise sources at training time:
  1. Oracle premises: the actual lemma names referenced in the target tactic.
     These are the "ground truth" premises the model should learn to use.
  2. Retrieved premises: premises our retriever would have suggested for this
     state (simulating what happens at inference time).

We mix both: oracle premises ensure the model sees the right names during
training, while retrieved premises teach it to work with the noisy retriever
output it will get at test time.

Usage:
  python build_premise_augmented_dataset.py --in project/all_traces.jsonl --out project/seq2seq_premise_v1.jsonl
  python build_premise_augmented_dataset.py --in project/all_traces.jsonl --out project/seq2seq_premise_v1.jsonl --oracle-ratio 0.5
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from core_types import build_prompt
from premise_retriever import PremiseRetriever, extract_premises_from_tactic
from trace_io import iter_jsonl


def build_premise_augmented_prompt(
    state_pp: str,
    full_name: str,
    premises: list[str],
    max_premises: int = 10,
) -> str:
    """Build prompt with retrieved premises prepended.

    Format:
        Relevant premises: Foo.bar, Baz.qux, quux_comm
        Theorem: Some.theorem

        Proof state:
        ...
    """
    base = build_prompt(state_pp=state_pp, full_name=full_name)

    if premises:
        truncated = premises[:max_premises]
        prefix = "Relevant premises: " + ", ".join(truncated) + "\n"
        return prefix + base
    return base


def build_premise_augmented_dataset(
    in_path: str,
    out_path: str,
    retriever: PremiseRetriever,
    min_goal_drop: int = 1,
    dedup_state_action: bool = True,
    max_per_tactic: int = 0,
    oracle_ratio: float = 0.5,
    max_premises: int = 10,
    k_retrieved: int = 15,
    seed: int = 42,
) -> dict:
    """Build premise-augmented (prompt, tactic) pairs from trace transitions.

    For each training example:
      - Extract "oracle" premises from the target tactic
      - Retrieve premises using the retriever (simulating inference)
      - With probability oracle_ratio, use oracle premises (+ some retrieved)
      - Otherwise, use only retrieved premises

    This teaches the model two things:
      1. When given the right premise names, use them in the tactic
      2. When given noisy retriever output, still pick the useful ones

    Args:
        oracle_ratio: Fraction of examples where oracle premises are included.
            0.0 = pure retriever (test-time realistic), 1.0 = always oracle.
            Recommended: 0.5 (model learns both signal and noise tolerance).
    """
    rng = random.Random(seed)

    n_in = 0
    n_kept = 0
    n_with_oracle = 0
    n_with_retrieved_only = 0
    n_no_premises = 0
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
            full_name = obj.get("full_name", "")

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

            # ── Premise augmentation ──
            # 1. Extract oracle premises from the target tactic
            oracle_premises = extract_premises_from_tactic(tactic)

            # 2. Retrieve premises for this state (simulating inference)
            retrieved_premises = retriever.retrieve(state_pp, k=k_retrieved)

            # 3. Decide which premises to include
            use_oracle = rng.random() < oracle_ratio and oracle_premises

            if use_oracle:
                # Oracle mode: include ground-truth premises + some retrieved
                # Put oracle first so model learns they're most important
                combined = list(dict.fromkeys(oracle_premises + retrieved_premises))
                premises = combined[:max_premises]
                n_with_oracle += 1
            elif retrieved_premises:
                # Retriever-only mode: simulates test-time behavior
                premises = retrieved_premises[:max_premises]
                n_with_retrieved_only += 1
            else:
                premises = []
                n_no_premises += 1

            # Build augmented prompt
            prompt = build_premise_augmented_prompt(
                state_pp=state_pp,
                full_name=full_name,
                premises=premises,
                max_premises=max_premises,
            )

            row = {"prompt": prompt, "tactic": tactic}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "transitions_read": n_in,
        "kept": n_kept,
        "with_oracle_premises": n_with_oracle,
        "with_retrieved_only": n_with_retrieved_only,
        "no_premises": n_no_premises,
        "drop_no_progress": n_drop_no_progress,
        "drop_dedup": n_drop_dedup,
        "drop_tactic_cap": n_drop_tactic_cap,
        "unique_tactics": len(tactic_counts),
        "oracle_ratio": oracle_ratio,
        "max_premises": max_premises,
        "top_tactics": tactic_counts.most_common(20),
    }

    print(f"Read {n_in} transitions, kept {n_kept}")
    print(f"  Premise augmentation:")
    print(f"    with oracle:    {n_with_oracle}")
    print(f"    retrieved only: {n_with_retrieved_only}")
    print(f"    no premises:    {n_no_premises}")
    print(f"  Filtering:")
    print(f"    no_progress={n_drop_no_progress}, dedup={n_drop_dedup}, cap={n_drop_tactic_cap}")
    print(f"  Unique tactics: {len(tactic_counts)}")
    print(f"  Top 10 tactics:")
    for tac, count in tactic_counts.most_common(10):
        print(f"    {count:4d}  {tac}")
    print(f"\nWrote {n_kept} examples to {out_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build premise-augmented seq2seq dataset from traces."
    )
    parser.add_argument("--in", dest="in_path", default="project/all_traces.jsonl")
    parser.add_argument("--out", dest="out_path", default="project/seq2seq_premise_v1.jsonl")
    parser.add_argument("--premise-index", default="project/premise_index.json",
                        help="Path to premise retriever index (from build-premise-index).")
    parser.add_argument("--traces-for-index", default="",
                        help="Traces to build index from (default: same as --in).")
    parser.add_argument("--min-goal-drop", type=int, default=1)
    parser.add_argument("--dedup-state-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-per-tactic", type=int, default=0)
    parser.add_argument("--oracle-ratio", type=float, default=0.5,
                        help="Fraction of examples with oracle (ground-truth) premises. "
                             "0.0 = pure retriever, 1.0 = always oracle.")
    parser.add_argument("--max-premises", type=int, default=10)
    parser.add_argument("--k-retrieved", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load or build premise retriever
    from pathlib import Path
    retriever = PremiseRetriever()
    if Path(args.premise_index).exists():
        retriever.load_index(args.premise_index)
        print(f"Loaded premise index from {args.premise_index}")
    else:
        traces_for_idx = args.traces_for_index or args.in_path
        print(f"Building premise index from {traces_for_idx}...")
        retriever.build_index_from_traces(traces_for_idx)
        retriever.save_index(args.premise_index)
        print(f"Saved premise index to {args.premise_index}")

    build_premise_augmented_dataset(
        in_path=args.in_path,
        out_path=args.out_path,
        retriever=retriever,
        min_goal_drop=args.min_goal_drop,
        dedup_state_action=args.dedup_state_action,
        max_per_tactic=args.max_per_tactic,
        oracle_ratio=args.oracle_ratio,
        max_premises=args.max_premises,
        k_retrieved=args.k_retrieved,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
