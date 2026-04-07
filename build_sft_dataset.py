import argparse
import json
from collections import Counter

from actions import get_action_space, list_action_spaces, load_action_space
from core_types import build_prompt
from trace_io import iter_jsonl


def _is_progress_sample(proof_finished: bool, nb: int | None, na: int | None, min_goal_drop: int) -> bool:
    if proof_finished:
        return True
    if nb is None or na is None:
        return False
    return (nb - na) >= min_goal_drop


def build_sft_dataset(
    in_path: str,
    out_path: str,
    include_metadata: bool = False,
    action_space_name: str = "core_v1",
    action_space_file: str = "",
    min_goal_drop: int = 1,
    dedup_state_action: bool = True,
    max_per_label: int = 0,
):
    n_in = 0
    n_kept = 0
    n_out = 0
    n_drop_unknown_action = 0
    n_drop_no_progress = 0
    n_drop_dedup = 0
    n_drop_label_cap = 0

    actions = load_action_space(action_space_file) if action_space_file else get_action_space(action_space_name)
    action_to_idx = {t: i for i, t in enumerate(actions)}
    seen_state_action: set[tuple[str, str]] = set()
    label_counts = Counter()

    with open(out_path, "w", encoding="utf-8") as fout:
        for obj in iter_jsonl(in_path) or []:
            n_in += 1
            tactic = obj["tactic"]
            if tactic not in action_to_idx:
                n_drop_unknown_action += 1
                continue

            proof_finished = obj.get("proof_finished", False)
            nb = obj.get("num_goals_before")
            na = obj.get("num_goals_after")

            if not _is_progress_sample(proof_finished, nb, na, min_goal_drop=min_goal_drop):
                n_drop_no_progress += 1
                continue

            state_pp = obj["state_pp"]
            if dedup_state_action:
                key = (state_pp, tactic)
                if key in seen_state_action:
                    n_drop_dedup += 1
                    continue
                seen_state_action.add(key)

            label = action_to_idx[tactic]
            if max_per_label > 0 and label_counts[label] >= max_per_label:
                n_drop_label_cap += 1
                continue

            n_kept += 1
            label_counts[label] += 1
            prompt = build_prompt(
                state_pp=state_pp,
                full_name=obj.get("full_name", ""),
            )
            out = {
                "prompt": prompt,
                "label": action_to_idx[tactic],
            }
            if include_metadata:
                out["meta"] = {
                    "file_path": obj.get("file_path"),
                    "full_name": obj.get("full_name"),
                    "run_id": obj.get("run_id"),
                    "episode_id": obj.get("episode_id"),
                    "method": obj.get("method"),
                    "step": obj.get("step"),
                    "action_space": action_space_name if not action_space_file else f"file:{action_space_file}",
                    "min_goal_drop": min_goal_drop,
                }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"read {n_in} transitions, kept {n_kept}, wrote {n_out} SFT samples to {out_path}")
    print(
        "drop summary: "
        f"unknown_action={n_drop_unknown_action}, "
        f"no_progress={n_drop_no_progress}, "
        f"dedup={n_drop_dedup}, "
        f"label_cap={n_drop_label_cap}, "
        f"action_space_size={len(actions)}, "
        f"min_goal_drop={min_goal_drop}, "
        f"dedup_state_action={dedup_state_action}, "
        f"max_per_label={max_per_label}"
    )


def main():
    parser = argparse.ArgumentParser(description="Build SFT dataset from transition traces.")
    parser.add_argument("--in", dest="in_path", default="traces_from_search.jsonl")
    parser.add_argument("--out", dest="out_path", default="sft_dataset.jsonl")
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument("--action-space", default="core_v1", choices=list_action_spaces())
    parser.add_argument("--action-space-file", default="", help="Optional JSON file with {'actions': [...]}.")
    parser.add_argument(
        "--min-goal-drop",
        type=int,
        default=1,
        help="Keep non-finished transitions only when (num_goals_before - num_goals_after) >= this value.",
    )
    parser.add_argument(
        "--dedup-state-action",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep at most one sample per (state_pp, tactic) pair.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=0,
        help="Cap samples per action label (0 means unlimited).",
    )
    args = parser.parse_args()
    build_sft_dataset(
        args.in_path,
        args.out_path,
        args.include_metadata,
        action_space_name=args.action_space,
        action_space_file=args.action_space_file,
        min_goal_drop=args.min_goal_drop,
        dedup_state_action=args.dedup_state_action,
        max_per_label=args.max_per_label,
    )


if __name__ == "__main__":
    main()
