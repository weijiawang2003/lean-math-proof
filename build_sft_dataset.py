import argparse
import json

from actions import get_action_space, list_action_spaces, load_action_space
from core_types import build_prompt
from trace_io import iter_jsonl


def build_sft_dataset(
    in_path: str,
    out_path: str,
    include_metadata: bool = False,
    action_space_name: str = "core_v1",
    action_space_file: str = "",
):
    n_in = 0
    n_kept = 0
    n_out = 0
    n_drop_unknown_action = 0
    n_drop_no_progress = 0

    actions = load_action_space(action_space_file) if action_space_file else get_action_space(action_space_name)
    action_to_idx = {t: i for i, t in enumerate(actions)}

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

            if not proof_finished and (nb is None or na is None or not (na < nb)):
                n_drop_no_progress += 1
                continue

            n_kept += 1
            prompt = build_prompt(
                state_pp=obj["state_pp"],
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
                }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"read {n_in} transitions, kept {n_kept}, wrote {n_out} SFT samples to {out_path}")
    print(
        "drop summary: "
        f"unknown_action={n_drop_unknown_action}, "
        f"no_progress={n_drop_no_progress}, "
        f"action_space_size={len(actions)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Build SFT dataset from transition traces.")
    parser.add_argument("--in", dest="in_path", default="traces_from_search.jsonl")
    parser.add_argument("--out", dest="out_path", default="sft_dataset.jsonl")
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument("--action-space", default="core_v1", choices=list_action_spaces())
    parser.add_argument("--action-space-file", default="", help="Optional JSON file with {'actions': [...]}.")
    args = parser.parse_args()
    build_sft_dataset(
        args.in_path,
        args.out_path,
        args.include_metadata,
        action_space_name=args.action_space,
        action_space_file=args.action_space_file,
    )


if __name__ == "__main__":
    main()
