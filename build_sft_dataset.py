import json
from actions import ACTIONS

IN_PATH = "traces_from_search.jsonl"
OUT_PATH = "sft_dataset.jsonl"

def main():
    n_in = 0
    n_kept = 0
    n_out = 0

    with open(IN_PATH, "r", encoding="utf-8") as fin, \
         open(OUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            obj = json.loads(line)

            tactic = obj["tactic"]
            if tactic not in ACTIONS:
                continue

            proof_finished = obj.get("proof_finished", False)
            nb = obj.get("num_goals_before", None)
            na = obj.get("num_goals_after", None)

            # 关键过滤：只保留“goal 变少”或“证明结束”的步
            if not proof_finished:
                if nb is None or na is None:
                    continue
                if not (na < nb):
                    # 原地踏步（na == nb）或者更糟（na > nb）都丢掉
                    continue

            n_kept += 1

            state_pp = obj["state_pp"]
            full_name = obj.get("full_name", "")

            prompt = f"Theorem: {full_name}\n\nProof state:\n{state_pp}\n"
            label = ACTIONS.index(tactic)

            out = {
                "prompt": prompt,
                "label": label,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"read {n_in} transitions, kept {n_kept}, wrote {n_out} SFT samples to {OUT_PATH}")

if __name__ == "__main__":
    main()
