import json
from dataclasses import dataclass
from typing import List
from lean_dojo import *

# 固定任务域：mathlib4 @ 29dcec...
REPO_URL = "https://github.com/leanprover-community/mathlib4"
COMMIT   = "29dcec074de168ac2bf835a77ef68bbe069194c5"

@dataclass
class ProblemConfig:
    file_path: str
    full_name: str
    tactics: List[str]

# v0：最小题库，只放你已经验证过的超短证明
PROBLEMS: List[ProblemConfig] = [
    ProblemConfig(
        file_path="Mathlib/Data/Nat/Defs.lean",
        full_name="Nat.mul_add_mod'",
        tactics=[
            "rw [Nat.mul_comm, Nat.mul_add_mod]",
        ],
    ),

    # 以后你可以手动往这里加更多“人类 5–10 行 tactic 就能写完”的 lemma
    # 加之前先在 VSCode + Lean 里试好 tactic，再写到这里，保证不会破坏管线。
]

def collect_traces(out_path: str = "traces.jsonl"):
    repo = LeanGitRepo(url=REPO_URL, commit=COMMIT)

    for prob in PROBLEMS:
        theorem = Theorem(
            repo=repo,
            file_path=prob.file_path,
            full_name=prob.full_name,
        )
        print(f"\n=== Theorem: {prob.full_name} ===")

        with Dojo(theorem) as (dojo, state):
            for step_id, tac in enumerate(prob.tactics, start=1):
                print(f">>> step {step_id}, tactic: {tac}")
                result = dojo.run_tac(state, tac)

                sample = {
                    "domain": "mathlib4_nat_easy",
                    "file_path": str(theorem.file_path),
                    "full_name": theorem.full_name,
                    "step": step_id,
                    "state_pp": state.pp,
                    "tactic": tac,
                    "result_kind": type(result).__name__,
                    "proof_finished": isinstance(result, ProofFinished),
                }

                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

                if isinstance(result, ProofFinished):
                    print(f"step {step_id}: finished proof ✅")
                    break
                elif isinstance(result, LeanError):
                    print(f"step {step_id}: tactic failed ❌: {result.message}")
                    break
                else:
                    print(f"step {step_id}: state updated, continuing...")
                    state = result

if __name__ == "__main__":
    collect_traces("traces.jsonl")
