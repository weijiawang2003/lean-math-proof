import json
from lean_dojo import *

repo = LeanGitRepo(
    url="https://github.com/leanprover-community/mathlib4",
    commit="29dcec074de168ac2bf835a77ef68bbe069194c5",
)

theorem = Theorem(
    repo=repo,
    file_path="Mathlib/Data/Nat/Defs.lean",
    full_name="Nat.mul_add_mod'",
)

# 对这个定理暂时只有一条 tactic 序列，不过接口写成可以扩展
TACTICS = [
    "rw [Nat.mul_comm, Nat.mul_add_mod]",
]

out_path = "toy_trace.jsonl"

with Dojo(theorem) as (dojo, state):
    for step_id, tac in enumerate(TACTICS, start=1):
        result = dojo.run_tac(state, tac)

        sample = {
            "step": step_id,
            "file_path": str(theorem.file_path),
            "full_name": theorem.full_name,
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
            # result 是新的 TacticState，继续下一步
            print(f"step {step_id}: state updated, continuing…")
            state = result
