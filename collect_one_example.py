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

tactic = "rw [Nat.mul_comm, Nat.mul_add_mod]"

with Dojo(theorem) as (dojo, init_state):
    result = dojo.run_tac(init_state, tactic)

    sample = {
        "repo_url": repo.url,
        "commit": repo.commit,
        "file_path": str(theorem.file_path),      # <- convert PosixPath to str
        "full_name": theorem.full_name,
        "state_pp": init_state.pp,                # tactic state as plain text
        "tactic": tactic,
        "result_kind": type(result).__name__,     # e.g. "ProofFinished"
        "proof_finished": isinstance(result, ProofFinished),
    }

    with open("toy_dataset.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("wrote one example to toy_dataset.jsonl")
