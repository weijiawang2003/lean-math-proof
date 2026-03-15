import json
from dataclasses import dataclass
from typing import List

from lean_dojo import *
from actions import ACTIONS

REPO_URL = "https://github.com/leanprover-community/mathlib4"
COMMIT   = "29dcec074de168ac2bf835a77ef68bbe069194c5"

# 目前先只用你已验证过的这个定理；以后可以往这里加更多简单定理
from dataclasses import dataclass
from typing import List

@dataclass
class TheoremConfig:
    file_path: str   # Lean 文件相对路径（相对 mathlib 根目录）
    full_name: str   # 完整定理名，能被 `Theorem(repo, file_path, full_name)` 找到

# 目前的最小宇宙：只一个定理，已经验证可用
THEOREMS: List[TheoremConfig] = [
    TheoremConfig(
        file_path="Mathlib/Data/Nat/Defs.lean",
        full_name="Nat.mul_add_mod'",
    ),
    TheoremConfig(
        file_path="Mathlib/Data/Nat/Basic.lean",
        full_name="Nat.add_mod",
    ),
    TheoremConfig(
        file_path="Mathlib/Data/Nat/Basic.lean",
        full_name="Nat.mul_mod",
    ),
    TheoremConfig(
        file_path="Mathlib/Data/Nat/Basic.lean",
        full_name="Nat.mod_add_mod",
    ),
    TheoremConfig(
        file_path="Mathlib/Data/Set/Basic.lean",
        full_name="Set.ite_univ", 
    ),
    TheoremConfig(
        file_path="athlib/Data/Finset/Basic.lean",
        full_name="Finset.disjoint_insert_right"
    )
]
OUT_PATH = "traces_from_search.jsonl"


@dataclass
class Node:
    state: TacticState | None
    history: List[str]
    finished: bool


def log_transition(theorem: Theorem, state: TacticState, tactic: str, result):
    """把一条 (state, tactic, result) 写进 JSONL。"""
    sample = {
        "file_path": str(theorem.file_path),
        "full_name": theorem.full_name,
        "state_pp": state.pp,
        "tactic": tactic,
        "result_kind": type(result).__name__,
        "proof_finished": isinstance(result, ProofFinished),
        "num_goals_before": state.num_goals,
        "num_goals_after": 0 if isinstance(result, ProofFinished) else result.num_goals,
    }
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def search_and_log_for_theorem(file_path: str, full_name: str,
                               beam_width: int = 16, max_depth: int = 4):
    repo = LeanGitRepo(url=REPO_URL, commit=COMMIT)
    theorem = Theorem(repo, file_path, full_name)

    with Dojo(theorem) as (dojo, init_state):
        print(f"\n=== Theorem: {full_name} ===")
        print(init_state.pp)
        print(f"#goals: {init_state.num_goals}\n")

        beam: List[Node] = [Node(state=init_state, history=[], finished=False)]

        for depth in range(1, max_depth + 1):
            print(f"\n==== Depth {depth} ====")
            new_beam: List[Node] = []

            for node in beam:
                if node.finished or node.state is None:
                    new_beam.append(node)
                    continue

                for tac in ACTIONS:
                    result = dojo.run_tac(node.state, tac)

                    if isinstance(result, LeanError):
                        # tactic 直接报错的就不记也不扩展
                        continue

                    # 只记录“目标数不变或减少”的步，过滤掉把局面搞更糟的
                    goals_before = node.state.num_goals
                    goals_after = 0 if isinstance(result, ProofFinished) else result.num_goals

                    if goals_after <= goals_before:
                        log_transition(theorem, node.state, tac, result)

                    new_history = node.history + [tac]

                    if isinstance(result, ProofFinished):
                        print(f"FOUND PROOF at depth {depth} with history {new_history}")
                        new_beam.append(Node(state=None, history=new_history, finished=True))
                    else:
                        print(f"  OK: `{tac}` : {goals_before} -> {goals_after} goals")
                        new_beam.append(Node(state=result, history=new_history, finished=False))

            if not new_beam:
                print("No valid successors, stop.")
                break

            def score(node: Node):
                if node.finished:
                    return -1000
                if node.state is None:
                    return 0
                return node.state.num_goals

            new_beam.sort(key=score)
            beam = new_beam[:beam_width]
            print(f"Beam size: {len(beam)}, finished: {sum(n.finished for n in beam)}")


def main():
    open(OUT_PATH, "w").close() 
    for th in THEOREMS:
        search_and_log_for_theorem(th.file_path, th.full_name)


if __name__ == "__main__":
    main()
