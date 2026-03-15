import json
from dataclasses import dataclass
from typing import List

from lean_dojo import *


# 固定仓库：你已经下载好的 mathlib4 + Benchmark 4 commit
REPO_URL = "https://github.com/leanprover-community/mathlib4"
COMMIT   = "29dcec074de168ac2bf835a77ef68bbe069194c5"

# 目标定理：Nat.mul_add_mod'
FILE_PATH = "Mathlib/Data/Nat/Defs.lean"
FULL_NAME = "Nat.mul_add_mod'"

# -------- Step 2: 定义动作空间（tactic 子集） --------

ACTIONS: List[str] = [
    # 通用基础动作（这里多数会失败，但用来制造“99% 失败”的效果）
    "intro",
    "refl",
    "rfl",
    "simp",
    "assumption",
    "cases a",
    "induction a",

    # 与本题相关的“高价值动作”
    "rw [Nat.mul_add_mod]",
    "rw [Nat.mul_comm]",
    "rw [Nat.mul_comm, Nat.mul_add_mod]",
    "rw [Nat.mul_add_mod, Nat.mul_comm]",
    "simp [Nat.mul_add_mod, Nat.mul_comm]",
]

# -------- 搜索节点定义 --------

@dataclass
class Node:
    state: TacticState | None   # ProofFinished 时可以是 None
    history: List[str]
    finished: bool


def beam_search(
    repo_url: str,
    commit: str,
    file_path: str,
    full_name: str,
    beam_width: int = 16,
    max_depth: int = 4,
):
    repo = LeanGitRepo(url=repo_url, commit=commit)
    theorem = Theorem(repo, file_path, full_name)

    total_actions = 0
    total_success = 0   # 非 LeanError 的 step 数

    with Dojo(theorem) as (dojo, init_state):
        print("=== Initial state ===")
        print(init_state.pp)
        print(f"#goals: {init_state.num_goals}\n")

        # beam 初始化：只有一个节点
        beam: List[Node] = [Node(state=init_state, history=[], finished=False)]
        found_any_proof = False

        for depth in range(1, max_depth + 1):
            print(f"\n==== Depth {depth} ====")
            new_beam: List[Node] = []

            for node in beam:
                if node.finished or node.state is None:
                    # 已经证明完成的节点直接保留
                    new_beam.append(node)
                    continue

                for tac in ACTIONS:
                    total_actions += 1
                    result = dojo.run_tac(node.state, tac)

                    # tactic 不合法 / 爆 LeanError：跳过
                    if isinstance(result, LeanError):
                        continue

                    total_success += 1
                    new_history = node.history + [tac]

                    if isinstance(result, ProofFinished):
                        # 找到一条完整证明路径
                        found_any_proof = True
                        print(f"FOUND PROOF at depth {depth} with history:")
                        for i, h in enumerate(new_history, start=1):
                            print(f"  {i}. {h}")
                        # 我们仍然把这个节点丢进 beam，看后面还会不会出现别的路径
                        new_beam.append(Node(state=None, history=new_history, finished=True))
                    else:
                        # 得到一个新的中间 proof state
                        print(f"  OK: tactic `{tac}` produced new state with {result.num_goals} goal(s).")
                        new_beam.append(Node(state=result, history=new_history, finished=False))

            if not new_beam:
                print("No valid successors at this depth. Search stops.")
                break

            # 简单打分：优先保留 finished，其次 goals 数少的
            def score(node: Node):
                if node.finished:
                    return -1000
                if node.state is None:
                    return 0
                return node.state.num_goals

            new_beam.sort(key=score)
            # 截断到 beam_width
            beam = new_beam[:beam_width]

            n_finished = sum(1 for n in beam if n.finished)
            print(f"Beam size after depth {depth}: {len(beam)}, finished nodes in beam: {n_finished}")

        print("\n==== Search Summary ====")
        print(f"Total tactic attempts: {total_actions}")
        print(f"Non-error transitions: {total_success}")
        if total_actions > 0:
            success_rate = total_success / total_actions
            print(f"Approx. success rate: {success_rate:.3f}")
            ##预期应该很低，‘99% 失败’是正常的
        print(f"Found any full proof? {'YES' if found_any_proof else 'NO'}")


if __name__ == "__main__":
    beam_search(
        repo_url=REPO_URL,
        commit=COMMIT,
        file_path=FILE_PATH,
        full_name=FULL_NAME,
        beam_width=16,
        max_depth=4,
    )
