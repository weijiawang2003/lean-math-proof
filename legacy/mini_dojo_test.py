from lean_dojo import *

# 使用 LeanDojo Benchmark 4 的 mathlib4 版本
repo = LeanGitRepo(
    url="https://github.com/leanprover-community/mathlib4",
    commit="29dcec074de168ac2bf835a77ef68bbe069194c5",
)

theorem = Theorem(
    repo=repo,
    file_path="Mathlib/Data/Nat/Defs.lean",
    full_name="Nat.mul_add_mod'",
)

with Dojo(theorem) as (dojo, init_state):
    print("=== Initial state ===")
    # 直接打印 pretty-printed tactic state
    print(init_state.pp)
    print(f"#goals: {init_state.num_goals}")

    tactic = "rw [Nat.mul_comm, Nat.mul_add_mod]"
    print(f"\n>>> Running tactic: {tactic}")
    result = dojo.run_tac(init_state, tactic)

    print("\n=== Result ===")
    print(result)
