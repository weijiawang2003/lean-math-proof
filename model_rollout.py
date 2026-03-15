from lean_dojo import *
from policy import choose_tactic

REPO_URL = "https://github.com/leanprover-community/mathlib4"
COMMIT   = "29dcec074de168ac2bf835a77ef68bbe069194c5"
FILE_PATH = "Mathlib/Data/Nat/Defs.lean"
FULL_NAME = "Nat.mul_add_mod'"

MAX_STEPS = 5

def main():
    repo = LeanGitRepo(url=REPO_URL, commit=COMMIT)
    theorem = Theorem(repo, FILE_PATH, FULL_NAME)

    with Dojo(theorem) as (dojo, state):
        print("=== initial ===")
        print(state.pp)

        for step in range(1, MAX_STEPS + 1):
            tac = choose_tactic(state.pp, FULL_NAME)
            print(f"\nstep {step}, model chose tactic: {tac}")
            result = dojo.run_tac(state, tac)

            if isinstance(result, ProofFinished):
                print("Proof finished ✅")
                return
            elif isinstance(result, LeanError):
                print("Model tactic failed ❌:", result.message)
                return
            else:
                print("New state:")
                print(result.pp)
                state = result

        print("Max steps reached, proof not finished.")

if __name__ == "__main__":
    main()
