import argparse
import uuid

from lean_dojo import Dojo

from env import make_repo, make_theorem, run_transition
from experiment_io import init_run_artifacts, write_metrics
from policy import Policy
from tasks import get_theorems
from trace_io import append_jsonl


def main():
    parser = argparse.ArgumentParser(description="Run policy rollout on a theorem.")
    parser.add_argument("--theorem-set", default="nat_single")
    parser.add_argument("--theorem-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--domain", default="mathlib4")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--ckpt-dir", default="clf_ckpt",
                        help="Path to classifier checkpoint directory.")
    args = parser.parse_args()

    theorems = get_theorems(args.theorem_set)
    if args.theorem_index < 0 or args.theorem_index >= len(theorems):
        raise SystemExit(
            f"--theorem-index {args.theorem_index} out of range "
            f"(set '{args.theorem_set}' has {len(theorems)} theorem(s), "
            f"valid indices: 0..{len(theorems) - 1})"
        )
    theorem_cfg = theorems[args.theorem_index]

    if args.max_steps < 1:
        raise SystemExit(f"--max-steps must be >= 1, got {args.max_steps}")

    # Instantiate policy (lazy-loads checkpoint on first call)
    pol = Policy(ckpt_dir=args.ckpt_dir)

    run_id = f"rollout-{uuid.uuid4().hex[:8]}"
    episode_id = f"{theorem_cfg.full_name}-0"
    artifacts = init_run_artifacts(
        base_dir=args.out_dir,
        method="policy_rollout",
        run_id=run_id,
        config={
            "method": "policy_rollout",
            "theorem_set": args.theorem_set,
            "theorem_index": args.theorem_index,
            "max_steps": args.max_steps,
            "domain": args.domain,
            "ckpt_dir": args.ckpt_dir,
        },
    )

    repo = make_repo()
    theorem = make_theorem(repo, theorem_cfg)

    num_steps = 0
    finished = False
    has_error = False

    with Dojo(theorem) as (dojo, state):
        print("=== initial ===")
        print(state.pp)

        for step in range(1, args.max_steps + 1):
            tac = pol.choose_tactic(state.pp, theorem.full_name)
            print(f"\nstep {step}, model chose tactic: {tac}")

            outcome = run_transition(
                dojo,
                theorem,
                state,
                tac,
                step=step,
                domain=args.domain,
                run_id=run_id,
                episode_id=episode_id,
                method="policy_rollout",
            )
            append_jsonl(artifacts["traces_path"], outcome.record)
            num_steps = step

            if outcome.is_finished:
                print("Proof finished ✅")
                finished = True
                break
            if outcome.is_error:
                print(f"Model tactic failed ❌: {outcome.record.error_message}")
                has_error = True
                break

            state = outcome.next_state
            print("New state:")
            print(state.pp)

        if not finished and not has_error and num_steps >= args.max_steps:
            print("Max steps reached, proof not finished.")

    metrics = {
        "run_id": run_id,
        "method": "policy_rollout",
        "episode_id": episode_id,
        "theorem": {"file_path": theorem_cfg.file_path, "full_name": theorem_cfg.full_name},
        "max_steps": args.max_steps,
        "num_steps": num_steps,
        "finished": finished,
        "has_error": has_error,
        "traces_path": artifacts["traces_path"],
        "ckpt_dir": args.ckpt_dir,
    }
    write_metrics(artifacts["metrics_path"], metrics)
    print(f"\nRun artifacts: {artifacts['run_dir']}")


if __name__ == "__main__":
    main()
