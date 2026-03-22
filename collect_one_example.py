import argparse
import uuid

from lean_dojo import Dojo

from env import make_repo, make_theorem, run_transition
from experiment_io import init_run_artifacts, write_metrics
from tasks import get_theorems
from trace_io import append_jsonl

TACTIC = "rw [Nat.mul_comm, Nat.mul_add_mod]"


def main():
    parser = argparse.ArgumentParser(description="Collect one transition example.")
    parser.add_argument("--out", default="")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--theorem-set", default="nat_single")
    parser.add_argument("--theorem-index", type=int, default=0)
    parser.add_argument("--tactic", default=TACTIC)
    args = parser.parse_args()

    theorem_cfg = get_theorems(args.theorem_set)[args.theorem_index]
    run_id = f"one-{uuid.uuid4().hex[:8]}"
    artifacts = init_run_artifacts(
        base_dir=args.out_dir,
        method="single_example",
        run_id=run_id,
        config={
            "method": "single_example",
            "theorem_set": args.theorem_set,
            "theorem_index": args.theorem_index,
            "tactic": args.tactic,
        },
    )
    trace_path = args.out or artifacts["traces_path"]

    repo = make_repo()
    theorem = make_theorem(repo, theorem_cfg)

    with Dojo(theorem) as (dojo, init_state):
        outcome = run_transition(
            dojo,
            theorem,
            init_state,
            args.tactic,
            step=1,
            run_id=run_id,
            episode_id=f"{theorem_cfg.full_name}-0",
            method="single_example",
        )
        append_jsonl(trace_path, outcome.record)

    write_metrics(
        artifacts["metrics_path"],
        {
            "run_id": run_id,
            "method": "single_example",
            "trace_path": trace_path,
            "finished": outcome.is_finished,
            "has_error": outcome.is_error,
        },
    )
    print(f"Run artifacts: {artifacts['run_dir']}")


if __name__ == "__main__":
    main()
