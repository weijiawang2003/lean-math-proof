import argparse
import uuid
from dataclasses import dataclass

from lean_dojo import Dojo

from core_types import TheoremConfig
from env import make_repo, make_theorem, run_transition
from experiment_io import init_run_artifacts, write_metrics
from trace_io import append_jsonl


@dataclass(frozen=True)
class ProblemConfig:
    theorem: TheoremConfig
    tactics: list[str]


PROBLEMS: list[ProblemConfig] = [
    ProblemConfig(
        theorem=TheoremConfig(
            file_path="Mathlib/Data/Nat/Defs.lean",
            full_name="Nat.mul_add_mod'",
        ),
        tactics=["rw [Nat.mul_comm, Nat.mul_add_mod]"],
    ),
]


def collect_traces(out_path: str | None = None, domain: str = "mathlib4_nat_easy", out_dir: str = "runs"):
    run_id = f"collect-{uuid.uuid4().hex[:8]}"
    artifacts = init_run_artifacts(
        base_dir=out_dir,
        method="scripted_collect",
        run_id=run_id,
        config={
            "method": "scripted_collect",
            "domain": domain,
            "num_problems": len(PROBLEMS),
        },
    )
    trace_path = out_path or artifacts["traces_path"]

    repo = make_repo()
    episodes_total = 0
    episodes_finished = 0
    episodes_error = 0

    for idx, prob in enumerate(PROBLEMS):
        theorem = make_theorem(repo, prob.theorem)
        episode_id = f"episode-{idx}"
        episodes_total += 1
        print(f"\n=== Theorem: {prob.theorem.full_name} ===")

        with Dojo(theorem) as (dojo, state):
            for step_id, tac in enumerate(prob.tactics, start=1):
                print(f">>> step {step_id}, tactic: {tac}")
                outcome = run_transition(
                    dojo,
                    theorem,
                    state,
                    tac,
                    step=step_id,
                    domain=domain,
                    run_id=run_id,
                    episode_id=episode_id,
                    method="scripted_collect",
                )
                append_jsonl(trace_path, outcome.record)

                if outcome.is_finished:
                    print(f"step {step_id}: finished proof ✅")
                    episodes_finished += 1
                    break
                if outcome.is_error:
                    print(f"step {step_id}: tactic failed ❌: {outcome.record.error_message}")
                    episodes_error += 1
                    break

                print(f"step {step_id}: state updated, continuing...")
                state = outcome.next_state

    metrics = {
        "run_id": run_id,
        "method": "scripted_collect",
        "episodes_total": episodes_total,
        "episodes_finished": episodes_finished,
        "episodes_error": episodes_error,
        "trace_path": trace_path,
    }
    write_metrics(artifacts["metrics_path"], metrics)
    print(f"\nRun artifacts: {artifacts['run_dir']}")


def main():
    parser = argparse.ArgumentParser(description="Collect scripted traces.")
    parser.add_argument("--out", default="")
    parser.add_argument("--domain", default="mathlib4_nat_easy")
    parser.add_argument("--out-dir", default="runs")
    args = parser.parse_args()
    collect_traces(args.out or None, args.domain, args.out_dir)


if __name__ == "__main__":
    main()
