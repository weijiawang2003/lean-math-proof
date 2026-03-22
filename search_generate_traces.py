import argparse
import uuid
from dataclasses import dataclass

from lean_dojo import Dojo

from actions import ACTIONS
from env import make_repo, make_theorem, run_transition
from evaluate_traces import evaluate
from experiment_io import init_run_artifacts, write_metrics
from tasks import get_theorems
from trace_io import append_jsonl


@dataclass
class Node:
    state: object | None
    history: list[str]
    finished: bool


def search_and_log_for_theorem(cfg, out_path: str, run_id: str, beam_width: int = 16, max_depth: int = 4):
    repo = make_repo()
    theorem = make_theorem(repo, cfg)
    episode_id = cfg.full_name

    with Dojo(theorem) as (dojo, init_state):
        print(f"\n=== Theorem: {cfg.full_name} ===")
        print(init_state.pp)
        print(f"#goals: {init_state.num_goals}\n")

        beam: list[Node] = [Node(state=init_state, history=[], finished=False)]

        for depth in range(1, max_depth + 1):
            print(f"\n==== Depth {depth} ====")
            new_beam: list[Node] = []

            for node in beam:
                if node.finished or node.state is None:
                    new_beam.append(node)
                    continue

                for tac in ACTIONS:
                    outcome = run_transition(
                        dojo,
                        theorem,
                        node.state,
                        tac,
                        step=depth,
                        run_id=run_id,
                        episode_id=episode_id,
                        method="beam_search",
                    )
                    if outcome.is_error:
                        continue

                    goals_before = outcome.record.num_goals_before
                    goals_after = outcome.record.num_goals_after
                    if goals_before is not None and goals_after is not None and goals_after <= goals_before:
                        append_jsonl(out_path, outcome.record)

                    new_history = node.history + [tac]

                    if outcome.is_finished:
                        print(f"FOUND PROOF at depth {depth} with history {new_history}")
                        new_beam.append(Node(state=None, history=new_history, finished=True))
                    else:
                        print(f"  OK: `{tac}` : {goals_before} -> {goals_after} goals")
                        new_beam.append(Node(state=outcome.next_state, history=new_history, finished=False))

            if not new_beam:
                print("No valid successors, stop.")
                break

            def score(node: Node):
                if node.finished:
                    return -1000
                return node.state.num_goals

            new_beam.sort(key=score)
            beam = new_beam[:beam_width]
            print(f"Beam size: {len(beam)}, finished: {sum(n.finished for n in beam)}")


def main():
    parser = argparse.ArgumentParser(description="Generate traces from beam search.")
    parser.add_argument("--theorem-set", default="toy_search")
    parser.add_argument("--out", default="")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()

    run_id = f"search-{uuid.uuid4().hex[:8]}"
    artifacts = init_run_artifacts(
        base_dir=args.out_dir,
        method="beam_search",
        run_id=run_id,
        config={
            "method": "beam_search",
            "theorem_set": args.theorem_set,
            "beam_width": args.beam_width,
            "max_depth": args.max_depth,
        },
    )

    trace_path = args.out or artifacts["traces_path"]
    open(trace_path, "w", encoding="utf-8").close()

    for theorem in get_theorems(args.theorem_set):
        search_and_log_for_theorem(theorem, trace_path, run_id, args.beam_width, args.max_depth)

    metrics = evaluate(trace_path)
    write_metrics(artifacts["metrics_path"], metrics)
    print(f"\nRun artifacts: {artifacts['run_dir']}")


if __name__ == "__main__":
    main()
