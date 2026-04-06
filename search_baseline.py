import argparse
import uuid
from dataclasses import dataclass

from lean_dojo import Dojo

from actions import ACTIONS
from env import make_repo, make_theorem, run_transition
from experiment_io import init_run_artifacts, write_metrics
from tasks import get_theorems


@dataclass
class Node:
    state: object | None
    history: list[str]
    finished: bool


def beam_search(theorem_cfg, beam_width: int = 16, max_depth: int = 4):
    repo = make_repo()
    theorem = make_theorem(repo, theorem_cfg)

    total_actions = 0
    total_success = 0
    found_any_proof = False

    with Dojo(theorem) as (dojo, init_state):
        print("=== Initial state ===")
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
                    total_actions += 1
                    outcome = run_transition(dojo, theorem, node.state, tac, step=depth, method="beam_search")
                    if outcome.is_error:
                        continue

                    total_success += 1
                    new_history = node.history + [tac]

                    if outcome.is_finished:
                        found_any_proof = True
                        print(f"FOUND PROOF at depth {depth} with history:")
                        for i, h in enumerate(new_history, start=1):
                            print(f"  {i}. {h}")
                        new_beam.append(Node(state=None, history=new_history, finished=True))
                    else:
                        print(
                            f"  OK: tactic `{tac}` produced new state "
                            f"with {outcome.next_state.num_goals} goal(s)."
                        )
                        new_beam.append(Node(state=outcome.next_state, history=new_history, finished=False))

            if not new_beam:
                print("No valid successors at this depth. Search stops.")
                break

            def score(node: Node):
                if node.finished:
                    return -1000
                return node.state.num_goals

            new_beam.sort(key=score)
            beam = new_beam[:beam_width]

            n_finished = sum(1 for n in beam if n.finished)
            print(f"Beam size after depth {depth}: {len(beam)}, finished nodes in beam: {n_finished}")

    print("\n==== Search Summary ====")
    print(f"Total tactic attempts: {total_actions}")
    print(f"Non-error transitions: {total_success}")
    if total_actions > 0:
        print(f"Approx. success rate: {total_success / total_actions:.3f}")
    print(f"Found any full proof? {'YES' if found_any_proof else 'NO'}")

    return {
        "total_tactic_attempts": total_actions,
        "non_error_transitions": total_success,
        "transition_success_rate": (total_success / total_actions) if total_actions else 0.0,
        "found_any_proof": found_any_proof,
    }


def main():
    parser = argparse.ArgumentParser(description="Beam search baseline.")
    parser.add_argument("--theorem-set", default="nat_single")
    parser.add_argument("--theorem-index", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--out-dir", default="runs")
    args = parser.parse_args()

    theorem_cfg = get_theorems(args.theorem_set)[args.theorem_index]
    run_id = f"baseline-{uuid.uuid4().hex[:8]}"
    artifacts = init_run_artifacts(
        base_dir=args.out_dir,
        method="beam_baseline",
        run_id=run_id,
        config={
            "method": "beam_baseline",
            "theorem_set": args.theorem_set,
            "theorem_index": args.theorem_index,
            "beam_width": args.beam_width,
            "max_depth": args.max_depth,
        },
    )

    metrics = beam_search(theorem_cfg, beam_width=args.beam_width, max_depth=args.max_depth)
    metrics.update({
        "run_id": run_id,
        "theorem": {"file_path": theorem_cfg.file_path, "full_name": theorem_cfg.full_name},
    })
    write_metrics(artifacts["metrics_path"], metrics)
    print(f"\nRun artifacts: {artifacts['run_dir']}")


if __name__ == "__main__":
    main()
